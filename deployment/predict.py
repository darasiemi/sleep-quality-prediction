"""
FastAPI application for Sleep Quality Prediction Model
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import pandas as pd
from datetime import datetime
import logging
import uvicorn
from utils.build_lagged_features import build_lagged_features
from utils.load_model import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sleep Quality Prediction API",
    description="API for predicting sleep quality based on historical sleep and activity data",
    version="1.0.0"
)

# Global variables for model and configuration
pipeline = None
LAGS = (1, 2, 3, 7)
TARGET = "Sleep quality"
BASE_VARS = ("time_in_minutes", "Activity (steps)", "sleep_timing_bin", "Day")
TIME_COL = "Start"

class SleepRecord(BaseModel):
    """Single sleep record for input"""
    Start: str = Field(..., description="Start datetime in ISO format (e.g., '2024-01-15T22:30:00')")
    Sleep_quality: float = Field(..., alias="Sleep quality", description="Sleep quality rating")
    time_in_minutes: float = Field(..., description="Sleep duration in minutes")
    Activity_steps: int = Field(..., alias="Activity (steps)", description="Number of steps")
    sleep_timing_bin: int = Field(..., description="Sleep timing bin category")
    Day: int = Field(..., description="Day of week (0=Monday, 6=Sunday)")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Start": "2024-01-15T22:30:00",
                "Sleep quality": 4.5,
                "time_in_minutes": 450.0,
                "Activity (steps)": 8000,
                "sleep_timing_bin": 2,
                "Day": 1
            }
        }

class PredictionRequest(BaseModel):
    """Request body for prediction endpoint"""
    history: List[SleepRecord] = Field(
        ..., 
        min_items=8,
        description="Historical sleep records (minimum 8 records required for lag features)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "history": [
                    {
                        "Start": "2024-01-08T22:30:00",
                        "Sleep quality": 4.0,
                        "time_in_minutes": 420.0,
                        "Activity (steps)": 7500,
                        "sleep_timing_bin": 2,
                        "Day": 0
                    },
                    # ... more records
                ]
            }
        }

class PredictionResponse(BaseModel):
    """Response body for prediction endpoint"""
    predicted_sleep_quality: float = Field(..., description="Predicted sleep quality")
    prediction_timestamp: str = Field(..., description="Timestamp when prediction was made")
    input_records_count: int = Field(..., description="Number of historical records used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_sleep_quality": 4.2,
                "prediction_timestamp": "2024-01-16T10:30:00",
                "input_records_count": 8
            }
        }

class HealthResponse(BaseModel):
    """Response for health check endpoint"""
    status: str
    model_loaded: bool
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global pipeline
    try:
        model_path = "model/sleep_model_train.bin"
        pipeline  = load_model(model_path)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning("Model file not found. Please upload 'sleep_quality_model.pkl'")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sleep Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if pipeline is not None else "model_not_loaded",
        model_loaded=pipeline is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_sleep_quality(request: PredictionRequest):
    """
    Predict sleep quality for the next night based on historical data.
    
    The endpoint uses the provided historical records to create lag features
    and predicts the sleep quality for the upcoming night.
    
    Args:
        request: PredictionRequest containing historical sleep records
        
    Returns:
        PredictionResponse with predicted sleep quality
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure 'sleep_model_train.bin' is available."
        )
    
    try:
        # Convert request to DataFrame
        history_data = []
        for record in request.history:
            history_data.append({
                "Start": record.Start,
                "Sleep quality": record.Sleep_quality,
                "time_in_minutes": record.time_in_minutes,
                "Activity (steps)": record.Activity_steps,
                "sleep_timing_bin": record.sleep_timing_bin,
                "Day": record.Day
            })
        
        history_df = pd.DataFrame(history_data)
        
        # Convert Start to datetime
        history_df["Start"] = pd.to_datetime(history_df["Start"])
        
        # Build lag features
        X_features,y, lag_num, cat_num = build_lagged_features(history_df)
        
        # Expected feature columns from training
        feature_cols = [
            'Sleep quality_lag1', 'Sleep quality_lag2', 'Sleep quality_lag3',
            'Sleep quality_lag7', 'time_in_minutes_lag1', 'time_in_minutes_lag2',
            'time_in_minutes_lag3', 'time_in_minutes_lag7', 'Activity (steps)_lag1',
            'Activity (steps)_lag2', 'Activity (steps)_lag3',
            'Activity (steps)_lag7', 'sleep_timing_bin_lag1',
            'sleep_timing_bin_lag2', 'sleep_timing_bin_lag3',
            'sleep_timing_bin_lag7', 'Day_lag1', 'Day_lag2', 'Day_lag3',
            'Day_lag7'
        ]
        
        # Select only the features used in training
        X_pred = X_features[feature_cols]
        
        # Make prediction
        prediction = float(pipeline.predict(X_pred)[0])
        
        logger.info(f"Prediction made: {prediction}")
        
        return PredictionResponse(
            predicted_sleep_quality=round(prediction, 2),
            prediction_timestamp=datetime.now().isoformat(),
            input_records_count=len(request.history)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)