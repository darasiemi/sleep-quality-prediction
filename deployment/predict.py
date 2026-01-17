"""
FastAPI application for Sleep Quality Prediction Model
"""

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from utils.build_lagged_features import build_lagged_features
from utils.load_model import load_model as load_model_from_disk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sleep Quality Prediction API",
    description="API for predicting sleep quality based on historical sleep and activity data",
    version="1.0.0",
)

# Global variables for model and configuration
pipeline = None
LAGS = (1, 2, 3, 7)
TARGET = "Sleep quality"
BASE_VARS = ("time_in_minutes", "Activity (steps)", "sleep_timing_bin", "Day")
TIME_COL = "Start"
MODEL_PATH = "model/sleep_model_train.bin"


class SleepRecord(BaseModel):
    """Single sleep record for input"""

    Start: str = Field(
        ..., description="Start datetime in ISO format (e.g., '2024-01-15T22:30:00')"
    )
    Sleep_quality: float = Field(
        ..., alias="Sleep quality", description="Sleep quality rating"
    )
    time_in_minutes: float = Field(..., description="Sleep duration in minutes")
    Activity_steps: int = Field(
        ..., alias="Activity (steps)", description="Number of steps"
    )
    sleep_timing_bin: str = Field(..., description="Sleep timing bin category")
    Day: str = Field(..., description="Day of week (0=Monday, 6=Sunday)")


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint"""

    history: List[SleepRecord] = (
        Field(
            ...,
            min_items=8,
            description="Historical sleep records (minimum 8 records required for lag features)",
        ),
    )
    timestamp: datetime = Field(..., description="Timestamp when prediction was made")


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint"""

    prediction_timestamp: datetime = Field(
        ..., description="Timestamp when prediction was made"
    )
    predicted_sleep_quality: float = Field(..., description="Predicted sleep quality")


class HealthResponse(BaseModel):
    """Response for health check endpoint"""

    status: str
    model_loaded: bool
    timestamp: str


def ensure_model_loaded() -> None:
    """
    Load the model into the global `pipeline` if it isn't already loaded.
    """
    global pipeline

    if pipeline is not None:
        return

    try:
        pipeline = load_model_from_disk(MODEL_PATH)
        logger.info("Model loaded successfully (ensure_model_loaded)")
    except FileNotFoundError:
        logger.warning(f"Model file not found at {MODEL_PATH}")
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Please ensure '{MODEL_PATH}' is available. 1",
        )
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded due to an internal error: {str(e)}",
        )


@app.on_event("startup")
def load_model():
    global pipeline

    """Load the trained model on startup"""
    try:
        model_path = "model/sleep_model_train.bin"
        pipeline = load_model_from_disk(MODEL_PATH)
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.warning(f"Model file not found. Please upload {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sleep Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "predict": "/predict", "docs": "/docs"},
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if pipeline is not None else "model_not_loaded",
        model_loaded=pipeline is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_sleep_quality(request: PredictionRequest):
    """Predict sleep quality for the next night based on historical data."""
    ensure_model_loaded()

    # print("request",request)

    try:
        history_data = []
        for record in request.history:
            history_data.append(
                {
                    "Start": record.Start,
                    "Sleep quality": record.Sleep_quality,
                    "time_in_minutes": record.time_in_minutes,
                    "Activity (steps)": record.Activity_steps,
                    "sleep_timing_bin": record.sleep_timing_bin,
                    "Day": record.Day,
                }
            )

        history_df = pd.DataFrame(history_data)
        history_df["Start"] = pd.to_datetime(history_df["Start"])

        X_features, y, lag_num, cat_num = build_lagged_features(history_df)

        feature_cols = [
            "Sleep quality_lag1",
            "Sleep quality_lag2",
            "Sleep quality_lag3",
            "Sleep quality_lag7",
            "time_in_minutes_lag1",
            "time_in_minutes_lag2",
            "time_in_minutes_lag3",
            "time_in_minutes_lag7",
            "Activity (steps)_lag1",
            "Activity (steps)_lag2",
            "Activity (steps)_lag3",
            "Activity (steps)_lag7",
            "sleep_timing_bin_lag1",
            "sleep_timing_bin_lag2",
            "sleep_timing_bin_lag3",
            "sleep_timing_bin_lag7",
            "Day_lag1",
            "Day_lag2",
            "Day_lag3",
            "Day_lag7",
        ]

        X_pred = X_features[feature_cols]
        prediction = float(pipeline.predict(X_pred)[0])

        logger.info(f"Prediction made: {prediction}")

        return PredictionResponse(
            prediction_timestamp=request.timestamp,
            predicted_sleep_quality=round(prediction, 2),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
