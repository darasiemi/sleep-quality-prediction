"""
Test script for Sleep Quality Prediction API
"""

import json
from datetime import datetime, timedelta

import requests

from deployment.predict_one import feature_cols
from utils.build_lagged_features import build_lagged_features
from utils.load_data import load_data
from utils.load_model import load_model
from utils.preprocess import preprocess
from utils.split_data import train_test_split

# API base URL
# BASE_URL = "http://localhost:9696"
BASE_URL = "http://localhost:30080"


def test_root():
    """Test root endpoint"""
    print("\n" + "=" * 60)
    print("Testing Root Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_health():
    """Test health check endpoint"""
    print("\n" + "=" * 60)
    print("Testing Health Check Endpoint")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_predict():
    """Test prediction endpoint with sample data"""
    print("\n" + "=" * 60)
    print("Testing Prediction Endpoint")
    print("=" * 60)

    data_url = "archive/preprocessed_data.csv"
    parse_cols = ["Start", "End"]
    df = load_data(data_url, parse_cols)

    df = preprocess(df)

    df_train, df_test = train_test_split(df)

    # Get 8 consecutive records from test set
    X_prev_8 = df_test.iloc[8:16].copy()

    # Convert to API format
    history = []

    for _, row in X_prev_8.iterrows():
        record = {
            "Start": row["Start"].isoformat(),
            "Sleep quality": float(row["Sleep quality"]),
            "time_in_minutes": float(row["time_in_minutes"]),
            "Activity (steps)": int(row["Activity (steps)"]),
            "sleep_timing_bin": str(row["sleep_timing_bin"]),
            "Day": str(row["Day"]),
        }
        history.append(record)

    payload = {
        "history": history,
        "timestamp": df_test["Start"]
        .iloc[16]
        .isoformat(),  # Note we select the next row not in our feature to prevent data leakage
    }

    # X_single  = X_single[feature_cols]

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    # # print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print(f"\n Predicted Sleep Quality: {result['predicted_sleep_quality']}")
    else:
        print(f"Error: {response.text}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")


def test_predict_insufficient_data():
    """Test prediction with insufficient historical data"""
    print("\n" + "=" * 60)
    print("Testing Prediction with Insufficient Data")
    print("=" * 60)

    # Only 3 records (less than required 8)
    history = [
        {
            "Start": "2024-01-08T22:30:00",
            "Sleep quality": 4.0,
            "time_in_minutes": 420.0,
            "Activity (steps)": 7000,
            "sleep_timing_bin": "Early",
            "Day": "Sunday",
        },
        {
            "Start": "2024-01-09T22:30:00",
            "Sleep quality": 4.5,
            "time_in_minutes": 450.0,
            "Activity (steps)": 8000,
            "sleep_timing_bin": "Very late",
            "Day": "Monday",
        },
        {
            "Start": "2024-01-10T22:30:00",
            "Sleep quality": 3.5,
            "time_in_minutes": 400.0,
            "Activity (steps)": 6000,
            "sleep_timing_bin": "Very late",
            "Day": "Tuesday",
        },
    ]

    payload = {"history": history}

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")


def run_all_tests():
    """Run all test cases"""
    try:
        test_root()
        test_health()
        test_predict_insufficient_data()
        test_predict()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n Error: Could not connect to the API.")
        print(f"Make sure the API is running on {BASE_URL}")
    except Exception as e:
        print(f"\n Error running tests: {str(e)}")


if __name__ == "__main__":
    run_all_tests()
    # test_predict()
