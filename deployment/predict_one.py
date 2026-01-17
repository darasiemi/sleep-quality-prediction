from utils.load_data import load_data
from utils.preprocess import preprocess
from utils.split_data import train_test_split
from utils.build_lagged_features import build_lagged_features
from utils.load_model import load_model



feature_cols = ['Sleep quality_lag1', 'Sleep quality_lag2', 'Sleep quality_lag3',
       'Sleep quality_lag7', 'time_in_minutes_lag1', 'time_in_minutes_lag2',
       'time_in_minutes_lag3', 'time_in_minutes_lag7', 'Activity (steps)_lag1',
       'Activity (steps)_lag2', 'Activity (steps)_lag3',
       'Activity (steps)_lag7', 'sleep_timing_bin_lag1',
       'sleep_timing_bin_lag2', 'sleep_timing_bin_lag3',
       'sleep_timing_bin_lag7', 'Day_lag1', 'Day_lag2', 'Day_lag3',
       'Day_lag7']

if __name__ == "__main__":
    data_url = "archive/preprocessed_data.csv"
    parse_cols = ["Start", "End"]
    df = load_data(data_url, parse_cols)

    df = preprocess(df)

    df_train, df_test = train_test_split(df)

    X_prev_8 = df_test.iloc[8:16] 

    X_single, y, lag_num, lag_cat = build_lagged_features(X_prev_8)


    X_single  = X_single[feature_cols]

    model_path = "model/sleep_model_train.bin"

    pipeline  = load_model(model_path)

    y_pred = pipeline.predict(X_single)
    # print(y_pred)
    prediction = round(float(y_pred[0]),2)

    print(prediction)

    # Compare to actual value of row 16 (not row 15!)
    actual = float(df_test["Sleep quality"].iloc[16])
    print(actual)

    
    
    