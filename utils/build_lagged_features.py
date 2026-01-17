import pandas as pd

def build_lagged_features(
    history_df: pd.DataFrame,

    lags=(1, 2, 3, 7),
    target="Sleep quality",
    base_vars=("time_in_minutes", "Activity (steps)", "sleep_timing_bin", "Day"),
    time_col="Start",
):
    """
    Builds lag features using df.shift() exactly like my training code,
    then returns the latest single-row feature vector for prediction.

    history_df: data available up to (and including) the most recent COMPLETED record.
                Must contain target + base_vars + time_col.
    feature_cols: exact training feature columns (X.columns).
    """
    df_feat = history_df.sort_values(time_col).reset_index(drop=True).copy()

    for lag in lags:
        df_feat[f"{target}_lag{lag}"] = df_feat[target].shift(lag)

    # Lag the PREDICTORS (use PREVIOUS days' activity and timing)
    # These are known before you try to predict tonight's sleep
    base_vars = ["time_in_minutes", "Activity (steps)", "sleep_timing_bin", "Day"]
    for col in base_vars:
        for lag in lags:
            df_feat[f"{col}_lag{lag}"] = df_feat[col].shift(lag)


    # Drop rows with NaN in lagged features
    # Only keep rows where ALL lagged features are available
    lag_feature_cols = [f"{target}_lag{l}" for l in lags]
    for col in base_vars:
        lag_feature_cols += [f"{col}_lag{l}" for l in lags]

    df_feat = df_feat.dropna(subset=[target] + lag_feature_cols).reset_index(drop=True)

    # Define features 
    # Numeric lagged features
    lag_num = [f"{target}_lag{l}" for l in lags]
    for col in ["time_in_minutes", "Activity (steps)"]:
        lag_num += [f"{col}_lag{l}" for l in lags]

    # Categorical lagged features
    lag_cat = []
    for col in ["sleep_timing_bin", "Day"]:
        lag_cat += [f"{col}_lag{l}" for l in lags]

    # ALL features are lagged (no current-day leakage)
    feature_cols = lag_num + lag_cat

    X = df_feat[feature_cols]
    y = df_feat[target]


    print(f"Shape after feature engineering: X={X.shape}, y={y.shape}")
    # print(feature_cols)

    return X, y, lag_num, lag_cat
