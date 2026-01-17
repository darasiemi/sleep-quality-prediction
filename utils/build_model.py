from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import pickle


def build_model(X, y, lag_num, lag_cat):

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]), lag_num),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), lag_cat),
        ],
        remainder="drop"
    )

    rf = RandomForestRegressor(
        n_estimators=800,
        min_samples_split=20,
        min_samples_leaf=20,
        max_features="log2",
        max_depth=24,
        random_state=42,
        n_jobs=-1
    )

    # Full pipeline
    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("model", rf)
    ])


    tscv = TimeSeriesSplit(n_splits=5)

    scoring = {
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "R2": "r2"
    }

    cv = cross_validate(
        pipe,
        X, y,
        cv=tscv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    results_df = pd.DataFrame([{
        "model": "RF_fixed_params",
        "MAE_mean": -cv["test_MAE"].mean(),
        "MAE_std": cv["test_MAE"].std(),
        "R2_mean": cv["test_R2"].mean(),
        "R2_std": cv["test_R2"].std(),
    }])

    pipe.fit(X, y)
    print("model fitted")

    with open('model/sleep_model_train.bin', 'wb') as f_out:
        pickle.dump(pipe, f_out)
    
    print("model saved")
    
    print(results_df)

    return pipe
