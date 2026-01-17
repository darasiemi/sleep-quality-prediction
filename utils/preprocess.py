def preprocess(df):

    df = df.drop(
        columns=[
            "start_hour",
            "end_hour",
            "sleep_hour",
            "wake_hour",
            "sleep_sin",
            "sleep_cos",
            "wake_sin",
            "wake_cos",
            "Time in bed",
        ]
    )

    return df
