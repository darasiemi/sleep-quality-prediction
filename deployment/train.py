from utils.build_lagged_features import build_lagged_features
from utils.build_model import build_model
from utils.load_data import load_data
from utils.preprocess import preprocess
from utils.split_data import train_test_split

if __name__ == "__main__":
    data_url = "archive/preprocessed_data.csv"
    parse_cols = ["Start", "End"]
    df = load_data(data_url, parse_cols)

    df = preprocess(df)

    df_train, df_test = train_test_split(df)
    X, y, lag_num, lag_cat = build_lagged_features(df_train)
    pipe = build_model(X, y, lag_num, lag_cat)
