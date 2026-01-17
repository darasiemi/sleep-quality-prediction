def train_test_split(df_feat, train_size=0.8):

    split_point = int(df_feat.shape[0] * train_size)

    df_feat_train = df_feat.iloc[:split_point]
    df_feat_test = df_feat.iloc[split_point:]

    return df_feat_train, df_feat_test
