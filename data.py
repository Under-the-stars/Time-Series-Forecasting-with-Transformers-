def create_price_targets(df, horizon=5):
    """
    Predict next 5 CLOSE prices.
    target[t] = [Close[t+1], Close[t+2], ..., Close[t+horizon]]
    """
    close = df["Close"].values
    y = []

    for i in range(len(close) - horizon):
        y.append(close[i+1:i+1+horizon])

    return np.array(y)


def prepare_dataset(amzn_path, seq_len=60, horizon=5):
    print("🔹 Loading AMZN...")
    amzn_df = load_amzn_csv(amzn_path)

    print("🔹 Merging AMZN + SP500 + NASDAQ...")
    full_df = merge_sources(amzn_df)

    print("🔹 Creating targets (future CLOSE prices)...")
    y = create_price_targets(full_df, horizon)

    # Align X with y
    X = full_df.iloc[:-horizon].values

    print("🔹 Scaling X and y together...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale y using same scaler (but only Close column!)
    close_col = list(full_df.columns).index("Close")
    y_scaled = (y - scaler.mean_[close_col]) / scaler.scale_[close_col]

    print("🔹 Creating windows...")
    Xw, yw = create_windows(X_scaled, y_scaled, seq_len)

    print("🔹 Splitting train/val/test...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_time_series(Xw, yw)

    print("✅ Dataset ready!")
    print("Train:", X_train.shape)
    print("Val:",   X_val.shape)
    print("Test:",  X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
