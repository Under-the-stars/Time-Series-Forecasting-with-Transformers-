"""
Advanced data preprocessing pipeline for AMZN forecasting with:
- OHLCV from NASDAQ dataset
- SP500 index (^GSPC) via yfinance
- NASDAQ Composite (^IXIC) via yfinance
- 16 technical indicators (medium set)
- Stationary inputs (log returns)
- Target = 5-day future log returns
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Load AMZN CSV
# ============================================================

def load_amzn_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")
    return df


# ============================================================
# 2. Download Index Data (SP500 + NASDAQ)
# ============================================================

def load_index(symbol, start):
    df = yf.download(symbol, start=start)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    return df


# ============================================================
# 3. Technical Indicators
# ============================================================

def add_indicators(df):
    df = df.copy()

    # -------- Trend Indicators ----------
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    # -------- Momentum Indicators ----------
    df["ROC_10"] = df["Close"].pct_change(10)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["RSI"] = gain / (gain + loss)

    # -------- Volatility Indicators ----------
    df["Volatility_10"] = df["Close"].rolling(10).std()
    df["Volatility_20"] = df["Close"].rolling(20).std()

    df["ATR"] = (
        pd.concat([
            (df["High"] - df["Low"]),
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
    ).rolling(14).mean()

    # -------- Volume Indicators ----------
    df["Volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    df["Volume_change"] = df["Volume"].pct_change()

    return df


# ============================================================
# 4. Prepare Full Feature Set (stationary inputs)
# ============================================================

def make_stationary(df):
    df = df.copy()

    df["log_return"] = np.log(df["Close"]).diff()

    # Replace raw prices with log returns
    features = [
        "log_return",
        "Volume_z", "Volume_change",
        "SMA_5", "SMA_10", "SMA_20",
        "EMA_12", "EMA_26",
        "MACD", "MACD_signal",
        "ROC_10",
        "RSI",
        "Volatility_10", "Volatility_20",
        "ATR"
    ]

    return df[features]


# ============================================================
# 5. Build Dataset with AMZN + SP500 + NASDAQ
# ============================================================

def merge_sources(amzn_df):
    start_date = amzn_df["Date"].min()

    sp500 = load_index("^GSPC", start_date)
    nasdaq = load_index("^IXIC", start_date)

    sp500["sp500_ret"] = np.log(sp500["Close"]).diff()
    nasdaq["nasdaq_ret"] = np.log(nasdaq["Close"]).diff()

    # Keep only market returns
    sp500 = sp500[["sp500_ret"]]
    nasdaq = nasdaq[["nasdaq_ret"]]

    amzn = amzn_df.set_index("Date")

    # Add indicators & stationary transform
    amzn = add_indicators(amzn)
    amzn = make_stationary(amzn)

    # Merge everything
    merged = amzn.join(sp500, how="left").join(nasdaq, how="left")

    # Fill missing index values
    merged = merged.ffill().dropna()

    return merged


# ============================================================
# 6. Create Multi-Step Targets (future 5-day returns)
# ============================================================

def create_targets(df, horizon=5):
    close = df["log_return"]  # Already stationary

    y = []

    for i in range(len(close) - horizon):
        future_returns = close.iloc[i+1:i+1+horizon].values
        y.append(future_returns)

    y = np.array(y)

    return y


# ============================================================
# 7. Sliding Windows
# ============================================================

def create_windows(X, y, seq_len):
    X_windows = []
    y_windows = []

    for i in range(len(X) - seq_len - 5):
        X_windows.append(X[i:i+seq_len])
        y_windows.append(y[i+seq_len])

    return np.array(X_windows), np.array(y_windows)


# ============================================================
# 8. Train/Val/Test Split
# ============================================================

def split_time_series(X, y):
    n = len(X)
    train = int(n * 0.7)
    val = int(n * 0.1) + train

    return (
        X[:train], y[:train],
        X[train:val], y[train:val],
        X[val:], y[val:]
    )


# ============================================================
# 9. Main Dataset Preparation Function
# ============================================================

def prepare_dataset(amzn_path, seq_len=60, horizon=5):

    print("🔹 Loading AMZN...")
    amzn_df = load_amzn_csv(amzn_path)

    print("🔹 Merging AMZN + SP500 + NASDAQ...")
    full_df = merge_sources(amzn_df)

    print("🔹 Creating targets...")
    y = create_targets(full_df, horizon)

    print("🔹 Extracting features...")
    X = full_df.values[:-horizon]

    print("🔹 Scaling X and y separately...")
    scaler_x = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)

    X = scaler_x.transform(X)
    y = scaler_y.transform(y)

    print("🔹 Creating windows...")
    Xw, yw = create_windows(X, y, seq_len)

    print("🔹 Splitting train/val/test...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_time_series(Xw, yw)

    print("✅ Dataset ready!")
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_x, scaler_y
