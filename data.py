"""
data.py — Clean, correct preprocessing for AMZN stock forecasting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Load & Clean AMZN NASDAQ CSV
# ============================================================

def load_amzn_csv(path):
    """
    Loads AMZN CSV (NASDAQ dataset) with columns:
    Date, Low, Open, Volume, High, Close, Adjusted Close
    """
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)

    # Keep only post-IPO (AMZN IPO: May 1997)
    df = df[df["Date"] >= "1997-05-01"]

    df = df.set_index("Date")

    # Select clean feature set (order matters)
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Remove rows with missing values
    df = df.dropna()

    return df


# ============================================================
# 2. Create Sliding Windows
# ============================================================

def create_windows(features, targets, seq_len=60, out_len=5):
    """
    features: numpy array of shape (N, num_features)
    targets: numpy array of shape (N,)
    Returns:
      X: (num_samples, seq_len, num_features)
      y: (num_samples, out_len)
    """

    X, y = [], []
    N = len(features)

    for i in range(N - seq_len - out_len + 1):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len : i + seq_len + out_len])

    return np.array(X), np.array(y)


# ============================================================
# 3. Full Dataset Pipeline
# ============================================================

def prepare_dataset(
    path,
    seq_len=60,
    out_len=5,
    train_ratio=0.7,
    val_ratio=0.1
):
    """
    Steps:
      1. Load AMZN NASDAQ CSV
      2. Extract features + target
      3. Scale features and target separately
      4. Create sliding windows BEFORE split
      5. Split windows into train/val/test
    """

    print("🔹 Loading AMZN NASDAQ data...")
    df = load_amzn_csv(path)

    values = df.values
    close_prices = df["Close"].values.reshape(-1, 1)

    # ---------------------------------------------------------
    # Scale X and y separately (correct way)
    # ---------------------------------------------------------
    print("🔹 Scaling features and target...")

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    scaled_x = scaler_x.fit_transform(values)
    scaled_y = scaler_y.fit_transform(close_prices)

    # ---------------------------------------------------------
    # Create windows BEFORE split
    # ---------------------------------------------------------
    print("🔹 Creating windows...")

    X_all, y_all = create_windows(
        scaled_x,
        scaled_y.flatten(),
        seq_len=seq_len,
        out_len=out_len
    )

    num_samples = len(X_all)
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    # ---------------------------------------------------------
    # Time-aware split
    # ---------------------------------------------------------
    print("🔹 Splitting into train/val/test...")

    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val,   y_val   = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test,  y_test  = X_all[val_end:], y_all[val_end:]

    print(" Dataset ready!")
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")

    return (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        scaler_x, scaler_y
    )
