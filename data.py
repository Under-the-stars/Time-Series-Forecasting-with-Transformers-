"""
data.py
Full preprocessing pipeline for 5-step stock forecasting
compatible with Option C Transformer model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# Load Raw CSV (Ticker)
# ============================================================

def load_ticker_csv(path):
    """
    Loads raw ticker CSV using the known column order:
    Date, Low, Open, Volume, High, Close, Adjusted Close
    """
    df = pd.read_csv(path)

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# ============================================================
# Clean + Select Columns
# ============================================================

def clean_and_select(df):
    """
    Keeps only the 5 features we use as inputs:
    Open, High, Low, Close, Volume

    Given the CSV column order:
    Date, Low, Open, Volume, High, Close, Adjusted Close
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date")
    df = df.set_index("Date")

    # Resample to business days & forward-fill missing days
    df = df.asfreq("B").ffill()

    # Select the 5 input features in correct order
    df = df[[
        "Open",      # index 2
        "High",      # index 4
        "Low",       # index 1
        "Close",     # index 5
        "Volume"     # index 3
    ]]

    return df


# ============================================================
# Scaling (Train-only fit)
# ============================================================

def scale_features(df):
    """
    Fits scaler ONLY on training portion to avoid leakage.
    Returns scaled DataFrame and fitted scaler.
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return scaled_df, scaler


def transform_with_scaler(df, scaler):
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


# ============================================================
# Multi-Step Windowing (60 → next 5)
# ============================================================

def create_windows_multi_step(data, seq_len=60, out_len=5):
    """
    data: numpy array of shape (num_timesteps, num_features)
    outputs:
       X: (num_samples, seq_len, num_features)
       y: (num_samples, out_len)
    """
    X, y = [], []
    close_idx = 3  # Close is 4th column in our selected features

    for i in range(len(data) - seq_len - out_len + 1):
        X.append(data[i : i + seq_len])  
        y.append(data[i + seq_len : i + seq_len + out_len, close_idx])

    return np.array(X), np.array(y)


# ============================================================
# Train/Val/Test Split
# ============================================================

def split_data(X, y, train_ratio=0.7, val_ratio=0.1):
    """
    Time-aware split:
    70% train, 10% validation, 20% test
    """

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# Full Pipeline
# ============================================================

def prepare_dataset(path, seq_len=60, out_len=5):
    """
    Full pipeline:
    - load CSV
    - clean + resample
    - split train/val/test before scaling
    - scale features using ONLY train split
    - create windows for all splits
    """

    print("🔹 Loading raw data...")
    df = load_ticker_csv(path)

    print("🔹 Cleaning & selecting columns...")
    df = clean_and_select(df)

    print("🔹 Splitting before scaling...")
    # Convert to numpy for splitting
    values = df.values
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    print("🔹 Scaling (train-only fit)...")
    df_train_scaled, scaler = scale_features(df_train)
    df_val_scaled = transform_with_scaler(df_val, scaler)
    df_test_scaled = transform_with_scaler(df_test, scaler)

    print("🔹 Creating windows...")
    X_train, y_train = create_windows_multi_step(df_train_scaled.values, seq_len, out_len)
    X_val, y_val = create_windows_multi_step(df_val_scaled.values, seq_len, out_len)
    X_test, y_test = create_windows_multi_step(df_test_scaled.values, seq_len, out_len)

    print("✅ Dataset ready!")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
