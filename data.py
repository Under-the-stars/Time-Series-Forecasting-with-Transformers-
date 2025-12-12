import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# LOAD AMZN CSV (your original loader)
# ============================================================

def load_ticker_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("Date").set_index("Date")
    return df

# Alias — use generic loader for AMZN
def load_amzn_csv(path):
    return load_ticker_csv(path)


# ============================================================
# PRICE TARGETS (future close prices)
# ============================================================

def create_price_targets(df, horizon=5):
    """
    Find AMZN close column from merged dataset,
    then create multi-step future price targets.
    """
    # Detect the AMZN Close column (after merge)
    close_cols = [c for c in df.columns if "Close" in c and "AMZN" in c]
    if len(close_cols) == 0:
        raise KeyError("No AMZN Close column found in merged dataframe!")
    close_col = close_cols[0]

    close = df[close_col].values
    y = []

    for i in range(len(close) - horizon):
        y.append(close[i+1:i+1+horizon])

    return np.array(y), close_col


# ============================================================
# FULL DATASET PIPELINE
# ============================================================

def prepare_dataset(amzn_path, seq_len=60, horizon=5):
    print("🔹 Loading AMZN...")
    amzn_df = load_amzn_csv(amzn_path)

    print("🔹 Merging AMZN + SP500 + NASDAQ...")
    full_df = merge_sources(amzn_df)

    print("🔹 Creating targets (future CLOSE prices)...")
    y, close_col = create_price_targets(full_df, horizon)

    # Align X with y
    X = full_df.iloc[:-horizon].values

    print("🔹 Scaling X and y together...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale y using the close column's mean and std from scaler
    close_idx = full_df.columns.tolist().index(close_col)
    y_scaled = (y - scaler.mean_[close_idx]) / scaler.scale_[close_idx]

    print("🔹 Creating windows...")
    Xw, yw = create_windows(X_scaled, y_scaled, seq_len)

    pri
