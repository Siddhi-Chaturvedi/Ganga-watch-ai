"""
=============================================================================
preprocessing.py
Scale data, create time-series sequences, train/test split
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SEQ_LEN     = config["model"]["seq_len"]
TEST_RATIO  = config["model"]["test_ratio"]
TARGET_COL  = config["model"]["target_col"]
SCALER_PATH = config["paths"]["scaler"]


def preprocess(df, target_col=TARGET_COL, seq_len=SEQ_LEN, test_ratio=TEST_RATIO):
    """
    Full preprocessing pipeline:
      1. Scale features with MinMaxScaler
      2. Create sliding window sequences
      3. Train / test split (no shuffle — time series!)

    Parameters
    ----------
    df         : clean time-indexed DataFrame
    target_col : column to predict (default: WQI)
    seq_len    : look-back window in days (default: 30)
    test_ratio : fraction of data for testing (default: 0.2)

    Returns
    -------
    X_train, X_test  : shape (samples, seq_len, n_features)
    y_train, y_test  : shape (samples, 1)
    scaler_y         : fitted scaler for inverse transform
    """
    print(f"[preprocessing] Sequence length = {seq_len} days")
    print(f"[preprocessing] Target column   = {target_col}")
    print(f"[preprocessing] Test ratio      = {test_ratio}")

    features = df.columns.tolist()

    # ── Scale ───────────────────────────────────────────────────────────────
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_scaled = scaler_X.fit_transform(df[features])
    y_scaled = scaler_y.fit_transform(df[[target_col]])

    # ── Save scalers ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler_X, SCALER_PATH.replace(".pkl", "_X.pkl"))
    joblib.dump(scaler_y, SCALER_PATH.replace(".pkl", "_y.pkl"))
    print(f"[preprocessing] Scalers saved.")

    # ── Sliding window sequences ──────────────────────────────────────────────
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i - seq_len:i])
        y_seq.append(y_scaled[i])

    X_seq = np.array(X_seq)   # (N, seq_len, n_features)
    y_seq = np.array(y_seq)   # (N, 1)

    # ── Train / test split ───────────────────────────────────────────────────
    split    = int(len(X_seq) * (1 - test_ratio))
    X_train  = X_seq[:split]
    X_test   = X_seq[split:]
    y_train  = y_seq[:split]
    y_test   = y_seq[split:]

    print(f"[preprocessing] Train shape: X={X_train.shape}  y={y_train.shape}")
    print(f"[preprocessing] Test  shape: X={X_test.shape}   y={y_test.shape}")

    return X_train, X_test, y_train, y_test, scaler_y


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_data

    df = load_data(use_api=False)
    X_train, X_test, y_train, y_test, scaler_y = preprocess(df)
    print("Preprocessing complete.")