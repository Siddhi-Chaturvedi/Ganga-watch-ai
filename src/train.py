"""
=============================================================================
train.py
Train all 5 models, evaluate metrics, save weights & plots
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import yaml
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.metrics import calculate_metrics, print_metrics, print_metrics_table
from src.model_lstm        import build_lstm
from src.model_gru         import build_gru
from src.model_bilstm      import build_bilstm
from src.model_tcn         import build_tcn
from src.model_transformer import build_transformer

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

EPOCHS       = config["training"]["epochs"]
BATCH_SIZE   = config["training"]["batch_size"]
PATIENCE     = config["training"]["patience"]
MODELS_DIR   = config["paths"]["saved_models"]
RESULTS_DIR  = config["paths"]["results"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks(model_name):
    save_path = os.path.join(MODELS_DIR, f"{model_name.lower()}.h5")
    return [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=0),
        ModelCheckpoint(save_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
    ]


# ── Train single model ────────────────────────────────────────────────────────
def train_model(model, model_name, X_train, y_train, X_test, y_test, scaler_y):
    print(f"\n  Training {model_name}...")

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(model_name),
        verbose=1
    )

    # Predict & inverse transform
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler_y.inverse_transform(y_test).flatten()

    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(model_name, metrics)

    return history, y_true, y_pred, metrics


# ── Train all models ──────────────────────────────────────────────────────────
def train_all(X_train, X_test, y_train, y_test, scaler_y):
    seq_len    = X_train.shape[1]
    n_features = X_train.shape[2]

    model_builders = {
        "LSTM"       : build_lstm(seq_len, n_features),
        "GRU"        : build_gru(seq_len, n_features),
        "BiLSTM"     : build_bilstm(seq_len, n_features),
        "TCN"        : build_tcn(seq_len, n_features),
        "Transformer": build_transformer(seq_len, n_features),
    }

    results   = {}
    histories = []

    for name, model in model_builders.items():
        hist, y_true, y_pred, metrics = train_model(
            model, name, X_train, y_train, X_test, y_test, scaler_y
        )
        results[name] = {"y_true": y_true, "y_pred": y_pred,
                         "metrics": metrics, "history": hist}
        histories.append(hist)

    print_metrics_table(results, list(model_builders.keys()))

    # Save metrics CSV
    rows = [{"Model": n, **results[n]["metrics"]} for n in results]
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "all_metrics.csv"), index=False
    )
    print(f"\n[train] Metrics saved to {RESULTS_DIR}/all_metrics.csv")

    return results, histories, list(model_builders.keys())


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader   import load_data
    from src.preprocessing import preprocess

    df = load_data(use_api=True)
    X_train, X_test, y_train, y_test, scaler_y = preprocess(df)
    results, histories, model_names = train_all(X_train, X_test, y_train, y_test, scaler_y)