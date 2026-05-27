"""
=============================================================================
main.py
GANGA RIVER WATER QUALITY FORECASTING — FULL PIPELINE
  Step 1 : Load data  (data.gov.in API  OR  synthetic fallback)
  Step 2 : Preprocess (scale, sequences, train/test split)
  Step 3 : Train      (LSTM | GRU | BiLSTM | TCN | Transformer)
  Step 4 : Evaluate   (RMSE | MSE | MAE | MAPE | R²)
  Step 5 : Visualise  (predictions, metrics, loss curves, scatter)

Run:
    python main.py              ← uses API key from config.yaml
    python main.py --synthetic  ← uses synthetic data (no API needed)
=============================================================================
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── Allow imports from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader   import load_data
from src.preprocessing import preprocess
from src.train         import train_all
from src.visualise     import generate_all_plots


def main(use_api=True, state_filter="Uttar Pradesh"):
    print("=" * 65)
    print("  GANGA RIVER WATER QUALITY FORECASTING")
    print("  AI-enabled DSS | Satellite | IoT | Dynamic Models")
    print("=" * 65)

    # ── STEP 1: Load Data ────────────────────────────────────────────────────
    print("\n[STEP 1/5] Loading data...")
    df = load_data(use_api=use_api, state_filter=state_filter)
    print(df.describe().round(2))

    # ── STEP 2: Preprocess ───────────────────────────────────────────────────
    print("\n[STEP 2/5] Preprocessing...")
    X_train, X_test, y_train, y_test, scaler_y = preprocess(df)

    # ── STEP 3 & 4: Train + Evaluate ─────────────────────────────────────────
    print("\n[STEP 3-4/5] Training all models & evaluating...")
    results, histories, model_names = train_all(
        X_train, X_test, y_train, y_test, scaler_y
    )

    # Attach histories to results for visualise
    for i, name in enumerate(model_names):
        results[name]["history"] = histories[i]

    # ── STEP 5: Visualise ─────────────────────────────────────────────────────
    print("\n[STEP 5/5] Generating plots...")
    generate_all_plots(results, model_names)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PROJECT COMPLETE!")
    print("  Check the results/ folder for:")
    print("    predictions_all_models.png")
    print("    metrics_comparison.png")
    print("    loss_curves.png")
    print("    scatter_actual_vs_predicted.png")
    print("    all_metrics.csv")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of API")
    parser.add_argument("--state", type=str, default="Uttar Pradesh",
                        help="State filter for API data")
    args = parser.parse_args()

    main(use_api=not args.synthetic, state_filter=args.state)