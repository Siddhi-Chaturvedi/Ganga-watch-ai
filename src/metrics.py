"""
=============================================================================
metrics.py
Compute RMSE, MSE, MAE, MAPE, R² for model evaluation
=============================================================================
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true, y_pred):
    """
    Calculate all evaluation metrics.

    Parameters
    ----------
    y_true : array-like, actual values
    y_pred : array-like, predicted values

    Returns
    -------
    dict with keys: MSE, RMSE, MAE, MAPE, R2
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)

    # MAPE — avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    r2   = r2_score(y_true, y_pred)

    return {
        "MSE" : round(mse,  6),
        "RMSE": round(rmse, 6),
        "MAE" : round(mae,  6),
        "MAPE": round(mape, 4),
        "R2"  : round(r2,   6),
    }


def print_metrics(name, metrics):
    """Pretty print metrics for one model."""
    print(f"\n{'─'*55}")
    print(f"  Model : {name}")
    print(f"{'─'*55}")
    print(f"  MSE   : {metrics['MSE']:.6f}")
    print(f"  RMSE  : {metrics['RMSE']:.6f}")
    print(f"  MAE   : {metrics['MAE']:.6f}")
    print(f"  MAPE  : {metrics['MAPE']:.4f} %")
    print(f"  R²    : {metrics['R2']:.6f}")
    print(f"{'─'*55}")


def print_metrics_table(results, model_names):
    """Print a comparison table for all models."""
    print("\n" + "=" * 72)
    print(f"{'MODEL':<15} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'MAPE(%)':>10} {'R²':>10}")
    print("=" * 72)
    for name in model_names:
        m = results[name]["metrics"]
        print(f"{name:<15} {m['MSE']:>10.4f} {m['RMSE']:>10.4f} "
              f"{m['MAE']:>10.4f} {m['MAPE']:>10.2f} {m['R2']:>10.4f}")
    print("=" * 72)