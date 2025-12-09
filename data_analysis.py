"""
data_analysis.py

Statistical analysis utilities for Predictive Maintenance.

Includes:
- RMS computation over time
- RMS trend visualization
- Statistical tests (K-S normality test, Levene variance test)
- Baseline model comparison (Moving Average)

This module contributes to:
  - Dataset distribution analysis
  - Feature stability evaluation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest, levene

from utils import build_time_axis


# ============================================================
# 1) RMS Computation
# ============================================================

def compute_rms(x: np.ndarray):
    """Return RMS of a signal vector."""
    return np.sqrt(np.mean(x ** 2))


def calculate_rms_over_time(raw_data_dir, test_set, downsample_ratio=10):
    """
    Compute RMS for each vibration file in the IMS dataset.

    Parameters
    ----------
    raw_data_dir : str
    test_set : str
    downsample_ratio : int

    Returns
    -------
    rms_values : np.ndarray (N,)
    """

    folder = os.path.join(raw_data_dir, test_set)
    files = sorted(os.listdir(folder))

    rms_values = []

    for file in files:
        try:
            path = os.path.join(folder, file)
            data = pd.read_csv(path, sep="\t", header=None).values

            # Channel reduction (if 8 channels → 4)
            if data.shape[1] == 8:
                data = np.column_stack([
                    (data[:, 0] + data[:, 1]) / 2,
                    (data[:, 2] + data[:, 3]) / 2,
                    (data[:, 4] + data[:, 5]) / 2,
                    (data[:, 6] + data[:, 7]) / 2,
                ])

            # Downsample entire frame
            data = data[::downsample_ratio]

            # Per-file RMS (across all channels)
            rms_val = compute_rms(data)
            rms_values.append(rms_val)

        except Exception as e:
            print(f"[RMS ERROR] Failed processing {file}: {e}")

    return np.array(rms_values)


# ============================================================
# 2) RMS Trend Plot
# ============================================================

def plot_rms_trend(rms_values, test_set, output_dir):
    """
    Plot RMS evolution over time using a correct time axis.

    Time axis is derived from:
        - 5-minute interval (first 43 files) for 1st_test
        - 10-minute intervals otherwise
    """

    time_axis = build_time_axis(test_set, len(rms_values))

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis[:-1], rms_values, label="RMS", color="blue")
    plt.title(f"RMS Trend Over Time — {test_set}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("RMS Value")
    plt.grid(True)
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"rms_trend_{test_set}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] RMS Trend Plot: {save_path}")


# ============================================================
# 3) Statistical Tests
# ============================================================

def run_statistical_tests(rms_values):
    """
    Perform statistical tests on RMS sequence.

    Includes:
    - K-S Test for normality
    - Levene Test for variance stability (first vs second half)
    """

    print("\n=== Statistical Tests ===")

    # K-S Test against Normal Distribution
    ks_stat, ks_p = kstest(rms_values, "norm")
    print(f"K-S Test → stat={ks_stat:.4f}, p={ks_p:.4e}")

    # Levene Test for equal variances
    mid = len(rms_values) // 2
    lev_stat, lev_p = levene(rms_values[:mid], rms_values[mid:])
    print(f"Levene Test → stat={lev_stat:.4f}, p={lev_p:.4e}")

    if ks_p < 0.05:
        print("• RMS values do NOT follow a normal distribution.")
    else:
        print("• RMS values appear normally distributed.")

    if lev_p < 0.05:
        print("• Variance differs significantly (non-stationary).")
    else:
        print("• Variance is stable across segments.")


# ============================================================
# 4) Baseline Model Comparison
# ============================================================

def baseline_model_comparison(predictions, actuals, window=5):
    """
    Compare the Transformer model against a statistical baseline model.

    Baseline:
    - Moving Average (window = 5)
    - Predicts next timestep using past window average

    Metrics:
    - MSE comparison
    """

    print("\n=== Baseline Model Comparison ===")

    actual = actuals[:, 0]      # compare channel 1
    pred = predictions[:, 0]

    # Moving Average baseline
    baseline = pd.Series(actual).rolling(window=window).mean().fillna(method="bfill")

    mse_baseline = np.mean((baseline - actual) ** 2)
    mse_model = np.mean((pred - actual) ** 2)

    print(f"Baseline MA(w={window}) MSE  : {mse_baseline:.6f}")
    print(f"Transformer Model MSE       : {mse_model:.6f}")

    if mse_model < mse_baseline:
        print("→ Transformer outperforms the baseline.")
    else:
        print("→ Baseline is equal or better. Consider tuning.")
