"""
utils.py

Utility functions for:
- Reconstruction error smoothing
- Adaptive threshold calculation
- Adaptive warning-level classification
- Misc helper functions

Used by evaluation.py and main.py
"""

import numpy as np


# =====================================================================
# 1) Error Smoothing (Moving Average)
# =====================================================================

def smooth_errors(errors: np.ndarray, window_size: int = 20) -> np.ndarray:
    """
    Smooth reconstruction errors using simple moving average.

    Parameters
    ----------
    errors : ndarray
        Raw reconstruction errors.
    window_size : int
        Size of the moving average window.

    Returns
    -------
    ndarray
        Smoothed error array.
    """
    if window_size <= 1:
        return errors

    smoothed = np.convolve(errors, np.ones(window_size) / window_size, mode='same')
    return smoothed
# =====================================================================
# 2) Adaptive Threshold (mean + k * std)
# =====================================================================

def compute_adaptive_threshold(errors: np.ndarray, warmup_ratio: float = 0.1, k: float = 3.0) -> float:
    """
    Compute adaptive threshold based on initial 'healthy' portion of the data.

    Parameters
    ----------
    errors : ndarray
        Reconstruction errors.
    warmup_ratio : float
        Portion of earliest data assumed healthy (default: 10%)
    k : float
        Number of standard deviations above mean.

    Returns
    -------
    float
        Adaptive threshold.
    """

    N = len(errors)
    warmup_len = max(5, int(N * warmup_ratio))  # ensure at least some samples

    healthy_segment = errors[:warmup_len]
    mu = healthy_segment.mean()
    sigma = healthy_segment.std()

    threshold = mu + k * sigma
    return float(threshold)
# =====================================================================
# 4) Consecutive High Warning Detector
# =====================================================================

def detect_failure_index(warning_levels, required_consec=10):
    """
    Detect failure index based on consecutive 'High' warnings.

    Parameters
    ----------
    warning_levels : list[str]
    required_consec : int

    Returns
    -------
    int or None
    """

    consec = 0
    failure_idx = None

    for i, lvl in enumerate(warning_levels):
        if lvl == "High":
            consec += 1
            if consec >= required_consec and failure_idx is None:
                failure_idx = i - required_consec + 1
        else:
            consec = 0

    return failure_idx
# =====================================================================
# 5) RUL Computation (time-based)
# =====================================================================

def compute_rul(failure_index: int, num_samples: int, interval_min: int):
    """
    Computes Remaining Useful Life based on failure index and total samples.

    Returns None if failure not detected.
    """
    if failure_index is None:
        return None

    return (num_samples - 1 - failure_index) * interval_min

def build_time_axis(length, interval_min=10):
    """
    Creates a time axis array for plotting RMS or predictions.
    Example: length=100, interval_min=10 → [0, 10, 20, ...]
    """
    if length <= 0:
        return []

    return [i * interval_min for i in range(length)]

def get_adaptive_warning_level(error, threshold):
    """
    Returns adaptive warning level based on threshold:
    
    - None (normal)
    - Low
    - Medium
    - High
    
    The rule matches evaluation.py logic.
    """
    if error > threshold * 2.0:
        return "High"
    elif error > threshold * 1.5:
        return "Medium"
    elif error > threshold:
        return "Low"
    return "None"
