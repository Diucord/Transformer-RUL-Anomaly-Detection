"""
evaluation.py — FINAL MASTER VERSION
-------------------------------------
✔ Normal region 자동 탐지
✔ Adaptive thresholds = 정상구간 기반(mean + kσ)
✔ RAW error 기반 warning / RUL 계산
✔ Plot smoothing only (검출에는 영향 없음)
"""

import torch, os
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import create_compressed_dataloaders


# ============================================================
# 1. 정상구간 자동 탐지 (variance spike 기반)
# ============================================================

def detect_normal_region(errors, window=50, spike_threshold=3.0):
    """
    정상구간을 자동 추출하기 위한 variance spike detection.

    - rolling variance가 초기 안정구간(평균)에서 여러 σ 이상 튀면
      그 지점부터 degradation 시작이라고 판단.
    """

    # Rolling variance 계산
    roll_var = np.array([
        np.var(errors[max(0, i-window):i+1])
        for i in range(len(errors))
    ])

    # 초기 안정구간 통계
    init_mean = np.mean(roll_var[:500])
    init_std  = np.std(roll_var[:500])

    spike_limit = init_mean + spike_threshold * init_std

    # spike 발생 위치 찾기
    spike_idx = np.argmax(roll_var > spike_limit)

    if roll_var[spike_idx] <= spike_limit:
        spike_idx = len(errors) // 3    # fallback (IMS는 보통 초반은 healthy)

    return spike_idx, roll_var, spike_limit



# ============================================================
# 2. Adaptive Threshold (정상구간 기반)
# ============================================================

def build_adaptive_thresholds_from_normal(normal_errors):
    """
    정상구간 데이터만 기반으로 threshold 생성.
    """
    mean = np.mean(normal_errors)
    std  = np.std(normal_errors)

    thr_low    = mean + 3.0 * std
    thr_medium = mean + 4.5 * std
    thr_high   = mean + 6.0 * std

    return thr_low, thr_medium, thr_high


def compute_warning_level(error, thr_low, thr_med, thr_high):
    if error > thr_high:
        return "High"
    elif error > thr_med:
        return "Medium"
    elif error > thr_low:
        return "Low"
    return "None"



# ============================================================
# 3. Model Evaluation + RUL
# ============================================================

def evaluate_model_with_rul(
    model,
    npy_file,
    test_set,
    batch_size,
    threshold,          # legacy parameter (unused)
    consecutive_steps,
    interval_min,
    device
):
    model.eval()

    loader = create_compressed_dataloaders(
        npy_file=npy_file,
        batch_size=batch_size,
        shuffle=False
    )

    actuals_list, predictions_list, errors_list = [], [], []
    warning_levels_str, warning_levels_numeric = [], []
    time_axis = []
    sample_index = 0

    level_map = {"None":0, "Low":1, "Medium":2, "High":3}

    # ============================================================
    # 1) MODEL FORWARD
    # ============================================================
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].float().to(device)
            outputs = model(inputs)
            target = inputs[:, -1, :]

            batch_errors = torch.mean((outputs - target)**2, dim=1).cpu().numpy()

            actuals_list.append(target.cpu().numpy())
            predictions_list.append(outputs.cpu().numpy())
            errors_list.append(batch_errors)

            # 시간축 생성
            for _ in range(len(batch_errors)):
                time_axis.append(sample_index * interval_min)
                sample_index += 1

    actuals = np.concatenate(actuals_list)
    predictions = np.concatenate(predictions_list)
    errors = np.concatenate(errors_list)
    time_axis = np.array(time_axis)



    # ============================================================
    # 2) 정상구간 자동 탐지 (variance spike)
    # ============================================================
    spike_idx, roll_var, spike_limit = detect_normal_region(errors)

    # 정상구간 error
    normal_errors = errors[:spike_idx]

    # ============================================================
    # 3) Adaptive Threshold (정상구간 기반)
    # ============================================================
    thr_low, thr_med, thr_high = build_adaptive_thresholds_from_normal(normal_errors)

    # --- 정상구간 error 통계 ---
    err_mean = float(np.mean(normal_errors))
    err_std  = float(np.std(normal_errors))



    # ============================================================
    # 4) Warning Level 계산
    # ============================================================
    failure_idx = None
    consec = 0

    for i, e in enumerate(errors):
        lvl_str = compute_warning_level(e, thr_low, thr_med, thr_high)

        warning_levels_str.append(lvl_str)
        warning_levels_numeric.append(level_map[lvl_str])

        # 연속 High → failure index
        if lvl_str == "High":
            consec += 1
            if consec >= consecutive_steps and failure_idx is None:
                failure_idx = i - consecutive_steps + 1
        else:
            consec = 0



    # ============================================================
    # 5) RUL 계산
    # ============================================================
    estimated_rul = None
    if failure_idx is not None:
        estimated_rul = (len(errors) - 1 - failure_idx) * interval_min



    # ============================================================
    # 6) RETURN (main.py와 정확히 맞는 형식)
    # ============================================================
    return (
        actuals,
        predictions,
        errors,  
        warning_levels_numeric,
        warning_levels_str,

        failure_idx,
        estimated_rul,
        time_axis,

        # 정상구간 탐지 관련
        spike_idx,
        roll_var,
        spike_limit,

        # adaptive threshold + 정상구간 error stats
        thr_low,
        thr_med,
        thr_high,
        err_mean,
        err_std
    )


# ============================================================
# 4. 정상구간 검출 그래프 저장
# ============================================================

def save_normal_region_plot(errors, roll_var, spike_idx, spike_limit, time_axis, output_path):
    plt.figure(figsize=(15, 6))

    plt.plot(time_axis, errors, label="Reconstruction Error", alpha=0.7)
    plt.plot(time_axis, roll_var, label="Rolling Variance (window=200)", alpha=0.7)

    plt.axhline(spike_limit, color="red", linestyle="--", label="Spike Limit")

    if spike_idx is not None:
        plt.axvline(time_axis[spike_idx], color="red", linewidth=2, label="Spike Point")

    plt.xlabel("Time (min)")
    plt.ylabel("Error / Variance")
    plt.grid(True, linestyle=":")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[FIGURE] Normal region detection saved → {output_path}")




# ============================================================
# 5. Prediction + Warning Plot (smoothing allowed)
# ============================================================

def plot_results_with_warnings(
    actuals,
    predictions,
    errors,
    warning_levels_str,
    time_axis,
    plot_channel,
    output_dir,
    title,
    smooth_window=5,
    warn_stride=20
):

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        errors_smooth = np.convolve(errors, kernel, mode="same")
    else:
        errors_smooth = errors

    warn_colors = {"Medium": "gold", "High": "red"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10), sharex=True)

    # --- Prediction plot ---
    ax1.plot(time_axis, actuals[:, plot_channel], color="blue", label="Actual")
    ax1.plot(time_axis, predictions[:, plot_channel], color="orange", linestyle="--", label="Predicted")
    ax1.set_ylabel("Normalized Vibration")
    ax1.set_title(title)
    ax1.grid(True, linestyle=":")
    ax1.legend()

    # --- Error plot ---
    ax2.plot(time_axis, errors_smooth, color="purple")
    ax2.set_yscale("log")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("MSE Error (log-scale)")
    ax2.grid(True, linestyle=":")

    # warning markers
    for idx, lvl in enumerate(warning_levels_str):
        if lvl in ("Medium", "High") and idx % warn_stride == 0:
            ax1.axvline(time_axis[idx], color=warn_colors[lvl], alpha=0.7)
            ax2.axvline(time_axis[idx], color=warn_colors[lvl], alpha=0.7)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    name = title.replace(" ", "_").replace("→", "_to_")
    plt.savefig(os.path.join(output_dir, f"prediction_warning_{name}.png"), dpi=200)
    plt.close()


def plot_error_histogram(errors, thr_low, thr_med, thr_high, output_path):
    plt.figure(figsize=(10,6))

    # histogram
    plt.hist(errors, bins=80, color="skyblue", alpha=0.7, density=True, label="Error Distribution")

    # Gaussian fit
    mu = np.mean(errors)
    sigma = np.std(errors)
    x = np.linspace(min(errors), max(errors), 500)
    pdf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-((x-mu)**2)/(2*sigma**2))
    plt.plot(x, pdf, color="red", label=f"Gaussian Fit\nμ={mu:.4e}, σ={sigma:.4e}")

    # thresholds
    plt.axvline(thr_low, color='green', linestyle='--', label='Low Threshold')
    plt.axvline(thr_med, color='gold', linestyle='--', label='Medium Threshold')
    plt.axvline(thr_high, color='red', linestyle='--', label='High Threshold')

    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title("Error Distribution & Gaussian Fit")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[FIGURE] Saved → {output_path}")

def plot_failure_zoom(errors, warning_levels_str, failure_idx, time_axis, output_path, zoom_range=500):
    if failure_idx is None:
        print("[Zoom] No failure_idx → Skipping zoom plot.")
        return

    start = max(0, failure_idx - zoom_range)
    end = min(len(errors), failure_idx + zoom_range)

    plt.figure(figsize=(12,6))
    plt.plot(time_axis[start:end], errors[start:end], color="purple")

    plt.axvline(time_axis[failure_idx], color="red", linestyle="--", label="Failure Onset")
    
    plt.title("Failure Onset Zoom-In")
    plt.xlabel("Time (min)")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"[FIGURE] Saved zoom → {output_path}")
