"""
main.py

Full end-to-end pipeline for Transformer-based Predictive Maintenance
using the IMS Bearing Dataset.

Pipeline steps:
1. Preprocess raw IMS dataset (8→4 channel reduction + downsampling)
2. RMS computation + statistical analysis (K-S Test, Levene Test)
3. Transformer model training (checkpoint resume supported)
4. Evaluation on target test sets
5. Reconstruction error → Warning Level → Failure detection
6. RUL estimation (time-axis based)
7. Visualization & baseline comparison
"""

import os, torch, json
from datetime import datetime
from scipy import stats
import json as _json
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

# Core Data Handling
from data_preprocessing import (
    preprocess_and_save,
    create_compressed_dataloaders
)

# Statistical Analysis
from data_analysis import (
    calculate_rms_over_time,
    plot_rms_trend,
    run_statistical_tests,
    #baseline_model_comparison
)

# Model and Training
from training import train_model
from model_transformer import AdvancedAutoInformerModel

# Evaluation and Visualization
from evaluation import (
    evaluate_model_with_rul,
    plot_results_with_warnings,
    save_normal_region_plot,
    plot_error_histogram,
    plot_failure_zoom
)

# Utilities
def model_path(base_dir, train_set):
    """Canonical path for the 'latest' final model (overwritten when training saves)."""
    return os.path.join(base_dir, f"{train_set}_model.pth")

def timestamped_model_path(base_dir, train_set):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{train_set}_model_{ts}.pth")


# ==========================================================
#                       MAIN PIPELINE
# ==========================================================

if __name__ == '__main__':
    # ==============================================================
    # 0) Configuration Loading and Environment Setup
    # ==============================================================
    with open("config.json", "r") as f:
        config = json.load(f)

    DIRS = config["directories"]
    DATA = config["data_settings"]
    TRAIN_HP = config["training_hyperparameters"]
    MODEL_ARCH = config["model_architecture"]
    EVAL_PARAM = config["evaluation_parameters"]

    RAW_DATA_DIR = DIRS["raw_data_dir"]
    PROCESSED_DIR = DIRS["processed_dir"]
    RESULT_DIR = DIRS["result_dir"]
    SAVED_MODELS_DIR = DIRS["saved_models_dir"]

    # Create directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # Create result subfolders
    STAT_DIR = os.path.join(RESULT_DIR, "statistics")
    os.makedirs(STAT_DIR, exist_ok=True)

    TRAIN_LOG_DIR = os.path.join(RESULT_DIR, "training_logs")
    os.makedirs(TRAIN_LOG_DIR, exist_ok=True)

    EVAL_DIR = os.path.join(RESULT_DIR, "evaluation")
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Device setup
    device_str = TRAIN_HP.get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")


    # ==============================================================
    # 1) Preprocess (skip if npy exists)
    # ==============================================================
    print("\n==================== STEP 1: PREPROCESSING ====================\n")

    for test_set in DATA["test_sets"]:
        save_path = os.path.join(PROCESSED_DIR, f"{test_set}_processed.npy")

        if os.path.exists(save_path):
            print(f"[Skip] Preprocessed file already exists: {save_path}")
        else:
            print(f"[Preprocess] Creating preprocessed data for {test_set}")
            preprocess_and_save(
                data_dir=RAW_DATA_DIR,
                test_set=test_set,
                save_dir=PROCESSED_DIR,
                downsample_ratio=DATA["downsample_ratio"]
            )


    # ==============================================================
    # 2) RMS Analysis + Statistical Validation (skip if saved)
    # ==============================================================
    print("\n==================== STEP 2: RMS ANALYSIS & STATISTICAL TESTS ====================\n")

    for test_set in DATA["test_sets"]:
        stat_json_path = os.path.join(STAT_DIR, f"{test_set}_statistics.json")

        if os.path.exists(stat_json_path):
            print(f"[Skip] Statistical summary exists: {stat_json_path}")
            continue

        print(f"[Stats] Running RMS analysis for: {test_set}")

        rms_values = calculate_rms_over_time(
            raw_data_dir=RAW_DATA_DIR,
            test_set=test_set,
            downsample_ratio=DATA["downsample_ratio"]
        )

        # Save raw RMS arrays
        rms_npy_path = os.path.join(STAT_DIR, f"{test_set}_rms.npy")
        np.save(rms_npy_path, rms_values)
        np.savetxt(os.path.join(STAT_DIR, f"{test_set}_rms.csv"), rms_values, delimiter=",")

        # Plot RMS
        try:
            plot_rms_trend(rms_values, test_set, RESULT_DIR)
        except Exception as e:
            print(f"[Warning] plot_rms_trend failed: {e}")

        # K-S test for normality
        mu = float(np.mean(rms_values))
        sigma = float(np.std(rms_values))
        ks_stat, ks_p = stats.kstest(rms_values, 'norm', args=(mu, sigma))

        # Levene test for variance
        try:
            chunks = np.array_split(rms_values, 4)
            lev_stat, lev_p = stats.levene(*chunks)
        except Exception:
            lev_stat, lev_p = float('nan'), float('nan')

        summary = {
            "test_set": test_set,
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "levene_stat": float(lev_stat),
            "levene_p": float(lev_p),
            "mean": mu,
            "std": sigma,
            "generated_at": datetime.now().isoformat()
        }

        # Save JSON
        with open(stat_json_path, "w", encoding="utf-8") as f:
            _json.dump(summary, f, indent=2)

        # Save TXT
        with open(os.path.join(STAT_DIR, f"{test_set}_statistics.txt"), "w") as f:
            f.write(f"--- Statistical Analysis for {test_set} ---\n")
            f.write(f"K-S Test → stat={ks_stat:.4f}, p={ks_p:.4e}\n")
            f.write(f"Levene Test → stat={lev_stat:.4f}, p={lev_p:.4e}\n")

        # Original helper (optional)
        try:
            run_statistical_tests(rms_values)
        except Exception:
            pass

    # ==============================================================
    # 3) Train Transformer Model per Test Set
    #    (Skip training if latest model exists)
    # ==============================================================
    print("\n==================== STEP 3: MODEL TRAINING ====================\n")

    models = {}

    for train_set in DATA["test_sets"]:
        print(f"\n=== Handling model for {train_set} ===")

        npy_file = os.path.join(PROCESSED_DIR, f"{train_set}_processed.npy")
        if not os.path.exists(npy_file):
            raise FileNotFoundError(f"Processed file not found: {npy_file}")

        # Load data shape
        arr = np.load(npy_file, allow_pickle=True)
        if arr.ndim != 3:
            raise ValueError(f"Expected (N, seq_len, C), got {arr.shape}")
        input_dim = int(arr.shape[2])

        checkpoint_file = os.path.join(SAVED_MODELS_DIR, f"{train_set}_checkpoint.pth")
        latest_model_file = model_path(SAVED_MODELS_DIR, train_set)

        # ----------------------------------------------------------
        # Case 0: If latest model file already exists → SKIP TRAINING
        # ----------------------------------------------------------
        if os.path.exists(latest_model_file) and not os.path.exists(checkpoint_file):
            print(f"[Skip Training] Latest model exists → {latest_model_file}")
            print("→ Loading this pretrained model and moving to evaluation.")

            model = AdvancedAutoInformerModel(
                input_dim=input_dim,
                model_dim=MODEL_ARCH["model_dim"],
                num_heads=MODEL_ARCH["num_heads"],
                num_layers=MODEL_ARCH["num_layers"],
                dropout=MODEL_ARCH["dropout"]
            ).to(device)

            model.load_state_dict(torch.load(latest_model_file, map_location=device))
            model.eval()

            models[train_set] = model
            continue

        # ----------------------------------------------------------
        # Case 1: Checkpoint exists → Resume training
        # ----------------------------------------------------------
        if os.path.exists(checkpoint_file):
            print(f"[Resume] Found checkpoint → {checkpoint_file}")

            model = AdvancedAutoInformerModel(
                input_dim=input_dim,
                model_dim=MODEL_ARCH["model_dim"],
                num_heads=MODEL_ARCH["num_heads"],
                num_layers=MODEL_ARCH["num_layers"],
                dropout=MODEL_ARCH["dropout"]
            ).to(device)

            train_loader = create_compressed_dataloaders(
                npy_file=npy_file,
                batch_size=TRAIN_HP["batch_size"],
                shuffle=True,
                num_workers=0
            )

            print("[Train] Resuming training from checkpoint…")
            model, trained, logs = train_model(
                train_loader=train_loader,
                model=model,
                num_epochs=TRAIN_HP["num_epochs"],
                device=device,
                lr=TRAIN_HP["learning_rate"],
                weight_decay=TRAIN_HP["weight_decay"],
                save_path=checkpoint_file,
                log_dir=os.path.join(TRAIN_LOG_DIR, train_set)
            )

            # Save final updated model
            if trained:
                ts_model = timestamped_model_path(SAVED_MODELS_DIR, train_set)
                torch.save(model.state_dict(), ts_model)
                torch.save(model.state_dict(), latest_model_file)
                print(f"[Final Model Saved] Timestamped: {ts_model}")
                print(f"[Final Model Saved] Latest: {latest_model_file}")
                os.remove(checkpoint_file)

            models[train_set] = model
            continue

        # ----------------------------------------------------------
        # Case 2: No model, no checkpoint → Train from scratch
        # ----------------------------------------------------------
        print("[Training Required] No existing model. Training from scratch!")

        model = AdvancedAutoInformerModel(
            input_dim=input_dim,
            model_dim=MODEL_ARCH["model_dim"],
            num_heads=MODEL_ARCH["num_heads"],
            num_layers=MODEL_ARCH["num_layers"],
            dropout=MODEL_ARCH["dropout"]
        ).to(device)

        train_loader = create_compressed_dataloaders(
            npy_file=npy_file,
            batch_size=TRAIN_HP["batch_size"],
            shuffle=True,
            num_workers=0
        )

        model, trained, logs = train_model(
            train_loader=train_loader,
            model=model,
            num_epochs=TRAIN_HP["num_epochs"],
            device=device,
            lr=TRAIN_HP["learning_rate"],
            weight_decay=TRAIN_HP["weight_decay"],
            save_path=checkpoint_file,
            log_dir=os.path.join(TRAIN_LOG_DIR, train_set)
        )

        if trained:
            ts_model = timestamped_model_path(SAVED_MODELS_DIR, train_set)
            torch.save(model.state_dict(), ts_model)
            torch.save(model.state_dict(), latest_model_file)
            print(f"[Final Model Saved] Timestamped: {ts_model}")
            print(f"[Final Model Saved] Latest: {latest_model_file}")

            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)

        models[train_set] = model


    # ==============================================================
    # 4) Evaluation & RUL Estimation
    # ==============================================================
    print("\n==================== STEP 4: EVALUATION ====================\n")

    table_path = os.path.join(EVAL_DIR, "warning_level_summary.csv")
    if not os.path.exists(table_path):
        with open(table_path, "w") as f:
            f.write("train_set,test_set,failure_idx,RUL_min,thr_low,thr_med,thr_high,mean_error,std_error\n")

    for train_set in DATA["test_sets"]:
        model = models[train_set]

        for test_set in DATA["test_sets"]:
            print(f"\n=== Evaluating {train_set} → {test_set} ===")

            result_output_dir = os.path.join(EVAL_DIR, f"{train_set}_to_{test_set}")
            os.makedirs(result_output_dir, exist_ok=True)

            npy_file = os.path.join(PROCESSED_DIR, f"{test_set}_processed.npy")

            (
                actuals,
                predictions,
                errors,
                warning_levels_numeric,
                warning_levels_str,
                failure_idx,
                estimated_rul,
                time_axis,
                spike_idx,
                roll_var,
                spike_limit,
                thr_low,
                thr_med,
                thr_high,
                err_mean,
                err_std
            ) = evaluate_model_with_rul(
                model=model,
                npy_file=npy_file,
                test_set=test_set,
                batch_size=TRAIN_HP["batch_size"],
                threshold=EVAL_PARAM["threshold"],
                consecutive_steps=EVAL_PARAM["consecutive_steps"],
                interval_min=DATA["rul_interval_min"],
                device=device
            )

            # ----------------- Save Raw Outputs -----------------
            np.savetxt(os.path.join(result_output_dir, "actuals.csv"), actuals, delimiter=",")
            np.savetxt(os.path.join(result_output_dir, "predictions.csv"), predictions, delimiter=",")
            np.savetxt(os.path.join(result_output_dir, "errors.csv"), errors, delimiter=",")
            np.savetxt(os.path.join(result_output_dir, "warning_levels.csv"), warning_levels_numeric, delimiter=",")
            np.savetxt(os.path.join(result_output_dir, "time_axis.csv"), time_axis, delimiter=",")

            # ----------------- Save Summary Table -----------------
            with open(table_path, "a") as f:
                f.write(f"{train_set},{test_set},{failure_idx},{estimated_rul},"
                        f"{thr_low},{thr_med},{thr_high},{err_mean},{err_std}\n")

            # ----------------- Metrics -----------------
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)

            summary = {
                "train_set": train_set,
                "test_set": test_set,
                "mse": mse,
                "mae": mae,
                "failure_idx": failure_idx,
                "estimated_rul": estimated_rul
            }

            with open(os.path.join(result_output_dir, "evaluation_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

            # # Baseline
            # try:
            #     baseline_model_comparison(predictions, actuals)
            # except Exception:
            #     pass

            # print(f"[Saved] Evaluation results → {result_output_dir}")
            # print(f"[Metrics] MSE={mse:.5f}, MAE={mae:.5f}")

            # if failure_idx is None:
            #     print("No critical failure detected.")
            # else:
            #     print(f"Critical Warning Index: {int(failure_idx)}")
            #     print(f"Estimated RUL  : {estimated_rul:.3f} minutes")


            # with open(os.path.join(result_output_dir, "evaluation_summary.json"), "w") as f:
            #     json.dump(summary, f, indent=2)

            # ----------------- Main Figures -----------------
            plot_results_with_warnings(
                actuals,
                predictions,
                errors,
                warning_levels_str,
                time_axis,
                plot_channel=EVAL_PARAM["plot_channel"],
                output_dir=result_output_dir,
                title=f"RUL Prediction ({train_set} → {test_set})"
            )

            save_normal_region_plot(
                errors=errors,
                roll_var=roll_var,
                spike_idx=spike_idx,
                spike_limit=spike_limit,
                time_axis=time_axis,
                output_path=os.path.join(result_output_dir, "normal_region_detection.png")
            )

            plot_error_histogram(
                errors=errors,
                thr_low=thr_low,
                thr_med=thr_med,
                thr_high=thr_high,
                output_path=os.path.join(result_output_dir, "error_histogram.png")
            )

            plot_failure_zoom(
                errors=errors,
                warning_levels_str=warning_levels_str,
                failure_idx=failure_idx,
                time_axis=time_axis,
                output_path=os.path.join(result_output_dir, "failure_zoom.png")
            )


    print("\n==================== ALL TASKS COMPLETED ====================\n")