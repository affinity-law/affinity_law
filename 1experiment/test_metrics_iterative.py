import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import project_config as cfg
from candidate_utils import load_candidate_model_class
from dataset import create_test_loader


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


@torch.no_grad()
def predict_all_windows(model, dataloader, device):
    """Collect predictions and ground truth for all sliding windows, shape [num_windows, pre_len, num_features]."""
    model.eval()
    preds, targets = [], []
    for x, y in dataloader:
        preds.append(model(x.to(device)).cpu())
        targets.append(y)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


def inverse_standardize(values, mean, std):
    return values * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def average_future_steps(values):
    """Average overlapping predictions from multiple windows for the same future time step."""
    num_windows, pre_len, num_features = values.shape
    sums = np.zeros((num_windows + pre_len - 1, num_features), dtype=np.float64)
    counts = np.zeros((num_windows + pre_len - 1, 1), dtype=np.float64)

    for window_idx in range(num_windows):
        for horizon_idx in range(pre_len):
            sums[window_idx + horizon_idx] += values[window_idx, horizon_idx]
            counts[window_idx + horizon_idx] += 1

    return sums / counts


def choose_evaluation_series(raw_preds_std, raw_targets_std, mean, std):
    final_preds_std = average_future_steps(raw_preds_std)
    final_targets_std = average_future_steps(raw_targets_std)

    if cfg.EVALUATE_ON_NORMALIZED_DATA:
        return final_preds_std, final_targets_std, "normalized"

    final_preds = inverse_standardize(final_preds_std[None, :, :], mean, std)[0]
    final_targets = inverse_standardize(final_targets_std[None, :, :], mean, std)[0]
    return final_preds, final_targets, "original"


def choose_raw_window_series(raw_preds_std, raw_targets_std, mean, std):
    """Choose the data space used for raw windows metrics.

    Raw windows metrics do not apply overlapping window averaging; they are computed
    directly on predictions and targets of shape [num_windows, pre_len, num_features].
    To align with the conventions used by overall_metrics, this is also controlled by
    cfg.EVALUATE_ON_NORMALIZED_DATA:
    - True: compute in normalized space;
    - False: inverse-standardize back to original space before computing.
    """
    if cfg.EVALUATE_ON_NORMALIZED_DATA:
        return raw_preds_std, raw_targets_std, "normalized"

    raw_preds = inverse_standardize(raw_preds_std, mean, std)
    raw_targets = inverse_standardize(raw_targets_std, mean, std)
    return raw_preds, raw_targets, "original"


def calc_basic_metrics(preds, targets):
    error = preds - targets
    return {
        "rmse": np.sqrt(np.mean(error ** 2)),
        "mae": np.mean(np.abs(error)),
        "wmape": np.sum(np.abs(error)) / np.maximum(np.sum(np.abs(targets)), 1e-6),
        "smape": np.mean(2 * np.abs(error) / np.maximum(np.abs(preds) + np.abs(targets), 1e-6)),
    }


def select_mae_rmse_wmape(metrics):
    """Extract the three most commonly used metrics for comparison against baseline models.

    wmape is kept in the internal consistent convention: 0.10 means 10%.
    wmape_percent is provided as an extra display field for direct comparison
    with percentage results in papers.
    """
    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "wmape": metrics["wmape"],
        "wmape_percent": metrics["wmape"] * 100,
    }


def calc_feature_wise_error(preds, targets, top_k=10):
    result = []
    for feature_idx in range(preds.shape[1]):
        item = calc_basic_metrics(preds[:, feature_idx], targets[:, feature_idx])
        item["feature"] = feature_idx
        result.append(item)
    return sorted(result, key=lambda item: item["mae"], reverse=True)[:top_k]


def calc_horizon_wise_error(raw_preds_std, raw_targets_std):
    """Compute metrics per prediction horizon, using standardized (non-averaged) window predictions.

    This metric is specifically provided for the next round of LLM-based error diagnosis:
    - horizon=1 corresponds to the 1st future step;
    - horizon=pre_len corresponds to the farthest prediction step.

    Note: this does not alter the final test metrics or plotting logic; the continuous
    prediction curve still uses final_preds/final_targets from average_future_steps().
    """
    result = []
    for horizon_idx in range(raw_preds_std.shape[1]):
        item = calc_basic_metrics(raw_preds_std[:, horizon_idx, :], raw_targets_std[:, horizon_idx, :])
        item["horizon"] = horizon_idx + 1
        item["data_space"] = "normalized_raw_windows"
        result.append(item)
    return result


def calc_regime_error(preds, targets):
    flat_preds = preds.reshape(-1)
    flat_targets = targets.reshape(-1)
    q1, q2 = np.quantile(flat_targets, [1 / 3, 2 / 3])
    masks = {
        "low": flat_targets <= q1,
        "middle": (flat_targets > q1) & (flat_targets <= q2),
        "high": flat_targets > q2,
    }
    result = {name: {**calc_basic_metrics(flat_preds[mask], flat_targets[mask]), "sample_count": int(mask.sum())}
              for name, mask in masks.items()}
    result["thresholds"] = {"q33": q1, "q66": q2}
    return result


def safe_corrcoef(x, y):
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def calc_residual_diagnostics(preds, targets):
    residual = (preds - targets).reshape(-1)
    target = targets.reshape(-1)
    residual_centered = residual - residual.mean()
    target_centered = target - target.mean()

    return {
        "bias_mean_error": np.mean(residual),
        "residual_std": np.std(residual),
        "lag1_autocorr": safe_corrcoef(residual_centered[:-1], residual_centered[1:]),
        "residual_target_corr": safe_corrcoef(residual_centered, target_centered),
    }


def save_prediction_plots(preds, targets, output_dir, data_space):
    if not cfg.SAVE_PREDICTION_PLOTS:
        print("SAVE_PREDICTION_PLOTS=False, skipping prediction plot saving.")
        return

    plot_len = len(preds) if cfg.TEST_PLOT_POINTS is None else min(int(cfg.TEST_PLOT_POINTS), len(preds))
    num_features = preds.shape[1] if cfg.TEST_PLOT_MAX_FEATURES is None else min(int(cfg.TEST_PLOT_MAX_FEATURES), preds.shape[1])

    for feature_idx in range(num_features):
        plt.figure(figsize=(12, 5))
        plt.plot(targets[:plot_len, feature_idx], label=f"Ground Truth feature {feature_idx}")
        plt.plot(preds[:plot_len, feature_idx], label=f"Prediction feature {feature_idx}")
        plt.title(f"Test Set: Averaged Prediction vs Ground Truth | Feature {feature_idx} | {data_space}")
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Value" if data_space == "normalized" else "Original Value")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"{cfg.PREDICTION_PLOT_PREFIX}_{feature_idx}.png")
        plt.savefig(fig_path, dpi=cfg.TEST_PLOT_DPI)
        plt.close()
        print("Saved plot:", fig_path)


def main():
    output_dir = getattr(cfg, "TEST_OUTPUT_DIR", "") or os.path.join(cfg.RESULT_DIR, "manual_run", "test")
    model_path = getattr(cfg, "MODEL_WEIGHTS_PATH", "") or os.path.join(cfg.RESULT_DIR, "manual_run", "train", cfg.BEST_MODEL_FILE)
    stats_path = getattr(cfg, "TRAIN_STATS_NPZ_PATH", "") or os.path.join(cfg.RESULT_DIR, "manual_run", "train", cfg.TRAIN_STATS_FILE)
    candidate_model_path = getattr(cfg, "CANDIDATE_MODEL_PATH", "") or os.path.join(cfg.RESULT_DIR, "manual_run", cfg.GENERATED_MODEL_FILE)
    seed = getattr(cfg, "SEARCH_SEED", None)
    os.makedirs(output_dir, exist_ok=True)

    # Load standardization parameters and task shape saved during training to ensure consistent model architecture.
    stats = np.load(stats_path)
    time_lag = int(stats["time_lag"])
    pre_len = int(stats["pre_len"])
    station_num = int(stats["station_num"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print("Random seed:", seed)
    print("Candidate model file:", candidate_model_path)
    print("Model weights file:", model_path)
    print("Test output directory:", output_dir)
    print("Evaluating on normalized data:", cfg.EVALUATE_ON_NORMALIZED_DATA)

    # Create test set DataLoader. The train-set mean/std are reused for standardization.
    test_loader, _, mean, std = create_test_loader(
        npy_path=cfg.NPY_PATH,
        time_lag=time_lag,
        pre_len=pre_len,
        batch_size=cfg.BATCH_SIZE,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
        num_workers=0,
        univariate=cfg.UNIVARIATE,
        seed=seed,
    )

    CandidateModel = load_candidate_model_class(
        candidate_model_path,
        auto_repair=bool(getattr(cfg, "AUTO_REPAIR_CANDIDATE_SYNTAX", True)),
    )
    model = CandidateModel(time_lag=time_lag, pre_len=pre_len, station_num=station_num).to(device)

    state_dict = torch.load(model_path, map_location=device)
    load_info = model.load_state_dict(state_dict, strict=False)

    if load_info.missing_keys:
        print("Missing keys when loading model:", load_info.missing_keys)
    if load_info.unexpected_keys:
        print("Ignored extra keys when loading model:", load_info.unexpected_keys)

    print("Model loaded:", model_path)

    # First collect multi-step predictions from all sliding windows, then average over overlapped time steps.
    # raw_*_for_metrics: no overlapping window averaging, used for raw windows metrics alignment with common baseline models.
    # final_*: with overlapping window averaging, used for existing overall_metrics, saving continuous prediction curves, and plotting.
    raw_preds_std, raw_targets_std = predict_all_windows(model, test_loader, device)
    raw_preds_for_metrics, raw_targets_for_metrics, raw_data_space = choose_raw_window_series(
        raw_preds_std, raw_targets_std, mean, std
    )
    raw_window_metrics = calc_basic_metrics(raw_preds_for_metrics, raw_targets_for_metrics)

    final_preds, final_targets, data_space = choose_evaluation_series(raw_preds_std, raw_targets_std, mean, std)
    final_metrics = calc_basic_metrics(final_preds, final_targets)

    pred_path = os.path.join(output_dir, cfg.FINAL_PREDS_FILE)
    target_path = os.path.join(output_dir, cfg.FINAL_TARGETS_FILE)
    np.save(pred_path, final_preds)
    np.save(target_path, final_targets)

    metrics = {
        "task": {
            "time_lag": time_lag,
            "pre_len": pre_len,
            "station_num": station_num,
            "univariate": cfg.UNIVARIATE,
            "evaluate_on_normalized_data": cfg.EVALUATE_ON_NORMALIZED_DATA,
            "evaluation_data_space": data_space,
            "metric_note": "overall_metrics, feature_wise_worst_error, regime_error, residual_diagnostics, npy saving, and plotting are all based on final_preds/final_targets after overlapping window averaging; raw_window_metrics are computed directly on non-averaged raw windows for easy alignment with common baseline models; horizon_wise_metrics are computed directly on non-averaged raw_preds_std/raw_targets_std for use in the next round of LLM-based diagnosis of errors at different prediction horizons.",
        },
        "evaluation_space": {
            "data_space": data_space,
            "raw_window_data_space": raw_data_space,
            "test_mae_final": final_metrics["mae"],
            "test_mae_raw_windows": raw_window_metrics["mae"],
        },
        "raw_window_metrics": {
            "data_space": raw_data_space,
            "metric_note": "Computed directly on raw_preds/raw_targets without overlapping averaging; wmape is a ratio, wmape_percent is a percentage.",
            "mae_rmse_wmape": select_mae_rmse_wmape(raw_window_metrics),
            "all_metrics": raw_window_metrics,
        },
        "overlapping_average_metrics": {
            "data_space": data_space,
            "metric_note": "Apply average_future_steps() to overlapping window predictions first, then compute metrics; this is also the convention used by existing overall_metrics and plotting. wmape is a ratio, wmape_percent is a percentage.",
            "mae_rmse_wmape": select_mae_rmse_wmape(final_metrics),
            "all_metrics": final_metrics,
        },
        "overall_metrics": final_metrics,
        "feature_wise_worst_error": calc_feature_wise_error(final_preds, final_targets),
        "horizon_wise_metrics": calc_horizon_wise_error(raw_preds_std, raw_targets_std),
        "regime_error": calc_regime_error(final_preds, final_targets),
        "residual_diagnostics": calc_residual_diagnostics(final_preds, final_targets),
        "saved_files": {"final_predictions": os.path.basename(pred_path),
                        "final_targets": os.path.basename(target_path)},
    }

    metrics_path = os.path.join(output_dir, cfg.TEST_METRICS_FILE)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=to_jsonable)

    print(f"Raw Windows Test RMSE: {raw_window_metrics['rmse']:.6f}")
    print(f"Raw Windows Test MAE : {raw_window_metrics['mae']:.6f}")
    print(f"Raw Windows Test WMAPE: {raw_window_metrics['wmape']:.6f} ({raw_window_metrics['wmape'] * 100:.3f}%)")
    print(f"Test RMSE (final averaged series): {final_metrics['rmse']:.6f}")
    print(f"Test MAE  (final averaged series): {final_metrics['mae']:.6f}")
    print(f"Test WMAPE (final averaged series): {final_metrics['wmape']:.6f} ({final_metrics['wmape'] * 100:.3f}%)")
    print("Evaluation data space:", data_space)
    print("Saved averaged predictions:", pred_path)
    print("Saved averaged ground truth:", target_path)
    print("Saved test metrics:", metrics_path)

    save_prediction_plots(final_preds, final_targets, output_dir, data_space)


if __name__ == "__main__":
    main()
