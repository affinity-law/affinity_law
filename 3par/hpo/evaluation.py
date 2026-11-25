"""Frozen-selection test evaluation using original-scale overlap-averaged metrics."""

import math
import time
from pathlib import Path

import numpy as np
import torch

from .artifacts import read_json, save_json
from .data import input_time_lag, input_window_details, test_loader
from .models import instantiate
from .training import get_device, set_seed


def average_future_steps(values):
    windows, horizon, channels = values.shape
    sums = np.zeros((windows + horizon - 1, channels), dtype=np.float64)
    counts = np.zeros((windows + horizon - 1, 1), dtype=np.float64)
    for window in range(windows):
        sums[window:window + horizon] += values[window]
        counts[window:window + horizon] += 1
    return sums / counts


def metrics(predictions, targets):
    error = predictions - targets
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
            "wmape": float(np.sum(np.abs(error)) / max(np.sum(np.abs(targets)), 1e-6))}


def test_request(task, config, settings, fit_result, seed):
    request = {"task_id": task["id"], "fit_attempt": fit_result["attempt"],
               "config": config, "seed": seed, "smoke": settings.get("smoke", False)}
    if "time_lag" in config:
        window = input_window_details(config, settings)
        if (any(fit_result.get(key, value) != value for key, value in window.items())
                or fit_result.get("config", {}).get("time_lag", window["time_lag"]) != window["time_lag"]):
            raise ValueError("Test input window does not match the trained configuration")
        request["input_window"] = window
    return request


def read_test_cache(task, config, settings, fit_result, folder, seed):
    """Validate a test cache without loading test data or running inference."""
    path = Path(folder) / "metrics.json"
    if not path.exists():
        return None
    cached = read_json(path)
    if (cached.get("request") != test_request(task, config, settings, fit_result, seed)
            or cached.get("status") != "COMPLETE" or cached.get("task_id") != task["id"]
            or cached.get("seed") != seed or cached.get("evaluation_space") != "original"
            or cached.get("aggregation") != "overlap_average"):
        raise ValueError(f"Test cache does not match the frozen task: {task['id']}/seed{seed}")
    if any(cached.get(key) != value for key, value in input_window_details(config, settings).items()):
        raise ValueError(f"Test cache input-window metadata differs from the frozen task: {task['id']}/seed{seed}")
    for metric in ("rmse", "mae", "wmape"):
        value = cached.get("overall_metrics", {}).get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Test cache contains an invalid metric: {task['id']}/{metric}")
    return cached


@torch.no_grad()
def evaluate_test(task, config, settings, root, fit_folder, fit_result, folder, seed):
    fit_folder, folder = Path(fit_folder), Path(folder)
    signature = test_request(task, config, settings, fit_result, seed)
    started = time.perf_counter()
    for key in ("checkpoint", "candidate_model", "stats"):
        if not (fit_folder / fit_result[key]).is_file():
            raise ValueError(f"Final evaluation input is missing: {key}")
    if not (Path(root) / task["data_path"]).is_file():
        raise ValueError("Test data file is missing")
    cached = read_test_cache(task, config, settings, fit_result, folder, seed)
    if cached is not None:
        return cached
    set_seed(seed)
    device = get_device(settings)
    with np.load(fit_folder / fit_result["stats"], allow_pickle=False) as stats:
        mean, std = stats["mean"], stats["std"]
        if int(stats["time_lag"]) != input_time_lag(config, settings):
            raise ValueError("Saved standardization stats use a different input time_lag")
        if "time_lag" in config:
            for key, value in input_window_details(config, settings).items():
                if key not in stats or stats[key].item() != value:
                    raise ValueError(f"Saved standardization stats use a different input-window {key}")
    batches = test_loader(Path(root) / task["data_path"], task, settings, config["batch_size"], seed, mean, std, config=config)
    code = (fit_folder / fit_result["candidate_model"]).read_text(encoding="utf-8")
    model = instantiate(code, task, config=config).to(device)
    model.load_state_dict(torch.load(fit_folder / fit_result["checkpoint"], map_location=device, weights_only=True), strict=True)
    model.eval()
    predictions, targets = [], []
    for x, y in batches:
        pred = model(x.to(device)).cpu()
        if pred.shape != y.shape or not torch.isfinite(pred).all():
            raise FloatingPointError("Final predictions have an invalid shape or contain NaN/Inf")
        predictions.append(pred.numpy())
        targets.append(y.numpy())
    raw_predictions, raw_targets = np.concatenate(predictions), np.concatenate(targets)
    final_predictions = average_future_steps(raw_predictions) * std + mean
    final_targets = average_future_steps(raw_targets) * std + mean
    result = {"request": signature, "status": "COMPLETE", "seed": seed, "task_id": task["id"],
              "overall_metrics": metrics(final_predictions, final_targets),
              "raw_window_metrics": metrics(raw_predictions * std + mean, raw_targets * std + mean),
              "evaluation_space": "original", "aggregation": "overlap_average",
              "wmape_note": "ratio; 0.1 means 10%", "params": fit_result["params"],
              "param_ratio": fit_result["param_ratio"], "elapsed_seconds": time.perf_counter() - started,
              **input_window_details(config, settings),
              "smoke": settings.get("smoke", False)}
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "averaged_predictions.npy", final_predictions)
    np.save(folder / "averaged_targets.npy", final_targets)
    save_json(folder / "metrics.json", result)
    return result
