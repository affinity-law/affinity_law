"""Train one fixed configuration with normalized validation-loss early stopping."""

import csv
import gc
import math
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .artifacts import atomic_bytes, next_attempt, read_json, save_json, update_json
from .data import input_time_lag, input_window_details, train_val_loaders
from .losses import loss_details
from .models import candidate_code, inspect_candidate, instantiate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(settings):
    name = settings["training"]["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(name)


def loss_value(prediction, target, settings, config=None):
    error = prediction - target
    loss = loss_details(config, settings)
    return (loss["mse_weight"] * error.square().mean()
            + loss["mae_weight"] * error.abs().mean())


def overlap_average(values):
    windows, horizon, channels = values.shape
    sums = np.zeros((windows + horizon - 1, channels), dtype=np.float64)
    counts = np.zeros((windows + horizon - 1, 1), dtype=np.float64)
    for window in range(windows):
        sums[window:window + horizon] += values[window]
        counts[window:window + horizon] += 1
    return sums / counts


@torch.no_grad()
def evaluate(model, batches, settings, device, mean=None, std=None, config=None):
    model.eval()
    total_loss = total_mae = total_se = 0.0
    samples = elements = 0
    predictions, targets = [], []
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        if pred.shape != y.shape or not torch.isfinite(pred).all():
            raise FloatingPointError("Validation output has an invalid shape or contains NaN/Inf")
        n = len(x)
        total_loss += loss_value(pred, y, settings, config).item() * n
        total_mae += (pred - y).abs().mean().item() * n
        total_se += (pred - y).square().sum().item()
        samples += n
        elements += pred.numel()
        if mean is not None and std is not None:
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    normalized_rmse = math.sqrt(total_se / elements)
    selection_rmse = normalized_rmse
    if predictions:
        averaged_predictions = overlap_average(np.concatenate(predictions)) * std + mean
        averaged_targets = overlap_average(np.concatenate(targets)) * std + mean
        selection_rmse = float(np.sqrt(np.mean((averaged_predictions - averaged_targets) ** 2)))
    return {"val_loss": total_loss / samples, "val_mae": total_mae / samples,
            "val_normalized_rmse": normalized_rmse, "val_rmse": selection_rmse}


def curve_summary(history):
    return {"first": history[0], "last": history[-1],
            "minimum_train_loss": min(row["train_loss"] for row in history),
            "minimum_val_loss": min(row["val_loss"] for row in history)}


def training_request(task, config, settings, seed, epochs, patience, purpose):
    """Define the shared cache identity used by training and read-only summaries."""
    request = {"task_id": task["id"], "config": config, "seed": seed,
               "epochs": epochs, "patience": patience, "purpose": purpose,
               "training_settings": settings["training"], "smoke": settings.get("smoke", False),
               "smoke_prefix_points": settings.get("smoke_prefix_points"),
               "train_ratio": settings["train_ratio"], "val_ratio": settings["val_ratio"]}
    # Keep historical request identity intact when reading a run without this field.
    if "loss_profile" in config:
        request["loss"] = loss_details(config, settings)
    if "time_lag" in config:
        request["input_window"] = input_window_details(config, settings)
    return request


def fit(task, config, settings, root, folder, seed, epochs, patience, purpose, record_path=None):
    """Return a result record, reusing completed matches and preserving interrupted attempts."""
    root, folder = Path(root), Path(folder)
    record_path = Path(record_path) if record_path is not None else folder / "training.json"
    request = training_request(task, config, settings, seed, epochs, patience, purpose)
    saved = read_json(record_path) if record_path.exists() else {}
    if "training" in saved:
        raise ValueError("A training record already exists; use a new run name")
    training = {"request": request, "attempts": []}
    if not (root / task["data_path"]).is_file() or not (root / task["model_path"]).is_file():
        raise FileNotFoundError("Task input data or model source is missing")
    attempt = next_attempt(folder)
    started = time.perf_counter()
    result = {"status": "FAIL", "seed": seed,
              "val_rmse": None, "attempt": attempt.name, "config": config,
              "purpose": purpose, "smoke": settings.get("smoke", False),
              **input_window_details(config, settings),
              **loss_details(config, settings)}
    training["attempts"].append({"directory": attempt.name, "started_unix": time.time()})
    update_json(record_path, training=training)
    model = optimizer = train_batches = val_batches = None
    try:
        torch.set_num_threads(settings["training"]["cpu_threads"])
        model_file = attempt / "candidate_model.py"
        atomic_bytes(model_file, candidate_code(task, config, root).encode("utf-8"))
        # Match historical L RNG use: seed, loaders, three preflight batches, then the final model.
        set_seed(seed)
        device = get_device(settings)
        train_batches, val_batches, mean, std = train_val_loaders(
            root / task["data_path"], task, settings, config["batch_size"], seed, config=config)
        inspection = inspect_candidate(task, config, settings, root, check_forward=True,
                                       batch_sizes=(1, 2, config["batch_size"]), preserve_rng=False)
        result.update(params=inspection["params"], param_ratio=inspection["param_ratio"])
        model = instantiate(inspection["code"], task, config=config).to(device)
        atomic_bytes(attempt / "model_structure.txt", str(model).encode("utf-8"))
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        checkpoint = attempt / "best_model_weights.pth"
        stats_file = attempt / "train_standardization_stats.npz"
        window_stats = {"time_lag": input_time_lag(config, settings), **input_window_details(config, settings)}
        np.savez(stats_file, mean=mean, std=std, pre_len=task["horizon"], station_num=task["channels"], **window_stats)
        history, best, no_improve, steps = [], None, 0, 0
        loop_start = time.perf_counter()
        with (attempt / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_mae",
                                                          "val_normalized_rmse", "val_rmse", "best_val_loss"])
            writer.writeheader()
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = total_samples = 0
                for x, y in train_batches:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    loss = loss_value(model(x), y, settings, config)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Training loss contains NaN/Inf")
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(x)
                    total_samples += len(x)
                    steps += 1
                scores = evaluate(model, val_batches, settings, device, mean, std, config=config)
                row = {"epoch": epoch, "train_loss": total_loss / total_samples, **scores}
                if not all(math.isfinite(float(value)) for value in row.values()):
                    raise FloatingPointError("Training or validation metrics are not finite")
                if best is None or scores["val_loss"] < best["val_loss"]:
                    best, no_improve = dict(row), 0
                    temporary = checkpoint.with_suffix(".tmp")
                    torch.save(model.state_dict(), temporary)
                    temporary.replace(checkpoint)
                else:
                    no_improve += 1
                row["best_val_loss"] = best["val_loss"]
                history.append(row)
                writer.writerow(row)
                handle.flush()
                if patience > 0 and no_improve >= patience:
                    break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - loop_start
        # Reload the saved weights to verify that the recorded metric matches the checkpoint.
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        saved_scores = evaluate(model, val_batches, settings, device, mean, std, config=config)
        if not math.isclose(saved_scores["val_loss"], best["val_loss"], rel_tol=1e-6, abs_tol=1e-8):
            raise RuntimeError("Checkpoint validation loss differs from the recorded value")
        result.update(status="COMPLETE", **saved_scores, best_epoch=best["epoch"],
                      checkpoint_value=saved_scores["val_loss"],
                      epochs_run=len(history), updates=steps, training_seconds=training_seconds,
                      curve=curve_summary(history), device=str(device),
                      train_windows=len(train_batches.dataset), val_windows=len(val_batches.dataset),
                      history=str((attempt / "training_history.csv").relative_to(folder)))
        for key, path in (("checkpoint", checkpoint), ("candidate_model", model_file), ("stats", stats_file)):
            result[key] = str(path.relative_to(folder))
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        result.update(error_type=type(exc).__name__, error=str(exc),
                      failure_kind="oom" if "out of memory" in str(exc).lower() else "training_error")
        atomic_bytes(attempt / "error.txt", traceback.format_exc().encode("utf-8"))
    finally:
        del model, optimizer, train_batches, val_batches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result["elapsed_seconds"] = time.perf_counter() - started
    save_json(attempt / "metrics.json", result)
    training["attempts"][-1]["result"] = result
    update_json(record_path, training=training)
    return result
