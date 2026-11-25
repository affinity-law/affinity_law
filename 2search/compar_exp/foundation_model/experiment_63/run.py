"""Run one formal context-64 foundation-model benchmark against the R timeline."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODEL_ROOT / "_runtime"))
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch

import common


def model_class(adapter_path: Path, model_name: str):
    spec = importlib.util.spec_from_file_location(f"formal63_{model_name}", adapter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CandidateModel


def protocol(model_name, model_settings, settings, all_tasks, device):
    adapter = MODEL_ROOT / model_settings["adapter"]
    checkpoint = MODEL_ROOT / model_settings["checkpoint"]
    base = {
        "protocol_version": "r_aligned_context64_v1",
        "model": model_name,
        "context_length": settings["context_length"],
        "native_patch_length": model_settings["native_patch_length"],
        "patch_count": model_settings["patch_count"],
        "horizons": settings["horizons"],
        "train_ratio": settings["train_ratio"],
        "val_ratio": settings["val_ratio"],
        "target_start_offset": settings["target_start_offset"],
        "target_alignment": settings["target_alignment"],
        "normalization": settings["normalization"],
        "aggregation": settings["aggregation"],
        "evaluation_space": settings["evaluation_space"],
        "point_forecast": settings["point_forecast"],
        "window_stride": 1,
        "zero_shot": True,
        "seed_repetitions": 1,
        "inference_seed": 2026,
        "output_contract": "[B, horizon, channels]",
        "channel_mode": model_settings["channel_mode"],
        "task_count": len(all_tasks),
        "task_ids": [task["id"] for task in all_tasks],
        "tasks_manifest_sha256": common.sha256_file(common.TASKS_PATH),
        "settings_sha256": common.sha256_file(common.SETTINGS_PATH),
        "adapter_sha256": common.sha256_file(adapter),
        "runner_sha256": common.sha256_file(Path(__file__)),
        "common_sha256": common.sha256_file(Path(common.__file__)),
        "checkpoint": common.read_json(checkpoint / "provenance.json"),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "execution_device": device,
        "batch_size_note": "Execution setting recorded per task; it may be reduced when resuming after OOM.",
    }
    base["experiment_id"] = common.object_digest(base)
    return base


def cached_result(folder: Path, experiment_id: str, task, data_hash: str):
    path = folder / "metrics.json"
    if not path.is_file():
        return None
    result = common.read_json(path)
    expected = {
        "status": "COMPLETE", "experiment_id": experiment_id,
        "task_id": task["id"], "horizon": task["horizon"],
        "channels": task["channels"], "data_sha256": data_hash,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing task cache does not match: {task['id']}")
    for filename in ("averaged_predictions.npy", "averaged_targets.npy", "target_time_indices.npy"):
        if not (folder / filename).is_file():
            raise ValueError(f"Incomplete task cache: {folder / filename}")
    return result


@torch.inference_mode()
def evaluate(model, model_name, task, settings, data_path, data_hash,
             experiment_id, batch_size, folder, save_window_arrays):
    started = time.perf_counter()
    normalized, mean, std, train_end, val_end, total_length = common.aligned_evaluation_data(
        data_path, settings)
    context = settings["context_length"]
    horizon, channels = task["horizon"], task["channels"]
    if normalized.shape[1] != channels:
        raise ValueError(f"Manifest/data channel mismatch for {task['id']}")
    window_count = len(normalized) - context - horizon + 1
    if window_count < 1:
        raise ValueError(f"Aligned test segment is too short for {task['id']}")
    model.horizon, model.station_num = horizon, channels
    predictions, targets = [], []
    for start in range(0, window_count, batch_size):
        indices = range(start, min(start + batch_size, window_count))
        x = torch.from_numpy(np.stack([normalized[index:index + context] for index in indices]))
        y = np.stack([normalized[index + context:index + context + horizon]
                      for index in indices])
        pred = model(x).detach().cpu().numpy()
        if pred.shape != y.shape or not np.isfinite(pred).all():
            raise ValueError(f"Invalid prediction: expected {y.shape}, got {pred.shape}")
        predictions.append(pred)
        targets.append(y)
        batch_number = start // batch_size + 1
        if start == 0 or batch_number % 20 == 0:
            done = min(start + batch_size, window_count)
            print(f"  {task['id']}: {done}/{window_count} windows", flush=True)
    raw_prediction = np.concatenate(predictions)
    raw_target = np.concatenate(targets)
    window_prediction = raw_prediction * std + mean
    window_target = raw_target * std + mean
    averaged_prediction = common.average_future_steps(raw_prediction) * std + mean
    averaged_target = common.average_future_steps(raw_target) * std + mean
    target_indices = np.arange(val_end + settings["target_start_offset"], total_length)
    expected_points = total_length - val_end - settings["target_start_offset"]
    if averaged_prediction.shape != (expected_points, channels):
        raise ValueError("Averaged output does not cover the R-aligned target interval")
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "averaged_predictions.npy", averaged_prediction)
    np.save(folder / "averaged_targets.npy", averaged_target)
    np.save(folder / "target_time_indices.npy", target_indices)
    np.savez(folder / "normalization.npz", mean=mean, std=std)
    if save_window_arrays:
        np.save(folder / "window_predictions.npy", window_prediction)
        np.save(folder / "window_targets.npy", window_target)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "status": "COMPLETE", "experiment_id": experiment_id,
        "model": model_name, "task_id": task["id"], "dataset": task["dataset"],
        "horizon": horizon, "channels": channels,
        "data_type": common.data_type(channels), "data_sha256": data_hash,
        "train_end": train_end, "val_end": val_end,
        "context_length": context,
        "past_context_points": context - settings["target_start_offset"],
        "target_start_offset": settings["target_start_offset"],
        "target_alignment": settings["target_alignment"],
        "target_start_index": int(target_indices[0]),
        "target_end_index_inclusive": int(target_indices[-1]),
        "target_points": len(target_indices), "window_stride": 1,
        "input_shape": [window_count, context, channels],
        "output_shape": list(raw_prediction.shape),
        "output_contract": "[B, horizon, channels]",
        "overall_metrics": common.metrics(averaged_prediction, averaged_target),
        "raw_window_metrics": common.metrics(window_prediction, window_target),
        "normalization": settings["normalization"],
        "aggregation": settings["aggregation"],
        "evaluation_space": settings["evaluation_space"],
        "point_forecast": settings["point_forecast"],
        "batch_size": batch_size, "window_arrays_saved": save_window_arrays,
        # Keep "params" for compatibility with the R comparison code.  The
        # explicit columns below make the meaning clear in metrics/summary.
        "params": parameter_count,
        "parameter_count": parameter_count,
        "parameter_count_million": round(parameter_count / 1_000_000, 6),
        "elapsed_seconds": time.perf_counter() - started,
    }
    common.write_json(folder / "metrics.json", result)
    return result


def rebuild_reports(run: Path, model_name: str, all_tasks, experiment_id: str):
    rows, completed = [], []
    for task in all_tasks:
        path = run / "tasks" / task["dataset"] / f"h{task['horizon']}" / "metrics.json"
        if not path.is_file():
            continue
        result = common.read_json(path)
        if (result.get("status") != "COMPLETE"
                or result.get("experiment_id") != experiment_id
                or result.get("task_id") != task["id"]):
            raise ValueError(f"Run contains a mismatched task result: {path}")
        completed.append(task["id"])
        parameter_count = result.get("parameter_count", result["params"])
        rows.append({
            "model": model_name, "dataset": task["dataset"],
            "data_type": result["data_type"], "station_num": task["channels"],
            "horizon": task["horizon"], "context_length": result["context_length"],
            "target_start_offset": result["target_start_offset"],
            **result["overall_metrics"],
            "params": parameter_count,
            "parameter_count": parameter_count,
            "parameter_count_million": round(parameter_count / 1_000_000, 6),
            "elapsed_seconds": result["elapsed_seconds"], "mode": "zero_shot",
            "seed": 2026, "aggregation": result["aggregation"],
            "evaluation_space": result["evaluation_space"],
        })
    if rows:
        common.write_csv(run / "summary.csv", rows)
    missing = [task["id"] for task in all_tasks if task["id"] not in set(completed)]
    status = {
        "status": "COMPLETE" if not missing else "PARTIAL",
        "completed_tasks": len(completed), "expected_tasks": len(all_tasks),
        "completed_datasets": len({task_id.rsplit("__h", 1)[0] for task_id in completed}),
        "expected_datasets": 63, "missing_tasks": missing,
    }
    common.write_json(run / "status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("timesfm3_0", "moirai2_0"))
    parser.add_argument("--run", default="r_aligned_context64_v1")
    parser.add_argument("--datasets", default="all",
                        help="all, inclusive range such as 1:10, or comma-separated names")
    parser.add_argument("--horizons", nargs="+", type=int, choices=(3, 12), default=[3, 12])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-window-arrays", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        parser.error("Run name may contain only letters, numbers, underscore and hyphen")
    settings = common.load_settings()
    _, all_tasks = common.load_manifest()
    tasks = common.select_tasks(all_tasks, args.datasets, args.horizons)
    model_settings = settings["models"][args.model]
    batch_size = args.batch_size or model_settings["default_batch_size"]
    if batch_size < 1:
        parser.error("Batch size must be positive")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.dry_run:
        print(f"DRY RUN model={args.model} datasets={len({t['dataset'] for t in tasks})} "
              f"tasks={len(tasks)} context=64 target_offset=12 device={device} batch={batch_size}")
        for task in tasks:
            print(task["id"])
        return
    run = MODEL_ROOT / args.model / "outputs" / args.run
    frozen_protocol = protocol(args.model, model_settings, settings, all_tasks, device)
    protocol_path = run / "protocol.json"
    if run.exists():
        if not args.resume:
            raise FileExistsError(f"Run exists; add --resume or use a new name: {run}")
        if not protocol_path.is_file() or common.read_json(protocol_path) != frozen_protocol:
            raise ValueError("Cannot resume because protocol, code, adapter, settings, or checkpoint changed")
    else:
        common.write_json(protocol_path, frozen_protocol)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    adapter = MODEL_ROOT / model_settings["adapter"]
    checkpoint = MODEL_ROOT / model_settings["checkpoint"]
    print(f"Loading {args.model}: context=64, patches={model_settings['patch_count']}, "
          f"device={device}, selected_tasks={len(tasks)}", flush=True)
    model = model_class(adapter, args.model)(
        horizon=3, station_num=1, checkpoint=checkpoint,
        device=device, batch_size=batch_size).eval()
    for task in tasks:
        folder = run / "tasks" / task["dataset"] / f"h{task['horizon']}"
        data_path = MODEL_ROOT / task["data_path"]
        data_hash = common.sha256_file(data_path)
        result = cached_result(folder, frozen_protocol["experiment_id"], task, data_hash)
        if result is None:
            try:
                result = evaluate(
                    model, args.model, task, settings, data_path, data_hash,
                    frozen_protocol["experiment_id"], batch_size, folder,
                    args.save_window_arrays)
            except Exception as error:
                common.write_json(folder / "failure.json", {
                    "status": "FAILED", "task_id": task["id"],
                    "error_type": type(error).__name__, "error": str(error),
                    "batch_size": batch_size, "device": device,
                })
                rebuild_reports(run, args.model, all_tasks, frozen_protocol["experiment_id"])
                raise
        print(f"COMPLETE {task['id']} {result['overall_metrics']}", flush=True)
        rebuild_reports(run, args.model, all_tasks, frozen_protocol["experiment_id"])
    status = rebuild_reports(run, args.model, all_tasks, frozen_protocol["experiment_id"])
    print(f"Progress: {status['completed_tasks']}/{status['expected_tasks']} tasks", flush=True)
    print(f"Summary: {run / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
