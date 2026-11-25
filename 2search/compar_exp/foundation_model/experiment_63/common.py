"""Shared, dependency-light contract for the formal 63-dataset experiment."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
PROJECT_ROOT = MODEL_ROOT.parent
SETTINGS_PATH = HERE / "settings.json"
TASKS_PATH = MODEL_ROOT / "data" / "tasks.json"


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def object_digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_settings():
    settings = read_json(SETTINGS_PATH)
    context = settings["context_length"]
    offset = settings["target_start_offset"]
    if context != 64 or offset != 12 or context < offset:
        raise ValueError("The formal experiment is frozen to context=64 and target offset=12")
    if settings["horizons"] != [3, 12]:
        raise ValueError("The formal experiment requires horizons [3, 12]")
    for name, model in settings["models"].items():
        patch = model["native_patch_length"]
        if context % patch or context // patch != model["patch_count"]:
            raise ValueError(f"Invalid native-patch contract for {name}")
    return settings


def load_manifest():
    manifest = read_json(TASKS_PATH)
    tasks = manifest.get("tasks", [])
    datasets = sorted({task["dataset"] for task in tasks})
    if manifest.get("dataset_count") != 63 or len(datasets) != 63:
        raise ValueError("Expected exactly 63 datasets")
    if manifest.get("task_count") != 126 or len(tasks) != 126:
        raise ValueError("Expected exactly 126 dataset/horizon tasks")
    expected = {(name, horizon) for name in datasets for horizon in (3, 12)}
    actual = {(task["dataset"], task["horizon"]) for task in tasks}
    if actual != expected:
        raise ValueError("Task manifest is not a complete 63 x 2 grid")
    return manifest, sorted(tasks, key=lambda task: (task["dataset"], task["horizon"]))


def select_tasks(tasks, selector="all", horizons=(3, 12)):
    names = sorted({task["dataset"] for task in tasks})
    if selector == "all":
        selected_names = names
    elif re.fullmatch(r"\d+(?::\d+)?", selector):
        bounds = [int(value) for value in selector.split(":")]
        start, end = bounds[0], bounds[-1]
        if not 1 <= start <= end <= len(names):
            raise ValueError(f"Dataset indices must be within 1..{len(names)}")
        selected_names = names[start - 1:end]
    else:
        selected_names = [name.strip() for name in selector.split(",") if name.strip()]
        unknown = sorted(set(selected_names) - set(names))
        if unknown:
            raise ValueError(f"Unknown datasets: {unknown}")
    horizon_set = set(horizons)
    return [task for task in tasks
            if task["dataset"] in selected_names and task["horizon"] in horizon_set]


def series_view(path: Path):
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim == 1:
        return data[:, None]
    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[2] == 1:
        return data[:, :, 0]
    raise ValueError(f"Unsupported data shape: {data.shape} at {path}")


def split_bounds(length, settings):
    train_end = int(length * settings["train_ratio"])
    val_end = train_end + int(length * settings["val_ratio"])
    return train_end, val_end


def aligned_evaluation_data(path: Path, settings):
    """Return normalized data whose first target is test split start + 12.

    With context=64 this prepends 52 observed validation points, exactly matching
    par_32/hpo/data.py::evaluation_segment for a time_lag=64 configuration.
    """
    data = series_view(path)
    train_end, val_end = split_bounds(len(data), settings)
    context = settings["context_length"]
    offset = settings["target_start_offset"]
    past_context = context - offset
    if val_end < past_context:
        raise ValueError("Insufficient pre-test history for aligned evaluation")
    train = np.asarray(data[:train_end], dtype=np.float32)
    evaluation = np.asarray(data[val_end - past_context:], dtype=np.float32)
    if not np.isfinite(train).all() or not np.isfinite(evaluation).all():
        raise ValueError("Training or aligned evaluation data contain NaN/Inf")
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    normalized = (evaluation - mean) / std
    return normalized, mean, std, train_end, val_end, len(data)


def average_future_steps(values):
    windows, horizon, channels = values.shape
    sums = np.zeros((windows + horizon - 1, channels), dtype=np.float64)
    counts = np.zeros((windows + horizon - 1, 1), dtype=np.float64)
    for index in range(windows):
        sums[index:index + horizon] += values[index]
        counts[index:index + horizon] += 1
    return sums / counts


def metrics(predictions, targets):
    error = predictions - targets
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "wmape": float(np.sum(np.abs(error)) / max(np.sum(np.abs(targets)), 1e-6)),
    }


def data_type(channels):
    return "uni" if channels == 1 else "multi"


def r_run_paths(values=None):
    root = PROJECT_ROOT / "par_32" / "outputs"
    if values:
        paths = []
        for value in values:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = root / value
            paths.append(candidate.resolve())
    else:
        paths = sorted((path.resolve() for path in root.iterdir() if path.is_dir()),
                       key=lambda path: path.name)
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"R run directories do not exist: {missing}")
    return paths


def index_tuned_r(tasks, run_paths, seeds=(2026, 2027, 2028)):
    """Index validation-selected R `best` outputs; never use test-selected recommended outputs."""
    ready, duplicates = {}, {}
    for run in run_paths:
        for task in tasks:
            folder = run / "tasks" / task["dataset"] / f"h{task['horizon']}"
            config_path = folder / "best" / "config.json"
            files = [folder / "best" / f"seed_{seed}" / "test" / "averaged_predictions.npy"
                     for seed in seeds]
            targets = [folder / "best" / f"seed_{seed}" / "test" / "averaged_targets.npy"
                       for seed in seeds]
            metric_files = [folder / "best" / f"seed_{seed}" / "test" / "metrics.json"
                            for seed in seeds]
            if not config_path.is_file() or not all(path.is_file() for path in files + targets + metric_files):
                continue
            key = task["id"]
            record = {"run": run, "folder": folder, "config": read_json(config_path),
                      "prediction_paths": files, "target_paths": targets,
                      "metric_paths": metric_files}
            if key in ready:
                duplicates.setdefault(key, [str(ready[key]["run"])]).append(str(run))
            else:
                ready[key] = record
    return ready, duplicates
