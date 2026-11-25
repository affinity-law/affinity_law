# -*- coding: utf-8 -*-
"""
Train and test one fixed architecture with the 1experiment pipeline.
=============================================================
Workflow
    1. Generate candidate_model.py deterministically from the specification with model_assembler.
    2. Set cfg and train with train_val_early_stop_iterative.main()
       (early stop on val_loss and save best_model_weights.pth with normalization statistics).
    3. Test via test_metrics_iterative.main() and read test_metrics.json
       (overall_metrics.rmse, mae and wmape on the original scale).

Calling main() directly
    run_step executes scripts with exec and snapshots/restores cfg. Here we configure
    each dataset and call main() directly to control state and reduce overhead.

Notes
    - cfg is shared with project_config.py. Reset all required fields before each call.
    - Return None on training or runtime-validation failure so the batch can continue.
"""

import csv
import json
import sys
from pathlib import Path

import config
from model_assembler import write_candidate_model


# Load pipeline modules lazily, once per process.
_cfg = None
_train_mod = None
_test_mod = None


def _load_pipeline():
    global _cfg, _train_mod, _test_mod
    if _cfg is None:
        run_dir = str(config.RUN_PIPELINE_DIR)
        if run_dir not in sys.path:
            sys.path.insert(0, run_dir)
        import project_config as pc
        import train_val_early_stop_iterative as tr
        import test_metrics_iterative as te
        _cfg, _train_mod, _test_mod = pc, tr, te
    return _cfg, _train_mod, _test_mod


def _set_cfg(cfg, npy_path, horizon, seed, station_num, train_dir, test_dir, candidate_path):
    cfg.NPY_PATH = str(npy_path)
    cfg.TIME_LAG = config.TIME_LAG
    cfg.PRE_LEN = horizon
    cfg.BATCH_SIZE = config.BATCH_SIZE
    cfg.EPOCHS = config.EPOCHS
    cfg.LR = config.LR
    cfg.WEIGHT_DECAY = config.WEIGHT_DECAY
    cfg.EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
    cfg.TRAIN_RATIO = config.TRAIN_RATIO
    cfg.VAL_RATIO = config.VAL_RATIO
    cfg.TEST_RATIO = 1.0 - config.TRAIN_RATIO - config.VAL_RATIO
    cfg.SEARCH_SEED = seed
    cfg.UNIVARIATE = (station_num == 1)
    cfg.TRAIN_OUTPUT_DIR = str(train_dir)
    cfg.CANDIDATE_MODEL_PATH = str(candidate_path)
    cfg.TEST_OUTPUT_DIR = str(test_dir)
    cfg.MODEL_WEIGHTS_PATH = str(train_dir / "best_model_weights.pth")
    cfg.TRAIN_STATS_NPZ_PATH = str(train_dir / "train_standardization_stats.npz")
    cfg.SAVE_PREDICTION_PLOTS = False          # Skip plots during batch evaluation.
    cfg.EVALUATE_ON_NORMALIZED_DATA = False    # Metrics on the original scale.


def _train_and_test(candidate_path, npy_path, horizon, seed, station_num, run_dir, tag):
    """Train and test the model at candidate_path. Return overall_metrics or None on failure."""
    cfg, tr_mod, te_mod = _load_pipeline()
    out = Path(run_dir) / tag
    train_dir = out / "train"
    test_dir = out / "test"

    _set_cfg(cfg, npy_path, horizon, seed, station_num, train_dir, test_dir, candidate_path)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        tr_mod.main()
        te_mod.main()
    except Exception as e:
        print(f"    [train_test FAIL] {tag} seed={seed} pre={horizon}: {type(e).__name__}: {str(e)[:150]}")
        return None

    metrics_path = test_dir / "test_metrics.json"
    if not metrics_path.exists():
        print(f"    [train_test FAIL] test_metrics.json was not generated: {metrics_path}")
        return None
    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)
    return m.get("overall_metrics")


def train_test(spec, npy_path, horizon, seed, station_num, run_dir, tag):
    """L: assemble the specification and then train and test."""
    out = Path(run_dir) / tag
    candidate_path = out / "candidate_model.py"
    out.mkdir(parents=True, exist_ok=True)
    write_candidate_model(spec, config.TIME_LAG, horizon, station_num, candidate_path)
    return _train_and_test(candidate_path, npy_path, horizon, seed, station_num, run_dir, tag)


def train_test_code(code, npy_path, horizon, seed, station_num, run_dir, tag):
    """LLM: save generated source and then train and test."""
    out = Path(run_dir) / tag
    candidate_path = out / "candidate_model.py"
    out.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(code, encoding="utf-8")
    return _train_and_test(candidate_path, npy_path, horizon, seed, station_num, run_dir, tag)


def quick_val_rmse(spec, npy_path, horizon, seed, station_num, run_dir, tag,
                   quick_epochs=40, quick_patience=5):
    """Screen a candidate with short training and return minimum validation RMSE without using test data."""
    cfg, tr_mod, _ = _load_pipeline()
    out = Path(run_dir) / tag
    train_dir = out / "train"
    test_dir = out / "test"
    candidate_path = out / "candidate_model.py"

    write_candidate_model(spec, config.TIME_LAG, horizon, station_num, candidate_path)
    _set_cfg(cfg, npy_path, horizon, seed, station_num, train_dir, test_dir, candidate_path)
    cfg.EPOCHS = quick_epochs
    cfg.EARLY_STOP_PATIENCE = quick_patience

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        tr_mod.main()
    except Exception:
        return float("inf")

    hist = train_dir / cfg.TRAIN_HISTORY_FILE
    if not hist.exists():
        return float("inf")
    best = float("inf")
    with open(hist, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                v = float(row.get("val_rmse", "inf"))
            except (TypeError, ValueError):
                v = float("inf")
            best = min(best, v)
    return best


if __name__ == "__main__":
    # Self-test: train a default architecture on a small NPY dataset.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from select_arch import default_arch
    from common import discover_datasets
    datasets = [ds for ds in discover_datasets() if ds["data_type"] == "uni"]
    if not datasets:
        raise FileNotFoundError(f"No univariate dataset available for the self-test:{config.TEST_DATA_ROOT}")
    npy = datasets[0]["npy_path"]
    print("Self-test NPY:", npy)
    import numpy as np
    a = np.load(npy)
    station_num = a.shape[1] if a.ndim == 2 else 1
    spec = default_arch()
    print("spec:", spec)
    res = train_test(spec, npy, 3, 2026, station_num, config.RESULT_ROOT / "_selftest", "default")
    print("overall_metrics:", res)
