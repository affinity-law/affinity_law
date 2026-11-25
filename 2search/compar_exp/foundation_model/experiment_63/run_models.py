"""Run the 63-dataset TimesFM 3.0 and Moirai 2.0 experiments in sequence.

Run this file directly after reviewing the settings below.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


# ============================== User settings ==============================

# Run name. An existing batch resumes from its completed tasks.
RUN_NAME = "r_aligned_context64_v1"

# "all" selects all 63 datasets. Alternatively, use "1:10", "11:20" for a contiguous range.
DATASETS = "all"

# Evaluate forecast horizons of 3 and 12 steps.
HORIZONS = (3, 12)

# "auto" prefers CUDA when available. Other choices include "cuda" or "cpu".
DEVICE = "auto"

# Reduce the corresponding batch size if GPU memory is insufficient, then resume.
TIMESFM_BATCH_SIZE = 32
MOIRAI_BATCH_SIZE = 256

# The comparison uses overlap-averaged predictions. False saves disk space.
SAVE_WINDOW_ARRAYS = False

# True checks the task range without loading models or predicting.
DRY_RUN = False

# =====================================================================

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
PYTHON_EXE = Path(sys.executable)
RUNNER = HERE / "run_portable.py"

MODELS = (
    ("timesfm3_0", "TimesFM 3.0", TIMESFM_BATCH_SIZE, "two 32-step patches"),
    ("moirai2_0", "Moirai 2.0", MOIRAI_BATCH_SIZE, "four 16-step patches"),
)


def _status(model_name: str) -> dict | None:
    path = MODEL_ROOT / model_name / "outputs" / RUN_NAME / "status.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _run_model(model_name: str, display_name: str, batch_size: int, patch_note: str) -> None:
    if batch_size < 1:
        raise ValueError(f"{display_name} batch size must be positive")

    run_dir = MODEL_ROOT / model_name / "outputs" / RUN_NAME
    command = [
        str(PYTHON_EXE), "-B", "-X", "utf8", str(RUNNER),
        "--model", model_name,
        "--run", RUN_NAME,
        "--datasets", DATASETS,
        "--horizons", *(str(value) for value in HORIZONS),
        "--device", DEVICE,
        "--batch-size", str(batch_size),
    ]
    if run_dir.exists():
        command.append("--resume")
    if SAVE_WINDOW_ARRAYS:
        command.append("--save-window-arrays")
    if DRY_RUN:
        command.append("--dry-run")

    print("\n" + "=" * 72, flush=True)
    print(f"Starting {display_name}: 64 input steps ({patch_note})", flush=True)
    print(f"datasets={DATASETS}, horizons={list(HORIZONS)}, batch size={batch_size}", flush=True)
    if run_dir.exists() and not DRY_RUN:
        print("Found an existing run; checking settings before resuming.", flush=True)
    print("=" * 72, flush=True)

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=HERE, env=environment, check=True)


def main() -> None:
    if not PYTHON_EXE.is_file():
        raise FileNotFoundError(f"Project Python executable not found:{PYTHON_EXE}")
    if not RUNNER.is_file():
        raise FileNotFoundError(f"Internal runner not found:{RUNNER}")

    for model in MODELS:
        _run_model(*model)

    if DRY_RUN:
        print("\nTask range verified. No forecasts were generated.", flush=True)
        return

    print("\nBoth model runs finished:", flush=True)
    for model_name, display_name, _, _ in MODELS:
        status = _status(model_name)
        summary = MODEL_ROOT / model_name / "outputs" / RUN_NAME / "summary.csv"
        if status is None:
            print(f"- {display_name}: status file not found", flush=True)
            continue
        print(
            f"- {display_name}: {status['completed_tasks']}/{status['expected_tasks']} tasks;"
            f"Results:{summary}",
            flush=True,
        )


if __name__ == "__main__":
    main()
