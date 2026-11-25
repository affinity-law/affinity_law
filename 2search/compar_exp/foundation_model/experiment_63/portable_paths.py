"""Current project paths; resolve frozen inference snapshots in memory."""
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import common

MODEL_ROOT = Path(__file__).resolve().parent.parent
COMPAR_EXP_ROOT = MODEL_ROOT.parent
SEARCH_ROOT = COMPAR_EXP_ROOT.parent
PROJECT_ROOT = SEARCH_ROOT.parent
TASKS_PATH = MODEL_ROOT / "tasks.json"
TEST_DATA_ROOT = PROJECT_ROOT / "0dataset" / "test"
TUNING_ROOT = PROJECT_ROOT / "3par"
RESULT_ROOT = SEARCH_ROOT / "compar_result"
L_MODEL_ROOT = RESULT_ROOT / "models_L"
L_SEEDS = (2026, 2027, 2028)
FOUNDATION_RUN = "r_aligned_context64_v1"  # Historical batch identifier, not the R method.


@lru_cache(maxsize=1)
def data_index():
    paths = {}
    for path in TEST_DATA_ROOT.rglob("*.npy"):
        if path.stem in paths:
            raise ValueError(f"Duplicate test dataset: {path.stem}")
        paths[path.stem] = path.resolve()
    return paths


def normalize_task_paths(task):
    """Resolve current input/model locations; keep historical metadata unchanged."""
    result = dict(task)
    dataset = task["dataset"]
    if dataset not in data_index():
        raise FileNotFoundError(f"Missing dataset under {TEST_DATA_ROOT}: {dataset}")
    folder = L_MODEL_ROOT / dataset / f"h{task['horizon']}" / f"law_s{task['source_seed']}"
    result.update(data_path=str(data_index()[dataset]),
                  model_path=str(folder / "candidate_model.py"),
                  weights_path=str(folder / "train/best_model_weights.pth"),
                  source_model=str(folder / "candidate_model.py"))
    return result


def normalize_manifest_paths(manifest):
    """Return a comparable copy without changing the frozen source manifest."""
    result = dict(manifest)
    result["source_root"] = str(PROJECT_ROOT)
    result["law_table_path"] = str(SEARCH_ROOT / "connect/document_summary.csv")
    result["tasks"] = [normalize_task_paths(task) for task in manifest["tasks"]]
    return result


def load_manifest():
    """Use the original manifest validation with the relocated snapshot."""
    original_path = common.TASKS_PATH
    common.TASKS_PATH = TASKS_PATH
    try:
        manifest, tasks = common.load_manifest()
    finally:
        common.TASKS_PATH = original_path
    return normalize_manifest_paths(manifest), [normalize_task_paths(task) for task in tasks]


def shared_input_signature(manifest):
    """Compare evaluation inputs; historical and current L architectures may differ."""
    return sorted((task["id"], task["dataset"], task["horizon"], task["length"],
                   task["channels"], str(Path(task["data_path"]).resolve()))
                  for task in normalize_manifest_paths(manifest)["tasks"])


def l_run_paths(values=None):
    """Optional tuned-L batches under all/3par/outputs."""
    root = TUNING_ROOT / "outputs"
    if values:
        paths = []
        for value in values:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = root / value
            paths.append(candidate.resolve())
    else:
        paths = (sorted((path.resolve() for path in root.iterdir() if path.is_dir()),
                        key=lambda path: path.name) if root.is_dir() else [])
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Tuned L run directories do not exist: {missing}")
    return paths


def index_tuned_l(tasks, run_paths, seeds=L_SEEDS):
    return common.index_tuned_r(tasks, run_paths, seeds)


def load_l_reference():
    """Aggregate current exp_L.csv instead of the obsolete historical R table."""
    path = RESULT_ROOT / "exp_L.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        key = f"{row['dataset']}__h{int(row['horizon'])}"
        seed = int(row["seed"])
        by_seed = grouped.setdefault(key, {})
        if seed in by_seed:
            raise ValueError(f"Duplicate L result: {key}/seed{seed}")
        if not all(np.isfinite(float(row[m])) for m in ("rmse", "mae", "wmape")):
            raise ValueError(f"Non-finite L result: {key}/seed{seed}")
        by_seed[seed] = row
    index = {}
    for key, by_seed in grouped.items():
        if set(by_seed) != set(L_SEEDS):
            raise ValueError(f"L seed set must be {L_SEEDS}: {key}")
        params = {int(float(row["params"])) for row in by_seed.values()}
        if len(params) != 1:
            raise ValueError(f"Inconsistent L parameter counts: {key}")
        entry = {"params": params.pop(), "seeds": L_SEEDS, "seed_count": len(L_SEEDS), "seed_rows": by_seed}
        for metric in ("rmse", "mae", "wmape"):
            values = [float(by_seed[seed][metric]) for seed in L_SEEDS]
            entry[f"{metric}_mean"] = float(np.mean(values))
            entry[f"{metric}_std"] = float(np.std(values, ddof=1))
        index[key] = entry
    return path, index
