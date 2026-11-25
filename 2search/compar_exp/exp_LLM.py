# -*- coding: utf-8 -*-
"""
Experiment LLM: generate model code directly from dataset features.
===========================================================================
For each test dataset, provide seven meta-features and request a complete PyTorch
CandidateModel without restricting the design to the 24-motif assembler library.
The prompt includes dataset features, interface and shape requirements, and
a short request to"avoid unnecessary complexity"in the generated model. candidate_utils checks syntax and runtime behavior.
Train only after the smoke check succeeds. Each seed requests a model independently and designs may repeat.
Prompts: ../compar_result/models_LLM/<dataset>/h<h>/prompt_s<seed>_a<attempt>.txt

Output: ../compar_result/exp_LLM.csv, one row per dataset, horizon and seed.
Run: python exp_LLM.py [--datasets first5|all|a,b] [--horizons 3] [--seeds 2026]
"""

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_TOP = _HERE.parent
for _p in (_TOP, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402
from common import (discover_datasets, select_datasets, common_argparser,
                    load_meta, ds_base, append_result, already_done)  # noqa: E402
from llm_oneshot import llm_generate_code  # noqa: E402
from train_one import train_test_code  # noqa: E402

ARCH = config.ARCH_LLM          # "llm"
OUT_CSV = config.RESULT_ROOT / "exp_LLM.csv"
RUN_ROOT = config.RESULT_ROOT / "models_LLM"


def _l_param_target(dataset, horizon):
    """Read the selected L parameter count. Only the count is disclosed to the LLM.
    Return None if a parallel L run has yet to record this dataset and horizon.
    Skip it until L finishes. Resuming fills only missing tasks."""
    path = config.RESULT_ROOT / "exp_L.csv"
    if not path.exists():
        raise FileNotFoundError("Complete L before running LLM: exp_L.csv is missing")
    df = pd.read_csv(path, encoding="utf-8-sig")
    rows = df[(df["dataset"] == dataset) & (df["horizon"] == horizon)]
    vals = rows["params"].dropna().unique()
    if len(vals) == 0:
        return None
    if len(vals) != 1:
        raise ValueError(f"{dataset} h{horizon} has inconsistent L parameter counts: {vals.tolist()}")
    return int(vals[0])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = common_argparser("Experiment LLM: generate a model independently for each of three seeds")
    args = args.parse_args()

    datasets = select_datasets(discover_datasets(), args.datasets,
                               limit=args.limit, range_slice=args.range)
    horizons = args.horizons or config.HORIZONS
    seeds = args.seeds or config.SEEDS
    print(f"[LLM] Datasets: {len(datasets)}, horizons {horizons}, seeds {seeds} -> {OUT_CSV}")

    total = 0
    for i, ds in enumerate(datasets, 1):
        print(f"\n===== LLM [{i}/{len(datasets)}] {ds['name']} "
              f"({ds['data_type']}, variables: {ds['station_num']}) =====")
        for h in horizons:
            param_target = _l_param_target(ds["name"], h)
            if param_target is None:
                print(f"  h{h} L has yet to record a parameter count. Skipping until L completes.")
                continue
            for seed in seeds:
                tag = f"{ARCH}_s{seed}"
                dir_h = RUN_ROOT / ds["name"] / f"h{h}"
                if already_done(dir_h, tag):
                    if not any(dir_h.glob(f"prompt_s{seed}_a*.txt")):
                        print(f"  [Warning] h{h} seed={seed} has a completed result but its original prompt is missing;"
                              "the exact prompt cannot be reconstructed")
                    continue
                code, params = llm_generate_code(
                    load_meta(ds["meta_csv"]), h, ds["station_num"],
                    param_target=param_target,
                    prompt_dir=dir_h, prompt_tag=f"s{seed}")
                if code is None:
                    print(f"  h{h} seed={seed} No valid model code after retries; skipping")
                    continue
                if not any(dir_h.glob(f"prompt_s{seed}_a*.txt")):
                    raise RuntimeError(f"{ds['name']} h{h} seed={seed} Prompt was not saved; stopping training")
                print(f"    h{h} seed={seed} params={params} Training...")
                metrics = train_test_code(code, ds["npy_path"], h, seed,
                                          ds["station_num"], dir_h, tag)
                row = {**ds_base(ds), "horizon": h, "arch": ARCH, "seed": seed,
                       "module_A": "LLM-code", "module_B": "LLM-code",
                       "module_C": "LLM-code", "module_E": "LLM-code",
                       "params": params, "param_target": param_target,
                       "param_ratio": params / max(param_target, 1),
                       "param_in_band": (config.LLM_PARAM_LO * param_target <= params <=
                                         config.LLM_PARAM_HI * param_target)}
                if metrics is not None:
                    row.update({"rmse": metrics["rmse"], "mae": metrics["mae"],
                                "wmape": metrics["wmape"]})
                else:
                    row.update({"rmse": None, "mae": None, "wmape": None})
                append_result(OUT_CSV, row)
                total += 1
    print(f"\n[Done] LLM: wrote {total} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
