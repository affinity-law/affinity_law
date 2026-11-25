# -*- coding: utf-8 -*-
"""
Experiment F: frequency-based architecture baseline.
============================================================
Choose the motifs with the highest training selection rate (p_base) by stage:
    A=Patch, B=[TCN Block, Linear Block] (the top two because Gated Fusion requires two B blocks).
    C=MLP Head, E=Gated Fusion.
Scale B/C widths to match the L capacity target from ../compar_result/exp_L.csv for each task.
F and LLM use the same target so that differences can be attributed to"motif selection"independently of"model capacity".
Train and test the architecture with three seeds, recording RMSE, MAE, WMAPE, parameters and motifs.

Output: ../compar_result/exp_F.csv with one row per dataset, horizon and seed; artifacts in models_F/.
Run: python exp_F.py [--datasets first10|all|a,b] [--horizons 3] [--seeds 2026]
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TOP = _HERE.parent
for _p in (_TOP, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402
from common import (discover_datasets, select_datasets, common_argparser,
                    load_meta, ds_base, spec_modules, append_result,
                    already_done, l_param_target)  # noqa: E402
from select_arch import frequency_default_arch  # noqa: E402
from model_assembler import scale_to_target_params  # noqa: E402
from train_one import train_test  # noqa: E402

ARCH = config.ARCH_DEFAULT          # "default"
OUT_CSV = config.RESULT_ROOT / "exp_F.csv"
RUN_ROOT = config.RESULT_ROOT / "models_F"


def build(ds, horizon):
    """Scale the frequency-based F architecture to p_L and return (scaled_spec, params)."""
    p_target = l_param_target(ds["name"], horizon)
    if p_target is None:
        return None, None, None
    spec = frequency_default_arch()
    scaled, params = scale_to_target_params(spec, config.TIME_LAG, horizon,
                                            ds["station_num"], p_target)
    return scaled, params, p_target


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = common_argparser("Experiment F: most frequent motifs per stage, matched to L capacity, three seeds")
    args = args.parse_args()

    datasets = select_datasets(discover_datasets(), args.datasets,
                               limit=args.limit, range_slice=args.range)
    horizons = args.horizons or config.HORIZONS
    seeds = args.seeds or config.SEEDS
    print(f"[F] Datasets: {len(datasets)}, horizons {horizons}, seeds {seeds} -> {OUT_CSV}")

    total = 0
    for i, ds in enumerate(datasets, 1):
        print(f"\n===== F [{i}/{len(datasets)}] {ds['name']} "
              f"({ds['data_type']}, variables: {ds['station_num']}) =====")
        for h in horizons:
            spec, params, p_target = build(ds, h)
            if spec is None:
                print(f"  h{h} L has yet to record a parameter count. Skipping until L completes.")
                continue
            print(f"  h{h} spec: A={spec.get('A')} B={spec.get('B')} "
                  f"C={spec.get('C')} E={spec.get('E')} "
                  f"params={params} (p_L={p_target}, ratio={params / max(p_target, 1):.3f})")
            for seed in seeds:
                tag = f"{ARCH}_s{seed}"
                if already_done(RUN_ROOT / ds["name"] / f"h{h}", tag):
                    continue
                print(f"    [{ds['name']} h{h}] {ARCH} seed={seed} ...")
                metrics = train_test(spec, ds["npy_path"], h, seed,
                                     ds["station_num"], RUN_ROOT / ds["name"] / f"h{h}", tag)
                row = {**ds_base(ds), "horizon": h, "arch": ARCH, "seed": seed,
                       **spec_modules(spec), "params": params,
                       "param_target": p_target,
                       "param_ratio": params / max(p_target, 1)}
                if metrics is not None:
                    row.update({"rmse": metrics["rmse"], "mae": metrics["mae"],
                                "wmape": metrics["wmape"]})
                else:
                    row.update({"rmse": None, "mae": None, "wmape": None})
                append_result(OUT_CSV, row)
                total += 1
    print(f"\n[Done] F: wrote {total} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
