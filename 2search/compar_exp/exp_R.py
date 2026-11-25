# -*- coding: utf-8 -*-
"""
Experiment R: random architecture baseline.
====================================================================
Sample from the complete motif library at each stage:
    Sample 0-2 preprocessing motifs for A, one forecast head for C and one composition for E.
    Match B count to L: one for Single Block, two distinct B blocks for other compositions.
Draw one architecture per dataset and horizon using random.Random(f"{name}|h{h}") as the seed.
The draw is reproducible. Adjust width_scale to the capacity target p_L from exp_L.csv.
Train and test the same architecture with three seeds.

Output: ../compar_result/exp_R.csv, one row per dataset, horizon and seed; artifacts in models_R/.
Run: python exp_R.py [--datasets first10|all|a,b] [--horizons 3] [--seeds 2026]
"""

import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TOP = _HERE.parent
for _p in (_TOP, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402
from common import (discover_datasets, select_datasets, common_argparser,
                    ds_base, spec_modules, append_result,
                    already_done, l_param_target)  # noqa: E402
from model_assembler import scale_to_target_params  # noqa: E402
from train_one import train_test  # noqa: E402

ARCH = config.ARCH_RANDOM          # "random"
OUT_CSV = config.RESULT_ROOT / "exp_R.csv"
RUN_ROOT = config.RESULT_ROOT / "models_R"

# Full motif library, matching the 24 assembler motifs. Stage D is inferred rather than"selected")
A_POOL = ["Identity", "Z-score", "RevIN", "Differencing",
          "Series Decomposition", "Patch", "Mask"]
B_POOL = ["Linear Block", "MLP Block", "CNN Block", "TCN Block", "GRU Block",
          "LSTM Block", "State Space Model Block", "Transformer Encoder Block",
          "Basis Block"]
C_POOL = ["Linear Head", "MLP Head", "Basis Projection"]
E_POOL = ["Single Block", "Sequential Composition", "Parallel Additive Branches",
          "Residual Modeling", "Gated Fusion"]


def random_arch(rng):
    """Sample a random architecture: 0-2 A motifs and one C and E motif each.
    Match B count to E using _fit_b_to_e: one for Single Block, two without replacement for compositions."""
    e = rng.choice(E_POOL)
    n_b = 1 if e == "Single Block" else 2
    return {
        "A": rng.sample(A_POOL, rng.randint(0, 2)),
        "B": rng.sample(B_POOL, n_b),
        "C": rng.choice(C_POOL),
        "E": e,
    }


def build(ds, horizon):
    """Sample R using a fixed seed and scale to p_L. Return (scaled_spec, params)."""
    p_target = l_param_target(ds["name"], horizon)
    if p_target is None:
        return None, None, None
    rng = random.Random(f"{ds['name']}|h{horizon}")   # Use a reproducible seed per dataset and horizon.
    spec = random_arch(rng)
    scaled, params = scale_to_target_params(spec, config.TIME_LAG, horizon,
                                            ds["station_num"], p_target)
    return scaled, params, p_target


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = common_argparser("Experiment R: random architecture with B count determined by E, capacity matched to L, three seeds")
    args = args.parse_args()

    datasets = select_datasets(discover_datasets(), args.datasets,
                               limit=args.limit, range_slice=args.range)
    horizons = args.horizons or config.HORIZONS
    seeds = args.seeds or config.SEEDS
    print(f"[R] Datasets: {len(datasets)}, horizons {horizons}, seeds {seeds} -> {OUT_CSV}")

    total = 0
    for i, ds in enumerate(datasets, 1):
        print(f"\n===== R [{i}/{len(datasets)}] {ds['name']} "
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
    print(f"\n[Done] R: wrote {total} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
