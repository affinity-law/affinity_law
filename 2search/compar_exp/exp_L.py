# -*- coding: utf-8 -*-
"""
Experiment L: architecture recommendation from affinity laws.
===========================================================
For each test dataset, select stage-specific motifs from the law table.
    Rank matching rules by grade and robust effect size within each stage.
    Use the most frequent stage motif if no A, B or C rule matches.
    Use Single Block and one B if E has no match; a matching composition rule uses two B blocks.
Assemble with the original model_assembler implementations and default hyperparameters.
Train and test each architecture with three seeds, recording RMSE, MAE, WMAPE,
parameter counts and selected motifs. summary.py aggregates seed statistics.

Output: ../compar_result/exp_L.csv, one row per dataset, horizon and seed.
Run: python exp_L.py [--datasets first10|all|a,b] [--horizons 3] [--seeds 2026]
"""

import shutil
import csv
import hashlib
import json
import math
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
                    already_done)  # noqa: E402
from select_arch import load_law, load_overview, law_recommended, law_candidates  # noqa: E402
from rule_revisions import equivalent_selection
from model_assembler import count_params_for_spec  # noqa: E402
from train_one import train_test, quick_val_rmse  # noqa: E402

ARCH = config.ARCH_LAW          # "law"
OUT_CSV = config.RESULT_ROOT / "exp_L.csv"
RUN_ROOT = config.RESULT_ROOT / "models_L"


def build(ds, horizon, law=None):
    """Select an L candidate on validation data and return (best_spec, params)."""
    meta = load_meta(ds["meta_csv"])
    pair_dir = RUN_ROOT / ds['name'] / f'h{horizon}'
    cache = pair_dir / 'selection.json'
    sources = {'rules': config.LAW_TABLE, 'pooled': config.POOLED_LAW_TABLE,
               'features': Path(ds['meta_csv']), 'data': Path(ds['npy_path']),
               **{name: _HERE / name for name in ('select_arch.py', 'model_assembler.py', 'config.py', 'train_one.py')}}
    signature = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in sources.items()}
    signature['baseline'] = hashlib.sha256(load_overview().to_csv(index=False).encode('utf-8')).hexdigest()
    if cache.exists():
        saved = json.loads(cache.read_text(encoding='utf-8'))
        if equivalent_selection(saved['source_sha256']) != equivalent_selection(signature):
            raise ValueError('Existing selection uses different rules, data or code. Use a new output directory or back up the previous L results.')
        return saved['spec'], saved['params']
    cands = law_candidates(meta, horizon, law=law)
    sel_dir = RUN_ROOT / ds["name"] / f"h{horizon}" / "_select"

    best_spec, best_val = None, float("inf")
    evaluated = []
    for i, cand in enumerate(cands):
        v = quick_val_rmse(cand, ds["npy_path"], horizon, 2026,
                           ds["station_num"], sel_dir, f"cand{i}")
        print(f"    candidate{i}: A={cand.get('A')} B={cand.get('B')} "
              f"C={cand.get('C')} E={cand.get('E')} val_rmse={v:.4f}")
        evaluated.append({'spec': cand, 'val_rmse': v if math.isfinite(v) else None})
        if v < best_val:
            best_val, best_spec = v, cand

    if sel_dir.exists():
        resolved = sel_dir.resolve()
        if not resolved.is_relative_to(RUN_ROOT.resolve()) or resolved.name != '_select':
            raise ValueError('Candidate temporary directory is outside the current L output directory')
        shutil.rmtree(resolved)

    if best_spec is None:          # All candidate screening runs failed; using the single recommendation
        best_spec = law_recommended(meta, horizon, law=law)

    params = count_params_for_spec(best_spec, config.TIME_LAG, horizon, ds["station_num"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({'source_sha256': signature, 'spec': best_spec, 'params': params,
                                 'best_val_rmse': best_val if math.isfinite(best_val) else None,
                                 'candidate_results': evaluated}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"    Selected: A={best_spec.get('A')} B={best_spec.get('B')} "
          f"C={best_spec.get('C')} E={best_spec.get('E')} val_rmse={best_val:.4f}")
    return best_spec, params


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = common_argparser("Experiment L: rule-guided selection with three seeds")
    args.add_argument('--output-csv', type=Path, default=OUT_CSV,
                      help='Separate summary CSV for parallel dataset shards; model files still go to models_L')
    args = args.parse_args()
    out_csv = args.output_csv.resolve()

    datasets = select_datasets(discover_datasets(), args.datasets,
                               limit=args.limit, range_slice=args.range)
    horizons = args.horizons or config.HORIZONS
    seeds = args.seeds or config.SEEDS
    law = load_law()  # Stop on invalid input before reporting skipped tasks as complete.
    print(f"[L] Rule source: {config.LAW_TABLE} ({len(law)} rules)")
    print(f"[L] Datasets: {len(datasets)}, horizons {horizons}, seeds {seeds} -> {out_csv}")
    completed = set()
    if out_csv.exists():
        with out_csv.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get('rmse') and math.isfinite(float(row['rmse'])):
                    completed.add((row['dataset'], int(row['horizon']), int(row['seed'])))

    total = 0
    for i, ds in enumerate(datasets, 1):
        print(f"\n===== L [{i}/{len(datasets)}] {ds['name']} "
              f"({ds['data_type']}, variables: {ds['station_num']}) =====")
        for h in horizons:
            try:
                spec, params = build(ds, h, law=law)
            except Exception as e:
                print(f"  h{h} Rule-based recommendation failed: {type(e).__name__}: {str(e)[:120]}")
                continue
            print(f"  h{h} spec: A={spec.get('A')} B={spec.get('B')} "
                  f"C={spec.get('C')} E={spec.get('E')} params={params}")
            for seed in seeds:
                tag = f"{ARCH}_s{seed}"
                done = already_done(RUN_ROOT / ds["name"] / f"h{h}", tag)
                if done and (ds['name'], h, seed) in completed:
                    continue
                print(f"    [{ds['name']} h{h}] {ARCH} seed={seed} ...")
                if done:
                    metrics_path = RUN_ROOT / ds['name'] / f'h{h}' / tag / 'test/test_metrics.json'
                    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))['overall_metrics']
                else:
                    metrics = train_test(spec, ds["npy_path"], h, seed,
                                         ds["station_num"], RUN_ROOT / ds["name"] / f"h{h}", tag)
                row = {**ds_base(ds), "horizon": h, "arch": ARCH, "seed": seed,
                       **spec_modules(spec), "params": params}
                if metrics is not None:
                    row.update({"rmse": metrics["rmse"], "mae": metrics["mae"],
                                "wmape": metrics["wmape"]})
                else:
                    row.update({"rmse": None, "mae": None, "wmape": None})
                append_result(out_csv, row)
                total += 1
    print(f"\n[Done] L: wrote {total} rows to {out_csv}")


if __name__ == "__main__":
    main()
