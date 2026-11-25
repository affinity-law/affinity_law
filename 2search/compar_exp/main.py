# -*- coding: utf-8 -*-
"""Comparison entry point. exp_L/R/F/LLM and summary can also run separately."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys

import config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law-table", type=Path, help="Law table (default connect/document_summary.csv)")
    parser.add_argument("--experiments", nargs="+", choices=("L", "F", "R", "LLM"),
                        default=list(config.EXPERIMENTS))
    parser.add_argument("--datasets", default=config.DATASETS)
    parser.add_argument("--horizons", nargs="+", type=int, default=config.HORIZONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    parser.add_argument("--limit", type=int, default=config.DATASET_LIMIT)
    parser.add_argument("--range", nargs=2, type=int, default=config.DATASET_RANGE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Check dataset and law paths without training or API calls")
    args = parser.parse_args(argv)

    if args.law_table is not None:
        os.environ["TSF_LAW_TABLE"] = str(args.law_table.resolve())
    if args.output_dir is not None:
        os.environ["TSF_RESULT_ROOT"] = str(args.output_dir.resolve())
    importlib.reload(config)

    if args.check_only:
        from common import discover_datasets, select_datasets
        from select_arch import load_law, load_overview
        datasets = select_datasets(discover_datasets(), args.datasets, args.limit, args.range)
        print(f"Rules={config.LAW_TABLE}; test datasets={len(datasets)}; "
              f"Deployable rules={len(load_law())}; motif baselines={len(load_overview())}")
        print(f"Output directory:{config.RESULT_ROOT}")
        return

    here = Path(__file__).resolve().parent
    shared = ["--datasets", args.datasets,
              "--horizons", *map(str, args.horizons),
              "--seeds", *map(str, args.seeds),
              "--limit", str(args.limit), "--range", *map(str, args.range)]
    for experiment in args.experiments:
        subprocess.run([sys.executable, str(here / f"exp_{experiment}.py"), *shared], check=True)
    if config.RUN_SUMMARY and not args.skip_summary:
        subprocess.run([sys.executable, str(here / "summary.py")], check=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
