# -*- coding: utf-8 -*-
"""Discover provider-specific or pooled rules. Configure the analysis in config.py."""
import argparse
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryFile
import sys
import time

import config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai", nargs="+", choices=(*config.AI_TYPES, "combined"),
                        default=[*config.AI_TYPES, *(["combined"] if config.INCLUDE_COMBINED else [])],
                        help="Default: analyze each provider and then pool. Use combined for pooled analysis only")
    parser.add_argument("--input-dir", type=Path, default=config.INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true", default=config.VERBOSE)
    args = parser.parse_args(argv)

    from build_law_table import run_pipeline

    # Check all selected inputs before starting any provider.
    source_ai_types = config.AI_TYPES if "combined" in args.ai else args.ai
    for ai_type in source_ai_types:
        for horizon in (3, 12):
            for mode in ("univariate", "multivariate"):
                path = args.input_dir / f"{ai_type}_12_{horizon}_{mode}.csv"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing input:{path}")

    for index, ai_type in enumerate(args.ai, 1):
        started = time.monotonic()
        analysis = "pooled" if ai_type == "combined" else "provider-specific"
        print(f"[{index}/{len(args.ai)}] {ai_type}: starting{analysis}rule discovery...", flush=True)
        with TemporaryFile(mode="w+", encoding="utf-8") as log:
            try:
                if args.verbose:
                    result = run_pipeline(args.output_dir, ai_type, args.input_dir)
                else:
                    with redirect_stdout(log):
                        result = run_pipeline(args.output_dir, ai_type, args.input_dir)
            except Exception:
                if not args.verbose:
                    log.seek(0)
                    print("".join(log.readlines()[-25:]), file=sys.stderr)
                raise
        elapsed = time.monotonic() - started
        print(f"  Completed:{result['n_rules']} rules; grades {result['grades']}, "
              f"elapsed {elapsed:.1f} seconds", flush=True)
        print(f"  Law table:{result['law_table']}\n  Motif baseline:{result['module_baseline']}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
