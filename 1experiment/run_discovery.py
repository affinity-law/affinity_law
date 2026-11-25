"""Search selected training datasets, or use --npy_path/--npy_dir for explicit inputs.

Use --npy_path FILE --help to see the detailed experiment options.
"""
import argparse
from contextlib import nullcontext
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import project_config as cfg
from prepare_numeric_input import compatible_input

ROOT = Path(__file__).resolve().parents[1]


def run_selected(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai", "--llm_provider", choices=(*cfg.AI_TYPES, "openai"),
                        default=cfg.LLM_PROVIDER, help="Select chatgpt, deepseek or gemini")
    parser.add_argument("--datasets", default="first1", help="all, firstN, or comma-separated dataset names")
    parser.add_argument("--horizons", nargs="+", type=int, choices=(3, 12), default=[3, 12])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="List work without training or LLM calls")
    args, experiment_args = parser.parse_known_args(argv)
    ai_type = "chatgpt" if args.ai == "openai" else args.ai
    files = sorted((ROOT / "0dataset" / "train").rglob("*.npy"), key=lambda path: path.name)
    if args.datasets.startswith("first"):
        count = int(args.datasets[5:])
        if count < 1:
            parser.error("firstN requires N >= 1")
        files = files[:count]
    elif args.datasets != "all":
        requested = set(args.datasets.split(","))
        missing = requested - {p.stem for p in files}
        if missing:
            parser.error("Unknown datasets: " + ", ".join(sorted(missing)))
        files = [p for p in files if p.stem in requested]
    for horizon in args.horizons:
        for file in files:
            with file.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                shape, _, dtype = np.lib.format._read_array_header(stream, version)
            channels = shape[1] if len(shape) >= 2 else 1
            mode = "--univariate" if channels == 1 else "--multivariate"
            output = args.output_root.resolve() / f"12_{horizon}"
            print(f"[{ai_type}] {file.stem}: H={horizon}, channels={channels}, output={output}", flush=True)
            if args.dry_run:
                continue
            env = os.environ.copy()
            env.update(TIME_LAG="12", PRE_LEN=str(horizon), UNIVARIATE="1" if channels == 1 else "0",
                       RESULT_ROOT=str(output))
            with (compatible_input(file) if dtype.hasobject else nullcontext(file)) as prepared:
                subprocess.run([sys.executable, "-B", "-X", "utf8", str(Path(__file__).resolve()),
                                "--ai", ai_type,
                                "--npy_path", str(prepared), "--time_lag", "12", "--pre_len", str(horizon),
                                "--output_dir", str(output), mode, *experiment_args], env=env, check=True)


def run_explicit(argv=None):
    """Run one file or a directory using the original experiment arguments."""
    parser = argparse.ArgumentParser(
        description="LLM-driven time-series forecasting model design framework",
    )

    # ── Data source (mutually exclusive) ──
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--npy_path", type=str, default=None,
                            help="Path to a single .npy file (single-file mode)")
    data_group.add_argument("--npy_dir", type=str, default=None,
                            help="Directory containing .npy files (batch mode)")

    # ── Output ──
    parser.add_argument("--output_dir", type=str, default="./result",
                        help="Result output root directory")

    # ── Task hyperparameters ──
    parser.add_argument("--time_lag", type=int, default=cfg.TIME_LAG,
                        help="Input window length (default: 12)")
    parser.add_argument("--pre_len", type=int, default=cfg.PRE_LEN,
                        help="Prediction horizon (default: 3)")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE,
                        help="Training batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS,
                        help="Maximum training epochs (default: 1000)")
    parser.add_argument("--lr", type=float, default=cfg.LR,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=cfg.WEIGHT_DECAY,
                        help="Weight decay (default: 0)")
    parser.add_argument("--early_stop_patience", type=int, default=cfg.EARLY_STOP_PATIENCE,
                        help="Early stopping patience (default: 15)")

    # ── Data split ──
    parser.add_argument("--train_ratio", type=float, default=cfg.TRAIN_RATIO,
                        help="Training set ratio (default: 0.6)")
    parser.add_argument("--val_ratio", type=float, default=cfg.VAL_RATIO,
                        help="Validation set ratio (default: 0.2)")

    # ── Variable mode ──
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--univariate", action="store_true", default=None,
                            help="Univariate mode (default)")
    mode_group.add_argument("--multivariate", action="store_true", default=None,
                            help="Multivariate mode")

    # ── Seasonal period ──
    parser.add_argument("--seasonal_period", type=int, default=cfg.SEASONAL_PERIOD,
                        help="Seasonal period (0=auto-estimate, default: 0)")

    # ── Iteration control ──
    parser.add_argument("--max_iterations", type=int, default=cfg.MAX_ITERATIONS,
                        help="Maximum iterations per dataset (default: 100)")
    parser.add_argument("--min_iterations", type=int, default=cfg.MIN_ITERATIONS_BEFORE_STOP,
                        help="Minimum iterations before early stop (default: 10)")
    parser.add_argument("--iteration_patience", type=int, default=cfg.ITERATION_PATIENCE,
                        help="Early stop patience in iterations (default: 8)")
    parser.add_argument("--max_npy_files", type=int, default=cfg.MAX_NPY_FILES,
                        help="Maximum .npy files to ACTUALLY PROCESS (0=all, skips already-completed, default: 0)")

    # ── Seed & stability ──
    parser.add_argument("--search_seed", type=int, default=cfg.SEARCH_SEED,
                        help="Phase 1 search seed (default: 2026)")
    parser.add_argument("--stability_seeds", type=int, nargs="+", default=cfg.STABILITY_SEEDS,
                        help="Phase 2 stability seeds (default: 2026 2027 2028)")
    parser.add_argument("--stability_top_k", type=int, default=cfg.STABILITY_TOP_K,
                        help="Number of top models for stability re-runs (default: 3)")

    # ── LLM ──
    parser.add_argument("--ai", "--llm_provider", choices=(*cfg.AI_TYPES, "openai"),
                        default=cfg.LLM_PROVIDER, help="Select chatgpt, deepseek or gemini")
    parser.add_argument("--llm_api_key", type=str, default="",
                        help="LLM API key")
    parser.add_argument("--llm_base_url", type=str, default="",
                        help="LLM API base URL")
    parser.add_argument("--llm_model", type=str, default="",
                        help="LLM model name")
    parser.add_argument("--llm_temperature", type=float, default=cfg.LLM_TEMPERATURE,
                        help="LLM temperature (default: 0.4)")

    # ── Proxy ──
    parser.add_argument("--http_proxy", type=str, default="",
                        help="HTTP proxy URL")
    parser.add_argument("--https_proxy", type=str, default="",
                        help="HTTPS proxy URL")

    # ── Other ──
    parser.add_argument("--top_k", type=int, default=cfg.TOP_K,
                        help="Number of top candidates to record (default: 5)")
    parser.add_argument("--top_k_sort_metric", type=str, default=cfg.TOP_K_SORT_METRIC,
                        choices=["val_rmse", "val_mae", "val_loss", "rmse", "mae", "wmape"],
                        help="Metric for sorting top-k candidates (default: rmse)")

    args = parser.parse_args(argv)

    # ── Apply CLI overrides to project_config ──
    cfg.TIME_LAG = args.time_lag
    cfg.PRE_LEN = args.pre_len
    cfg.BATCH_SIZE = args.batch_size
    cfg.EPOCHS = args.epochs
    cfg.LR = args.lr
    cfg.WEIGHT_DECAY = args.weight_decay
    cfg.EARLY_STOP_PATIENCE = args.early_stop_patience
    cfg.TRAIN_RATIO = args.train_ratio
    cfg.VAL_RATIO = args.val_ratio
    cfg.TEST_RATIO = 1.0 - args.train_ratio - args.val_ratio
    cfg.MAX_ITERATIONS = args.max_iterations
    cfg.MIN_ITERATIONS_BEFORE_STOP = args.min_iterations
    cfg.ITERATION_PATIENCE = args.iteration_patience
    cfg.SEARCH_SEED = args.search_seed
    cfg.STABILITY_SEEDS = list(args.stability_seeds)
    cfg.STABILITY_TOP_K = args.stability_top_k
    cfg.TOP_K = args.top_k
    cfg.TOP_K_SORT_METRIC = args.top_k_sort_metric
    cfg.SEASONAL_PERIOD = args.seasonal_period
    cfg.MAX_NPY_FILES = args.max_npy_files

    if args.univariate is not None:
        cfg.UNIVARIATE = args.univariate
    elif args.multivariate is not None:
        cfg.UNIVARIATE = not args.multivariate

    if args.output_dir:
        abs_output = os.path.abspath(args.output_dir)
        cfg.RESULT_ROOT = abs_output

    cfg.configure_llm(args.ai)
    if args.llm_api_key:
        cfg.LLM_API_KEY = args.llm_api_key
    if args.llm_base_url:
        cfg.LLM_BASE_URL = args.llm_base_url
    if args.llm_model:
        cfg.LLM_MODEL = args.llm_model
    if args.llm_temperature is not None:
        cfg.LLM_TEMPERATURE = args.llm_temperature
    if args.http_proxy:
        cfg.HTTP_PROXY = args.http_proxy
    if args.https_proxy:
        cfg.HTTPS_PROXY = args.https_proxy
    cfg.apply_proxy_settings()

    if not str(cfg.LLM_API_KEY).strip():
        parser.error(f"Set the API key in project_config.py for {cfg.LLM_PROVIDER} or set the corresponding environment variable.")
    print(f"[llm] provider={cfg.LLM_PROVIDER}, model={cfg.LLM_MODEL}, base_url={cfg.LLM_BASE_URL}")
    from run_iterative_loop import discover_npy_files, run_experiment_for_npy

    # ── Determine mode: batch (--npy_dir) vs single-file (--npy_path) ──
    if args.npy_dir:
        # Batch mode
        data_dir = os.path.abspath(args.npy_dir)
        cfg.DATASET_NPY_DIR = data_dir
        cfg.DATASET_CATEGORY = os.path.basename(data_dir)
        npy_files = discover_npy_files(data_dir)
        npy_total = len(npy_files)

        # ── Limit number of npy files ACTUALLY PROCESSED (skip already-completed) ──
        max_npy = args.max_npy_files
        if max_npy > 0:
            print(f"[config] --max_npy_files={max_npy}: will process up to {max_npy} .npy files (skipping already-completed).")

        processed = 0
        display_idx = 1
        for npy_path in npy_files:
            result = run_experiment_for_npy(npy_path, display_idx, npy_total)
            display_idx += 1
            if result:
                processed += 1
                if max_npy > 0 and processed >= max_npy:
                    print(f"\n[config] Reached --max_npy_files={max_npy} processed files; stopping.")
                    break

        print("\n" + "█" * 80)
        print(f"█ Processed {processed} datasets this run ({npy_total} total in directory).")
        print("█" * 80)

    elif args.npy_path:
        # Single-file mode
        npy_path = os.path.abspath(args.npy_path)
        if not os.path.isfile(npy_path):
            print(f"[error] File not found: {npy_path}")
            sys.exit(1)
        cfg.NPY_PATH = npy_path
        parent_dir = os.path.dirname(npy_path)
        cfg.DATASET_NPY_DIR = parent_dir
        cfg.DATASET_CATEGORY = os.path.basename(parent_dir)
        run_experiment_for_npy(npy_path, 1, 1)

        print("\n" + "█" * 80)
        print("█ Single-file experiment completed.")
        print("█" * 80)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if any(arg.split("=", 1)[0] in {"--npy_path", "--npy_dir"} for arg in argv):
        run_explicit(argv)
    else:
        run_selected(argv)


if __name__ == "__main__":
    main()
