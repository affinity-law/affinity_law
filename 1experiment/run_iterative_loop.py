import csv
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime

import numpy as np

import project_config as cfg
from candidate_utils import count_model_params, count_model_flops
from compute_metrics import compute_metrics_for_npy

cfg.apply_proxy_settings()

# ── Module-level counter for MAX_NPY_FILES ──
_npy_processed_count = 0


def _iter_name(iter_idx):
    template = getattr(cfg, "ITERATION_DIR_TEMPLATE", "iter_{:03d}")
    try:
        return template.format(iter_idx)
    except Exception:
        return f"iter_{iter_idx:03d}"


def _cfg_name(name, fallback):
    return str(getattr(cfg, name, fallback))


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dump_run_snapshot(result_dir):
    """Persist the effective configuration and software environment to the result dir.

    Reproducibility requirement (Nature): records every config value actually in
    effect for this run, plus package versions and hardware, so the run can be
    reconstructed and audited.
    """
    import platform as _platform
    import types as _types

    os.makedirs(result_dir, exist_ok=True)

    # ── config snapshot: every public, JSON-serializable cfg attribute ──
    config_snapshot = {}
    for name in sorted(dir(cfg)):
        if name.startswith("_"):
            continue
        value = getattr(cfg, name)
        if callable(value) or isinstance(value, _types.ModuleType):
            continue
        # Credentials are runtime inputs, not reproducibility artifacts.
        if name.endswith(("API_KEY", "SECRET", "PASSWORD", "TOKEN", "PROXY")):
            config_snapshot[name] = "[REDACTED]" if value else ""
            continue
        try:
            json.dumps(value)
            config_snapshot[name] = value
        except (TypeError, ValueError):
            config_snapshot[name] = repr(value)
    save_json(os.path.join(result_dir, "config_snapshot.json"), config_snapshot)

    # ── environment snapshot ──
    env_lines = [
        f"python={_platform.python_version()}",
        f"platform={_platform.platform()}",
        f"machine={_platform.machine()}",
    ]
    try:
        import torch as _torch
        env_lines.append(f"torch={_torch.__version__}")
        env_lines.append(f"cuda_available={_torch.cuda.is_available()}")
        if _torch.cuda.is_available():
            env_lines.append(f"cuda_device={_torch.cuda.get_device_name(0)}")
            env_lines.append(f"cuda_version={_torch.version.cuda}")
    except Exception as _e:  # pragma: no cover
        env_lines.append(f"torch=<import failed: {_e}>")
    for pkg in ("numpy", "pandas", "langchain_core", "langchain_openai",
                "statsmodels", "fvcore", "matplotlib"):
        try:
            mod = __import__(pkg)
            env_lines.append(f"{pkg}={getattr(mod, '__version__', '?')}")
        except Exception:
            env_lines.append(f"{pkg}=<not installed>")
    env_lines.append(f"timestamp={datetime.now().isoformat(timespec='seconds')}")

    with open(os.path.join(result_dir, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(env_lines) + "\n")
    print(f"[snapshot] config_snapshot.json + environment.txt saved to {result_dir}")


class StepError(RuntimeError):
    """Raised when a step script fails (drop-in for subprocess.CalledProcessError)."""
    def __init__(self, script_name, returncode, stderr_text=""):
        self.script_name = script_name
        self.returncode = returncode
        self._stderr_text = stderr_text
        super().__init__(f"{script_name} exited with code {returncode}")


def run_step(script_name, extra_args=None, cfg_overrides=None):
    """
    Run a script in-process with project_config path overrides applied temporarily.

    Uses exec() with a properly set up __main__ module instead of a subprocess,
    which avoids the PyInstaller bootloader issue and has no external dependencies.

    cfg attributes are saved before and restored after, providing isolation
    between iterations without the overhead and fragility of a subprocess.
    """
    extra_args = extra_args or []
    cfg_overrides = cfg_overrides or {}

    # Save state that the script may mutate
    saved_cfg = {}
    for key in cfg_overrides:
        if hasattr(cfg, key):
            saved_cfg[key] = getattr(cfg, key)

    # Also save keys commonly mutated by steps (RESULT_DIR, EXPERIMENT_ITERATION, etc.)
    for key in ("RESULT_DIR", "EXPERIMENT_ITERATION", "EXPERIMENT_ITER_DIR",
                "RUNTIME_DIR", "CANDIDATE_MODEL_PATH", "SELECTION_REASON_SAVE_PATH",
                "LATEST_TEST_METRICS_PATH", "LATEST_TRAIN_HISTORY_PATH",
                "LATEST_BEST_MODEL_PATH", "LATEST_TRAIN_STATS_PATH",
                "LATEST_FAILURE_FEEDBACK_PATH", "LATEST_MODEL_CODE_PATH"):
        if hasattr(cfg, key) and key not in saved_cfg:
            saved_cfg[key] = getattr(cfg, key)

    saved_argv = sys.argv[:]
    saved_environ = {
        "EXPERIMENT_RUN_ID": os.environ.get("EXPERIMENT_RUN_ID"),
        "EXPERIMENT_ITERATION": os.environ.get("EXPERIMENT_ITERATION"),
        "EXPERIMENT_ITER_DIR": os.environ.get("EXPERIMENT_ITER_DIR"),
    }

    # Apply overrides
    for key, value in cfg_overrides.items():
        setattr(cfg, key, value)

    if hasattr(cfg, "apply_proxy_settings"):
        cfg.apply_proxy_settings()

    # Set env vars that scripts may read
    if "EXPERIMENT_RUN_ID" in cfg_overrides:
        os.environ["EXPERIMENT_RUN_ID"] = str(cfg_overrides["EXPERIMENT_RUN_ID"])
    if "EXPERIMENT_ITERATION" in cfg_overrides:
        os.environ["EXPERIMENT_ITERATION"] = str(cfg_overrides["EXPERIMENT_ITERATION"])
    if "EXPERIMENT_ITER_DIR" in cfg_overrides:
        os.environ["EXPERIMENT_ITER_DIR"] = str(cfg_overrides["EXPERIMENT_ITER_DIR"])

    # Resolve script path: frozen (PyInstaller) vs development
    if getattr(sys, 'frozen', False):
        # PyInstaller --onedir bundles data files to sys._MEIPASS
        script_path = os.path.join(sys._MEIPASS, script_name)
        if not os.path.exists(script_path):
            # Fallback: maybe it's in CWD or an absolute path
            script_path = script_name
    else:
        # Development: scripts live alongside this module
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
        if not os.path.exists(script_path):
            script_path = script_name  # fallback: CWD or absolute

    sys.argv = [script_path] + extra_args

    print("\n" + "=" * 80)
    print("Starting:", script_name, " ".join(extra_args))
    print("=" * 80)

    try:
        # Inline implementation of runpy.run_path to avoid depending on runpy
        # (which may not be bundled by PyInstaller). Sets up __main__-like
        # environment and executes the target script in-process.
        _script_dir = os.path.dirname(os.path.abspath(script_path))
        _old_path0 = sys.path[0]
        _old_main = sys.modules.get("__main__")
        try:
            sys.path[0] = _script_dir
            _main_mod = type(sys)("__main__")
            _main_mod.__dict__.setdefault("__builtins__", __builtins__)
            _main_mod.__file__ = script_path
            sys.modules["__main__"] = _main_mod
            with open(script_path, "rb") as _f:
                _code = compile(_f.read(), script_path, "exec")
            exec(_code, _main_mod.__dict__)
        finally:
            sys.path[0] = _old_path0
            if _old_main is not None:
                sys.modules["__main__"] = _old_main
            elif "__main__" in sys.modules:
                del sys.modules["__main__"]
    except SystemExit as e:
        code = e.code if e.code is not None else 0
        if code != 0:
            print(f"\nScript failed: {script_name}")
            print(f"Exit code: {code}")
            if script_name == "LLM_with_RAG_iterative.py":
                print("Please check that LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL in project_config.py are correct.")
            raise StepError(script_name, code) from e
    except Exception as e:
        print(f"\nScript failed: {script_name}")
        traceback.print_exc()
        raise StepError(script_name, 1) from e
    finally:
        # Restore state
        for key, value in saved_cfg.items():
            setattr(cfg, key, value)
        sys.argv = saved_argv
        for k, v in saved_environ.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def reset_legacy_latest_feedback():
    """Legacy compatibility: clean up old runtime/latest_* files. The new pipeline no longer creates or depends on these duplicates."""
    for path in [
        getattr(cfg, "LATEST_TEST_METRICS_PATH", ""),
        getattr(cfg, "LATEST_TRAIN_HISTORY_PATH", ""),
        getattr(cfg, "LATEST_BEST_MODEL_PATH", ""),
        getattr(cfg, "LATEST_TRAIN_STATS_PATH", ""),
    ]:
        if path and os.path.exists(path):
            os.remove(path)


def make_iteration_dirs(iter_idx):
    iter_dir = os.path.join(cfg.RESULT_DIR, _iter_name(iter_idx))
    paths = {
        "iter_dir": iter_dir,
        "llm_input_dir": os.path.join(iter_dir, _cfg_name("LLM_INPUT_DIR_NAME", "llm_input")),
        "llm_output_dir": os.path.join(iter_dir, _cfg_name("LLM_OUTPUT_DIR_NAME", "llm_output")),
        "feedback_dir": os.path.join(iter_dir, _cfg_name("FEEDBACK_DIR_NAME", "feedback")),
        "train_dir": os.path.join(iter_dir, _cfg_name("TRAIN_DIR_NAME", "train")),
        "test_dir": os.path.join(iter_dir, _cfg_name("TEST_DIR_NAME", "test")),
    }

    if os.path.exists(iter_dir) and cfg.OVERWRITE_EXISTING_ITER_DIR:
        shutil.rmtree(iter_dir)
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def _prev_iter_model_paths(iter_idx, offset):
    """Return model code and module selection rationale paths for the iteration offset rounds back; return placeholder paths if they don't exist."""
    target = iter_idx - offset
    if target < 1:
        return {
            f"PREV{offset}_MODEL_CODE_PATH": os.path.join(cfg.RESULT_DIR, "_no_previous", cfg.GENERATED_MODEL_FILE),
            f"PREV{offset}_MODULE_REASON_PATH": os.path.join(cfg.RESULT_DIR, "_no_previous", cfg.MODULE_REASON_FILE),
        }
    prev_iter_dir = os.path.join(cfg.RESULT_DIR, _iter_name(target))
    prev_llm_out = os.path.join(prev_iter_dir, _cfg_name("LLM_OUTPUT_DIR_NAME", "llm_output"))
    return {
        f"PREV{offset}_MODEL_CODE_PATH": os.path.join(prev_llm_out, cfg.GENERATED_MODEL_FILE),
        f"PREV{offset}_MODULE_REASON_PATH": os.path.join(prev_llm_out, cfg.MODULE_REASON_FILE),
    }


def previous_feedback_paths(iter_idx):
    """Return the actual artifact paths from the previous two rounds, without copying to latest_feedback_*."""
    if iter_idx <= 1:
        return {
            "LATEST_TEST_METRICS_PATH": os.path.join(cfg.RESULT_DIR, "_no_previous", cfg.TEST_METRICS_FILE),
            "LATEST_TRAIN_HISTORY_PATH": os.path.join(cfg.RESULT_DIR, "_no_previous", cfg.TRAIN_HISTORY_FILE),
            "LATEST_BEST_MODEL_PATH": os.path.join(cfg.RESULT_DIR, "_no_previous", cfg.BEST_MODEL_FILE),
            "LATEST_TRAIN_STATS_PATH": os.path.join(cfg.RESULT_DIR, "_no_previous", cfg.TRAIN_STATS_FILE),
            "LATEST_FAILURE_FEEDBACK_PATH": os.path.join(cfg.RESULT_DIR, "_no_previous", "candidate_failure_feedback.json"),
            **_prev_iter_model_paths(iter_idx, 1),
            **_prev_iter_model_paths(iter_idx, 2),
        }

    prev_iter_dir = os.path.join(cfg.RESULT_DIR, _iter_name(iter_idx - 1))
    prev_train_dir = os.path.join(prev_iter_dir, _cfg_name("TRAIN_DIR_NAME", "train"))
    prev_test_dir = os.path.join(prev_iter_dir, _cfg_name("TEST_DIR_NAME", "test"))
    prev_feedback_dir = os.path.join(prev_iter_dir, _cfg_name("FEEDBACK_DIR_NAME", "feedback"))
    return {
        "LATEST_TEST_METRICS_PATH": os.path.join(prev_test_dir, cfg.TEST_METRICS_FILE),
        "LATEST_TRAIN_HISTORY_PATH": os.path.join(prev_train_dir, cfg.TRAIN_HISTORY_FILE),
        "LATEST_BEST_MODEL_PATH": os.path.join(prev_train_dir, cfg.BEST_MODEL_FILE),
        "LATEST_TRAIN_STATS_PATH": os.path.join(prev_train_dir, cfg.TRAIN_STATS_FILE),
        "LATEST_FAILURE_FEEDBACK_PATH": os.path.join(prev_feedback_dir, "candidate_failure_feedback.json"),
        **_prev_iter_model_paths(iter_idx, 1),
        **_prev_iter_model_paths(iter_idx, 2),
    }


def make_iter_cfg_overrides(iter_idx, paths):
    """Route all sub-script input/output paths to this round's iteration directory."""
    feedback_paths = previous_feedback_paths(iter_idx)
    train_dir = paths["train_dir"]
    test_dir = paths["test_dir"]
    return {
        "EXPERIMENT_ITERATION": iter_idx,
        "EXPERIMENT_ITER_DIR": paths["iter_dir"],
        "RESULT_DIR": cfg.RESULT_DIR,
        "RUNTIME_DIR": paths["llm_input_dir"],
        **feedback_paths,
        "CANDIDATE_MODEL_PATH": os.path.join(paths["llm_output_dir"], cfg.GENERATED_MODEL_FILE),
        "SELECTION_REASON_SAVE_PATH": os.path.join(paths["llm_output_dir"], cfg.MODULE_REASON_FILE),
        "TRAIN_OUTPUT_DIR": train_dir,
        "TEST_OUTPUT_DIR": test_dir,
        "MODEL_WEIGHTS_PATH": os.path.join(train_dir, cfg.BEST_MODEL_FILE),
        "TRAIN_STATS_NPZ_PATH": os.path.join(train_dir, cfg.TRAIN_STATS_FILE),
    }


def save_llm_input_manifest(iter_idx, paths, cfg_overrides):
    """
    Save the LLM input source manifest for this round, recording paths and existence only (no large file copies).
    """
    manifest = {
        "iteration": iter_idx,
        "source_iteration": iter_idx - 1 if iter_idx > 1 else None,
        "note": "First round has no previous feedback. From round 2 onward, the LLM reads previous train/test artifact paths directly, avoiding latest_feedback_* copies.",
        "feedback_sources": {
            "test_metrics": cfg_overrides["LATEST_TEST_METRICS_PATH"],
            "train_history": cfg_overrides["LATEST_TRAIN_HISTORY_PATH"],
            "best_model": cfg_overrides["LATEST_BEST_MODEL_PATH"],
            "train_stats": cfg_overrides["LATEST_TRAIN_STATS_PATH"],
        },
    }
    for key, path in manifest["feedback_sources"].items():
        manifest["feedback_sources"][key] = {"path": path, "exists": os.path.exists(path)}
    save_json(os.path.join(paths["llm_input_dir"], getattr(cfg, "LLM_INPUT_MANIFEST_FILE", "llm_input_manifest.json")), manifest)
    return manifest


def archive_current_feedback(paths, metrics, stop_result):
    """
    Save the current round's feedback summary.

    Large files from train/test are NOT copied to the feedback directory; feedback only saves
    the summary and path index. Actual artifacts remain saved independently in train/test.
    """
    feedback_dir = paths["feedback_dir"]
    train_dir = paths["train_dir"]
    test_dir = paths["test_dir"]

    files = {
        "test_metrics_file": os.path.join(test_dir, cfg.TEST_METRICS_FILE),
        "training_loss_file": os.path.join(train_dir, cfg.TRAIN_HISTORY_FILE),
        "best_model_weights_file": os.path.join(train_dir, cfg.BEST_MODEL_FILE),
        "train_standardization_stats_file": os.path.join(train_dir, cfg.TRAIN_STATS_FILE),
    }
    summary = {
        **files,
        "file_exists": {name: os.path.exists(path) for name, path in files.items()},
        "overall_metrics": metrics.get("overall_metrics", {}) if isinstance(metrics, dict) else {},
        "residual_diagnostics": metrics.get("residual_diagnostics", {}) if isinstance(metrics, dict) else {},
        "stop_judgement_file": os.path.join(feedback_dir, getattr(cfg, "STOP_JUDGEMENT_FILE", "stop_judgement.json")),
    }
    save_json(summary["stop_judgement_file"], stop_result)
    save_json(os.path.join(feedback_dir, getattr(cfg, "FEEDBACK_SUMMARY_FILE", "feedback_summary.json")), summary)
    return summary


def safe_float(value):
    try:
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    except (TypeError, ValueError):
        return None


def read_meta_features_dict(csv_path):
    """Read meta_features.csv, returning {indicator_name: value} dict."""
    if not os.path.exists(csv_path):
        print(f"[topk] Warning: meta_features.csv not found: {csv_path}")
        return {}
    result = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["indicator_name"]] = row.get("value", "")
    return result


def parse_module_names_from_md(md_path):
    """Parse module names from the '## Selected Modules' section of module_selection_reason.md."""
    if not os.path.exists(md_path):
        print(f"[topk] Warning: module_selection_reason.md not found: {md_path}")
        return ""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    section_match = re.search(
        r'##\s+Selected Modules\s*\n(.*?)(?=\n##\s|\Z)',
        content,
        re.DOTALL,
    )
    if not section_match:
        return ""

    section_text = section_match.group(1)
    names = re.findall(r'\d+\.\s+\*\*(.+?)\*\*', section_text)
    return ", ".join(names)


def parse_combination_strategy_from_md(md_path):
    """Extract text from the '## Combination Strategy' section of module_selection_reason.md, for matching E-category (combination paradigm) names."""
    if not os.path.exists(md_path):
        return ""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r'##\s+Combination Strategy\s*\n(.*?)(?=\n##\s|\Z)', content, re.DOTALL)
    return m.group(1).strip() if m else ""


# Module name -> A/B/C/D/E classification, from module_library_candidates.json
_MODULE_CATEGORY_MAP = {}
for _name in ["Identity", "Z-score", "RevIN", "Differencing", "Series Decomposition", "Patch", "Mask"]:
    _MODULE_CATEGORY_MAP[_name] = "A"
for _name in ["Linear Block", "MLP Block", "CNN Block", "TCN Block", "GRU Block", "LSTM Block", "State Space Model Block", "Transformer Encoder Block", "Basis Block"]:
    _MODULE_CATEGORY_MAP[_name] = "B"
for _name in ["Linear Head", "MLP Head", "Basis Projection"]:
    _MODULE_CATEGORY_MAP[_name] = "C"
for _name in ["Inverse Z-score", "Inverse RevIN", "Inverse Differencing", "Decomposition Recomposition"]:
    _MODULE_CATEGORY_MAP[_name] = "D"
for _name in ["Single Block", "Sequential Composition", "Parallel Additive Branches", "Residual Modeling", "Gated Fusion"]:
    _MODULE_CATEGORY_MAP[_name] = "E"


def _normalize_module_name(name):
    """Normalize non-canonical LLM-generated names (e.g., 'MLPBlock') to canonical names."""
    name = name.strip()
    if name in _MODULE_CATEGORY_MAP:
        return name
    # Try case-insensitive exact match first
    name_lower = name.lower()
    for canonical in _MODULE_CATEGORY_MAP:
        if canonical.lower() == name_lower:
            return canonical
    # Try collapsing whitespace (case-insensitive)
    collapsed = "".join(name.split()).lower()
    for canonical in _MODULE_CATEGORY_MAP:
        if "".join(canonical.split()).lower() == collapsed:
            return canonical
    return name


def classify_modules_from_md(md_path):
    """Parse module names from module_selection_reason.md and classify into the five A/B/C/D/E stages.

    E-category (combination paradigm) names are matched from both the 'Selected Modules'
    section and the 'Combination Strategy' section (scanning for keywords like
    Sequential Composition, Residual Modeling, etc.).

    Returns:
        dict with keys 'A','B','C','D','E', each value a comma-separated module name string.
    """
    empty = {"A": "", "B": "", "C": "", "D": "", "E": ""}
    names_str = parse_module_names_from_md(md_path)
    combination_text = parse_combination_strategy_from_md(md_path)

    # First, classify names matched from the 'Selected Modules' section
    if names_str:
        names = [n.strip() for n in names_str.split(",")]
    else:
        names = []

    categories = {"A": [], "B": [], "C": [], "D": [], "E": []}
    for name in names:
        canonical = _normalize_module_name(name)
        cat = _MODULE_CATEGORY_MAP.get(canonical, "")
        if cat:
            categories[cat].append(canonical)

    # Supplement: match E-category keywords from the 'Combination Strategy' section
    if combination_text:
        for canonical, cat in _MODULE_CATEGORY_MAP.items():
            if cat == "E":
                if canonical.lower() in combination_text.lower():
                    if canonical not in categories["E"]:
                        categories["E"].append(canonical)
            elif cat == "D":
                # D-category (Output Inverse) modules are also often mentioned only in
                # the Combination Strategy text (e.g., "AutoD(Inverse RevIN)")
                # rather than as numbered items in Selected Modules.
                if canonical.lower() in combination_text.lower():
                    if canonical not in categories["D"]:
                        categories["D"].append(canonical)

    # Auto-infer D-category from A-category:
    # If a preprocessing module is used, the corresponding inverse module is
    # functionally required. The LLM often omits it from Selected Modules.
    _A_TO_D = {
        "Z-score": "Inverse Z-score",
        "RevIN": "Inverse RevIN",
        "Differencing": "Inverse Differencing",
        "Series Decomposition": "Decomposition Recomposition",
    }
    for a_mod in categories["A"]:
        if a_mod in _A_TO_D:
            d_mod = _A_TO_D[a_mod]
            if d_mod not in categories["D"]:
                categories["D"].append(d_mod)

    # Return "none" for categories that have no selected modules (instead of "")
    return {k: ", ".join(v) if v else "none" for k, v in categories.items()}


def _arch_signature(cats):
    """Build an order-invariant architecture signature from the A/B/C/D/E module sets.

    Modules within each category are sorted, so candidates that list the same
    modules in a different order collapse to the same signature (which is what we
    want when deduping near-duplicate architectures).
    """
    if not cats:
        return ()
    parts = []
    for c in ("A", "B", "C", "D", "E"):
        val = cats.get(c, "")
        if not val or val == "none":
            parts.append((c, ()))
        else:
            mods = tuple(sorted(m.strip() for m in val.split(",") if m.strip()))
            parts.append((c, mods))
    return tuple(parts)


def record_topk_candidates(npy_path, history):
    """After each iteration, update the dataset's Top-K CSV: result/<dataset>/<run_timestamp>/topk_candidates.csv."""
    top_k = int(getattr(cfg, "TOP_K", 3))
    sort_metric = str(getattr(cfg, "TOP_K_SORT_METRIC", "rmse"))
    ascending = bool(getattr(cfg, "TOP_K_SORT_ASCENDING", True))

    npy_basename = os.path.splitext(os.path.basename(npy_path))[0]
    meta_features_path = os.path.join(cfg.BASE_RESULT_DIR, "meta_features.csv")
    meta_dict = read_meta_features_dict(meta_features_path)

    metric_key_map = {
        # Validation-driven selection (recommended): the sort metric is computed
        # on the validation set, keeping the test set free of selection bias.
        "val_rmse": "val_rmse",
        "val_mae": "val_mae",
        "val_loss": "val_loss",
        # Backward-compatible aliases also map to validation metrics.
        "rmse": "val_rmse",
        "mae": "val_mae",
    }
    sort_key = metric_key_map.get(sort_metric, "val_rmse")

    successful = [
        item for item in history
        if item.get(sort_key) is not None
    ]

    meta_order = getattr(cfg, "META_FEATURE_NAMES_ORDERED", [])
    csv_path = os.path.join(cfg.RESULT_DIR, getattr(cfg, "TOP_K_CSV_FILENAME", "topk_candidates.csv"))

    fieldnames = (
        ["Dataset", "Rank", "TIME_LAG", "PRE_LEN"]
        + meta_order
        + ["val_RMSE", "val_MAE", "val_loss", "RMSE", "MAE", "WMAPE", "Params", "FLOPs"]
        + ["A_Input_Preprocessing", "B_Backbone_Modeling", "C_Head_Projection", "D_Output_Inverse", "E_Combination_Paradigm", "Candidate_Model_Dir"]
    )

    time_lag = int(getattr(cfg, "TIME_LAG", 0))
    pre_len = int(getattr(cfg, "PRE_LEN", 0))

    new_rows = []
    if successful:
        for item in successful:
            item["_module_categories"] = classify_modules_from_md(
                item.get("module_reason_file", "")
            )

        # Dedup by architecture signature: keep only the best candidate (by the
        # sort metric) per A/B/C/D/E signature, so Top-K is diverse rather than
        # dominated by near-duplicate variants of one architecture.
        if bool(getattr(cfg, "TOP_K_DEDUP_BY_ARCH", True)):
            def _metric_v(item):
                v = safe_float(item.get(sort_key))
                return v if v is not None else float("inf")

            best_per_sig = {}
            for item in successful:
                sig = _arch_signature(item.get("_module_categories", {}))
                prev = best_per_sig.get(sig)
                if prev is None or _metric_v(item) < _metric_v(prev):
                    best_per_sig[sig] = item
            deduped_count = len(successful) - len(best_per_sig)
            successful = list(best_per_sig.values())
            if deduped_count:
                print(f"[topk] Dedup by architecture signature: dropped {deduped_count} near-duplicate candidate(s).")

        def sort_key_fn(item):
            v = safe_float(item.get(sort_key))
            if v is None:
                return (1, 0.0)
            return (0, v)

        successful.sort(key=sort_key_fn, reverse=not ascending)

        for rank, item in enumerate(successful[:top_k], start=1):
            row = {"Dataset": npy_basename, "Rank": rank, "TIME_LAG": time_lag, "PRE_LEN": pre_len}
            for name in meta_order:
                row[name] = meta_dict.get(name, "")
            row["val_RMSE"] = safe_float(item.get("val_rmse"))
            row["val_MAE"] = safe_float(item.get("val_mae"))
            row["val_loss"] = safe_float(item.get("val_loss"))
            row["RMSE"] = safe_float(item.get("overall_rmse"))
            row["MAE"] = safe_float(item.get("overall_mae"))
            row["WMAPE"] = safe_float(item.get("overall_wmape"))
            row["Params"] = item.get("total_params")
            row["FLOPs"] = item.get("total_flops")
            cats = item.get("_module_categories", {})
            row["A_Input_Preprocessing"] = cats.get("A", "none")
            row["B_Backbone_Modeling"] = cats.get("B", "none")
            row["C_Head_Projection"] = cats.get("C", "none")
            row["D_Output_Inverse"] = cats.get("D", "none")
            row["E_Combination_Paradigm"] = cats.get("E", "none")
            row["Candidate_Model_Dir"] = item.get("iteration_folder", "")
            new_rows.append(row)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    if new_rows:
        print(f"[topk] Dataset {npy_basename}: Top-{top_k} saved to {csv_path}")
        for i, item in enumerate(successful[:top_k], 1):
            print(f"  #{i} iter={item.get('iteration')} {sort_metric}={safe_float(item.get(sort_key))}")


def record_all_candidates_csv(npy_path, history):
    """Record ALL generated candidate models (not just top-K) with the same CSV format."""
    csv_path = os.path.join(cfg.RESULT_DIR, getattr(cfg, "ALL_CANDIDATES_CSV_FILENAME", "all_candidates.csv"))
    npy_basename = os.path.splitext(os.path.basename(npy_path))[0]
    meta_features_path = os.path.join(cfg.BASE_RESULT_DIR, "meta_features.csv")
    meta_dict = read_meta_features_dict(meta_features_path)
    meta_order = getattr(cfg, "META_FEATURE_NAMES_ORDERED", [])
    time_lag = int(getattr(cfg, "TIME_LAG", 0))
    pre_len = int(getattr(cfg, "PRE_LEN", 0))

    fieldnames = (
        ["Dataset", "Iteration", "TIME_LAG", "PRE_LEN"]
        + meta_order
        + ["val_RMSE", "val_MAE", "val_loss", "RMSE", "MAE", "WMAPE", "Params", "FLOPs"]
        + ["A_Input_Preprocessing", "B_Backbone_Modeling", "C_Head_Projection", "D_Output_Inverse", "E_Combination_Paradigm", "Candidate_Model_Dir"]
    )

    rows = []
    for item in history:
        cats = classify_modules_from_md(item.get("module_reason_file", ""))
        row = {"Dataset": npy_basename, "Iteration": item.get("iteration", "?"), "TIME_LAG": time_lag, "PRE_LEN": pre_len}
        for name in meta_order:
            row[name] = meta_dict.get(name, "")
        row["val_RMSE"] = safe_float(item.get("val_rmse"))
        row["val_MAE"] = safe_float(item.get("val_mae"))
        row["val_loss"] = safe_float(item.get("val_loss"))
        row["RMSE"] = safe_float(item.get("overall_rmse"))
        row["MAE"] = safe_float(item.get("overall_mae"))
        row["WMAPE"] = safe_float(item.get("overall_wmape"))
        row["Params"] = item.get("total_params")
        row["FLOPs"] = item.get("total_flops")
        row["A_Input_Preprocessing"] = cats.get("A", "none")
        row["B_Backbone_Modeling"] = cats.get("B", "none")
        row["C_Head_Projection"] = cats.get("C", "none")
        row["D_Output_Inverse"] = cats.get("D", "none")
        row["E_Combination_Paradigm"] = cats.get("E", "none")
        row["Candidate_Model_Dir"] = item.get("iteration_folder", "")
        rows.append(row)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[all] All {len(rows)} candidates saved to {csv_path}")


def extract_stop_metrics(metrics):
    metrics = metrics if isinstance(metrics, dict) else {}
    residual = metrics.get("residual_diagnostics", {})
    overall = metrics.get("overall_metrics", {})
    val = metrics.get("val_metrics", {})
    return {
        "overall_mae": safe_float(overall.get("mae")),
        "overall_rmse": safe_float(overall.get("rmse")),
        "overall_wmape": safe_float(overall.get("wmape")),
        "overall_smape": safe_float(overall.get("smape")),
        "bias_abs": safe_float(abs(residual["bias_mean_error"])) if residual.get("bias_mean_error") is not None else None,
        "residual_std": safe_float(residual.get("residual_std")),
        "lag1_autocorr_abs": safe_float(abs(residual["lag1_autocorr"])) if residual.get("lag1_autocorr") is not None else None,
        "residual_target_corr_abs": safe_float(abs(residual["residual_target_corr"])) if residual.get("residual_target_corr") is not None else None,
        # Validation metrics — the model-selection signal (stop / ranking / stability).
        # Test metrics (overall_*) are kept for reporting only, never for selection.
        "val_rmse": safe_float(val.get("val_rmse")),
        "val_mae": safe_float(val.get("val_mae")),
        "val_loss": safe_float(val.get("val_loss")),
    }


def extract_best_val_metrics(train_dir):
    """Read training_loss.csv and return the validation metrics at the best epoch
    (lowest val_loss — the checkpoint actually saved as best).

    This is the selection signal: the search ranks/stops on validation performance,
    not on the test set, to avoid test-set selection bias.
    """
    csv_path = os.path.join(train_dir, cfg.TRAIN_HISTORY_FILE)
    if not os.path.exists(csv_path):
        return {}
    best_row = None
    best_val = float("inf")
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = safe_float(row.get("val_loss"))
            if v is None:
                continue
            if v < best_val:  # first occurrence of the min == the saved best checkpoint
                best_val = v
                best_row = row
    if best_row is None:
        return {}
    epoch_raw = str(best_row.get("epoch", "")).strip()
    return {
        "val_loss": safe_float(best_row.get("val_loss")),
        "val_mae": safe_float(best_row.get("val_mae")),
        "val_rmse": safe_float(best_row.get("val_rmse")),
        "best_epoch": int(epoch_raw) if epoch_raw.lstrip("-").isdigit() else None,
    }


def compute_composite_score(stop_metrics, reference_metrics):
    """Normalize stop_metrics by reference_metrics; composite_score < 1 means better than reference."""
    total_weight = weighted_score = 0.0
    details = {}
    for name, weight in cfg.COMPOSITE_SCORE_WEIGHTS.items():
        value = safe_float(stop_metrics.get(name))
        reference = safe_float(reference_metrics.get(name))
        if value is None or reference is None or weight <= 0:
            continue
        normalized_value = value / max(abs(reference), 1e-12)
        weighted_score += weight * normalized_value
        total_weight += weight
        details[name] = {
            "raw_value": value,
            "reference_value": reference,
            "normalized_value": normalized_value,
            "weight": weight,
        }
    return (weighted_score / total_weight, details) if total_weight > 0 else (None, details)


def update_stop_state(iter_idx, metrics, state):
    """Early-stopping based on a SINGLE primary metric (scientific standard).

    The composite score (COMPOSITE_SCORE_WEIGHTS) is computed and logged for
    reference / analysis, but does NOT gate the stop decision — the search
    continues as long as the primary metric is improving.
    """
    stop_metrics = extract_stop_metrics(metrics)
    primary_metric = getattr(cfg, "STOP_PRIMARY_METRIC", "overall_mae")
    primary_value = safe_float(stop_metrics.get(primary_metric))
    min_rel_improvement = float(getattr(cfg, "STOP_PRIMARY_MIN_REL_IMPROVEMENT", 1e-4))

    # Composite score for reporting only (does not affect stop decision)
    if state["best_metrics"] is None:
        composite_score, score_details = 1.0, {}
    else:
        composite_score, score_details = compute_composite_score(stop_metrics, state["best_metrics"])

    # ── First round: use self as initial best ──
    if state["best_metrics"] is None:
        state["best_metrics"] = stop_metrics
        state["best_iteration"] = iter_idx
        state["best_primary_value"] = primary_value
        state["no_improve_rounds"] = 0
        return {
            "stop_metrics": stop_metrics,
            "composite_score": 1.0,
            "score_details": {},
            "primary_metric": primary_metric,
            "primary_value": primary_value,
            "best_iteration": iter_idx,
            "best_primary_value": primary_value,
            "no_improve_rounds": 0,
            "should_stop": False,
            "stop_reason": f"First round, baseline {primary_metric}={primary_value:.6g}; continue iterating.",
        }

    # ── Compare current round's primary metric against best so far ──
    best_primary = state.get("best_primary_value")
    improved = False

    if primary_value is not None and best_primary is not None:
        relative_improvement = (best_primary - primary_value) / max(abs(best_primary), 1e-12)
        improved = relative_improvement >= min_rel_improvement
    else:
        relative_improvement = None
        improved = False

    if improved:
        state["best_metrics"] = stop_metrics
        state["best_iteration"] = iter_idx
        state["best_primary_value"] = primary_value
        state["no_improve_rounds"] = 0
        result = {
            "stop_metrics": stop_metrics,
            "composite_score": composite_score,
            "score_details": score_details,
            "primary_metric": primary_metric,
            "primary_value": primary_value,
            "relative_improvement_vs_best": relative_improvement,
            "best_iteration": iter_idx,
            "best_primary_value": primary_value,
            "no_improve_rounds": 0,
            "should_stop": False,
            "stop_reason": (
                f"{primary_metric} improved by {relative_improvement:.6g} "
                f"({best_primary:.6g} → {primary_value:.6g}); updating best and continuing."
            ),
        }
        return result

    # ── No improvement this round ──
    state["no_improve_rounds"] += 1
    rel_str = f"{relative_improvement:.6g}" if relative_improvement is not None else "N/A"
    reason = (
        f"{primary_metric}={primary_value:.6g}, "
        f"{primary_metric} improvement vs best = {rel_str} "
        f"(threshold={min_rel_improvement}); "
        f"no_improve={state['no_improve_rounds']}/{cfg.ITERATION_PATIENCE}."
    )

    if iter_idx < cfg.MIN_ITERATIONS_BEFORE_STOP:
        reason += f" Only {iter_idx} rounds done (min={cfg.MIN_ITERATIONS_BEFORE_STOP}); continue."
    elif state["no_improve_rounds"] >= cfg.ITERATION_PATIENCE:
        reason += " Stopping."
        result = {
            "stop_metrics": stop_metrics,
            "composite_score": composite_score,
            "score_details": score_details,
            "primary_metric": primary_metric,
            "primary_value": primary_value,
            "relative_improvement_vs_best": relative_improvement,
            "best_iteration": state["best_iteration"],
            "best_primary_value": state.get("best_primary_value"),
            "no_improve_rounds": state["no_improve_rounds"],
            "should_stop": True,
            "stop_reason": reason,
        }
        return result

    result = {
        "stop_metrics": stop_metrics,
        "composite_score": composite_score,
        "score_details": score_details,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "relative_improvement_vs_best": relative_improvement,
        "best_iteration": state["best_iteration"],
        "best_primary_value": state.get("best_primary_value"),
        "no_improve_rounds": state["no_improve_rounds"],
        "should_stop": False,
        "stop_reason": reason,
    }
    return result


def build_history_item(iter_idx, paths, metrics, generated_model_path, module_reason_path, prompt_path, stop_result, feedback_summary):
    overall = metrics.get("overall_metrics", {}) if isinstance(metrics, dict) else {}
    val = metrics.get("val_metrics", {}) if isinstance(metrics, dict) else {}
    train_dir = paths["train_dir"]
    test_dir = paths["test_dir"]
    return {
        "iteration": iter_idx,
        "iteration_folder": paths["iter_dir"],
        "llm_input_folder": paths["llm_input_dir"],
        "llm_output_folder": paths["llm_output_dir"],
        "feedback_folder": paths["feedback_dir"],
        "generated_model_file": generated_model_path,
        "module_reason_file": module_reason_path,
        "llm_prompt_file": prompt_path,
        "train_folder": train_dir,
        "best_model_weights_file": os.path.join(train_dir, cfg.BEST_MODEL_FILE),
        "train_standardization_stats_file": os.path.join(train_dir, cfg.TRAIN_STATS_FILE),
        "training_loss_file": os.path.join(train_dir, cfg.TRAIN_HISTORY_FILE),
        "test_folder": test_dir,
        "test_metrics_file": os.path.join(test_dir, cfg.TEST_METRICS_FILE),
        "final_predictions_file": os.path.join(test_dir, cfg.FINAL_PREDS_FILE),
        "final_targets_file": os.path.join(test_dir, cfg.FINAL_TARGETS_FILE),
        "overall_mae": overall.get("mae"),
        "overall_rmse": overall.get("rmse"),
        "overall_wmape": overall.get("wmape"),
        "overall_smape": overall.get("smape"),
        "total_params": overall.get("total_params"),
        "total_flops": overall.get("total_flops"),
        # Validation metrics (selection signal) — see extract_best_val_metrics.
        "val_rmse": safe_float(val.get("val_rmse")),
        "val_mae": safe_float(val.get("val_mae")),
        "val_loss": safe_float(val.get("val_loss")),
        "val_metrics": val,
        "residual_diagnostics": metrics.get("residual_diagnostics", {}) if isinstance(metrics, dict) else {},
        "iteration_stop_judgement": stop_result,
        "feedback_summary": feedback_summary,
    }


def discover_npy_files(data_dir):
    """Scan a directory for all .npy files, return sorted absolute paths."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"npy data directory not found: {data_dir}")
    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".npy")
    )
    if not files:
        raise FileNotFoundError(f"No .npy files found under {data_dir}.")
    print(f"[discover] Found {len(files)} .npy files in {data_dir}:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    return files


def setup_experiment_for_npy(npy_path):
    """
    Prepare experiment for a single .npy file:
    1. Extract basename from .npy path
    2. Create result/<basename>/ directory
    3. Run compute_metrics to generate raw_indicators.csv and meta_features.csv
    4. Return base cfg_overrides (for subsequent subprocess use)
    """
    basename = os.path.splitext(os.path.basename(npy_path))[0]
    experiment_result_dir = os.path.join(cfg.RESULT_ROOT, cfg.DATASET_CATEGORY, basename)

    # Run compute_metrics
    seasonal_period = int(getattr(cfg, "SEASONAL_PERIOD", 0))
    compute_metrics_for_npy(
        npy_path=npy_path,
        output_dir=experiment_result_dir,
        seasonal_period=seasonal_period,
    )

    # Assemble cfg_overrides; subprocesses will inject these into project_config
    run_id = os.getenv("EXPERIMENT_RUN_ID") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    result_dir = os.path.join(experiment_result_dir, run_id)

    return {
        "NPY_PATH": npy_path,
        "RESULT_ROOT": cfg.RESULT_ROOT,
        "BASE_RESULT_DIR": experiment_result_dir,
        "RESULT_DIR": result_dir,
        "RUN_ID": run_id,
        "ORIGINAL_STATS_CSV_PATH": os.path.join(experiment_result_dir, "raw_indicators.csv"),
        "ABSTRACT_STATS_CSV_PATH": os.path.join(experiment_result_dir, "meta_features.csv"),
    }


def run_experiment_for_npy(npy_path, npy_index, npy_total):
    """Run the full LLM+train+test iterative loop for a single .npy file."""
    print("\n" + "█" * 80)
    print(f"█ Dataset [{npy_index}/{npy_total}]: {os.path.basename(npy_path)}")
    print("█" * 80)

    # The entry point has already applied environment defaults and explicit CLI overrides.

    # If any run_* subdirectory already has topk_candidates.csv, skip this dataset
    basename = os.path.splitext(os.path.basename(npy_path))[0]
    experiment_result_dir = os.path.join(cfg.RESULT_ROOT, cfg.DATASET_CATEGORY, basename)
    topk_csv_filename = getattr(cfg, "TOP_K_CSV_FILENAME", "topk_candidates.csv")

    existing_topk = None
    if os.path.isdir(experiment_result_dir):
        for entry in os.listdir(experiment_result_dir):
            candidate = os.path.join(experiment_result_dir, entry, topk_csv_filename)
            if os.path.isfile(candidate):
                existing_topk = candidate
                break
    if existing_topk:
        print(f"[skip] Dataset {basename} already has {existing_topk}; skipping iteration.")
        return False

    # ── MAX_NPY_FILES check: count only ACTUALLY PROCESSED files ──
    global _npy_processed_count
    max_npy = cfg.MAX_NPY_FILES
    if max_npy > 0 and _npy_processed_count >= max_npy:
        print(f"\n[config] MAX_NPY_FILES={max_npy} reached ({_npy_processed_count} files processed). Stopping.")
        sys.exit(0)

    # Step 1: Prepare experiment directory and indicator CSVs
    npy_overrides = setup_experiment_for_npy(npy_path)

    # Write experiment-level paths into the current process's cfg (for path functions in the main loop)
    for key, value in npy_overrides.items():
        setattr(cfg, key, value)

    # After RESULT_DIR is overridden, recompute derived paths that depend on it
    cfg.ITERATION_HISTORY_PATH = os.path.join(cfg.RESULT_DIR, cfg.ITERATION_HISTORY_FILE)

    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    dump_run_snapshot(cfg.RESULT_DIR)
    if cfg.RESET_LATEST_FEEDBACK_ON_START:
        reset_legacy_latest_feedback()

    history = []
    stop_state = {"best_metrics": None, "best_iteration": None, "no_improve_rounds": 0}

    for iter_idx in range(1, cfg.MAX_ITERATIONS + 1):
        print("\n" + "#" * 80)
        print(f"[{os.path.basename(npy_path)}] Round {iter_idx}")
        print("#" * 80)

        paths = make_iteration_dirs(iter_idx)
        cfg_overrides = make_iter_cfg_overrides(iter_idx, paths)
        # Merge experiment-level config (NPY_PATH, RESULT_DIR, CSV paths, etc.)
        cfg_overrides = {**npy_overrides, **cfg_overrides}
        save_llm_input_manifest(iter_idx, paths, cfg_overrides)

        # LLM generation step: tolerate unstable LLM output (e.g. missing
        # candidate_code, JSON parse errors, empty responses) by skipping this
        # round and continuing, instead of crashing the whole run.
        try:
            run_step("LLM_with_RAG_iterative.py", cfg_overrides=cfg_overrides)
        except (StepError, subprocess.CalledProcessError, RuntimeError, ValueError) as err:
            stderr_detail = getattr(err, "_stderr_text", "") or str(err)
            error_type = type(err).__name__
            returncode = getattr(err, "returncode", 1)
            failure_feedback = {
                "iteration": iter_idx,
                "stage": "llm",
                "error_type": error_type,
                "returncode": returncode,
                "message": (
                    "LLM output was unstable this round (e.g. missing candidate_code, "
                    "unparseable JSON, or empty response). Skipping this round; the next "
                    "round will retry generation from the same prompt/feedback."
                ),
                "stderr_tail": stderr_detail[-3000:] if len(stderr_detail) > 3000 else stderr_detail,
            }
            save_json(os.path.join(paths["feedback_dir"], "candidate_failure_feedback.json"), failure_feedback)
            history.append({
                "iteration": iter_idx,
                "iteration_folder": paths["iter_dir"],
                "failed_stage": "llm",
                "failure_feedback": failure_feedback,
            })
            save_json(cfg.ITERATION_HISTORY_PATH, history)
            print(f"[round {iter_idx}] LLM generation failed ({error_type}); skipping round and continuing.")
            continue

        generated_model_path = cfg_overrides["CANDIDATE_MODEL_PATH"]
        module_reason_path = cfg_overrides["SELECTION_REASON_SAVE_PATH"]
        prompt_path = os.path.join(paths["llm_input_dir"], getattr(cfg, "LLM_PROMPT_FILE", "llm_prompt.txt"))

        train_dir = paths["train_dir"]
        test_dir = paths["test_dir"]
        best_model_path = os.path.join(train_dir, cfg.BEST_MODEL_FILE)
        train_stats_path = os.path.join(train_dir, cfg.TRAIN_STATS_FILE)

        try:
            run_step(
                "train_val_early_stop_iterative.py",
                cfg_overrides=cfg_overrides,
            )
        except (StepError, subprocess.CalledProcessError, RuntimeError) as err:
            stderr_detail = getattr(err, "_stderr_text", "")
            error_type = type(err).__name__
            returncode = getattr(err, "returncode", 1)
            failure_feedback = {
                "iteration": iter_idx,
                "stage": "train",
                "error_type": error_type,
                "returncode": returncode,
                "message": (
                    "Candidate model failed before or during training. Next round must prioritize fixing "
                    "forward dimensions, output shape, Linear input dimensions, batch dim, and station_num handling."
                ),
                "stderr_tail": stderr_detail[-3000:] if len(stderr_detail) > 3000 else stderr_detail,
                "candidate_model_file": generated_model_path,
            }
            save_json(os.path.join(paths["feedback_dir"], "candidate_failure_feedback.json"), failure_feedback)
            history.append({
                "iteration": iter_idx,
                "iteration_folder": paths["iter_dir"],
                "generated_model_file": generated_model_path,
                "failed_stage": "train",
                "failure_feedback": failure_feedback,
            })
            save_json(cfg.ITERATION_HISTORY_PATH, history)
            continue

        try:
            run_step(
                "test_metrics_iterative.py",
                cfg_overrides=cfg_overrides,
            )
        except (StepError, subprocess.CalledProcessError, RuntimeError) as err:
            stderr_detail = getattr(err, "_stderr_text", "")
            error_type = type(err).__name__
            returncode = getattr(err, "returncode", 1)
            failure_feedback = {
                "iteration": iter_idx,
                "stage": "test",
                "error_type": error_type,
                "returncode": returncode,
                "message": (
                    "Test phase failed. Possible causes: model forward inconsistent with training "
                    "(e.g., batch-size sensitivity), weight loading shape mismatch, or "
                    "test_metrics_iterative.py internal error."
                ),
                "stderr_tail": stderr_detail[-3000:] if len(stderr_detail) > 3000 else stderr_detail,
                "candidate_model_file": generated_model_path,
                "best_model_file": best_model_path,
            }
            save_json(os.path.join(paths["feedback_dir"], "candidate_failure_feedback.json"), failure_feedback)
            history.append({
                "iteration": iter_idx,
                "iteration_folder": paths["iter_dir"],
                "generated_model_file": generated_model_path,
                "failed_stage": "test",
                "failure_feedback": failure_feedback,
            })
            save_json(cfg.ITERATION_HISTORY_PATH, history)
            continue

        metrics = load_json(os.path.join(test_dir, cfg.TEST_METRICS_FILE), default={})

        # Compute model parameter count and FLOPs
        train_stats_path = os.path.join(paths["train_dir"], cfg.TRAIN_STATS_FILE)
        if os.path.exists(train_stats_path):
            stats = np.load(train_stats_path)
            station_num = int(stats["station_num"])
            time_lag = int(stats["time_lag"])
            pre_len = int(stats["pre_len"])
        else:
            station_num = 1
            time_lag = int(getattr(cfg, "TIME_LAG", 12))
            pre_len = int(getattr(cfg, "PRE_LEN", 3))
        total_params = count_model_params(generated_model_path, time_lag, pre_len, station_num)
        total_flops = count_model_flops(generated_model_path, time_lag, pre_len, station_num)
        metrics.setdefault("overall_metrics", {})
        metrics["overall_metrics"]["total_params"] = total_params
        metrics["overall_metrics"]["total_flops"] = total_flops
        print(f"Model params: {total_params:,}  FLOPs: {total_flops}")

        # Validation metrics drive model selection (stop / ranking / stability);
        # the test set is reserved for final reporting only (avoids test-set
        # selection bias). Persist them into test_metrics.json so downstream LLM
        # feedback and CSV recording can read val + test from one file.
        metrics["val_metrics"] = extract_best_val_metrics(train_dir)
        save_json(os.path.join(test_dir, cfg.TEST_METRICS_FILE), metrics)

        stop_result = update_stop_state(iter_idx, metrics, stop_state)
        feedback_summary = archive_current_feedback(paths, metrics, stop_result)

        history_item = build_history_item(
            iter_idx,
            paths,
            metrics,
            generated_model_path,
            module_reason_path,
            prompt_path,
            stop_result,
            feedback_summary,
        )
        history.append(history_item)
        save_json(cfg.ITERATION_HISTORY_PATH, history)
        record_topk_candidates(npy_path, history)
        record_all_candidates_csv(npy_path, history)

        print("\nRound summary:")
        print(f"Dataset: {os.path.basename(npy_path)}")
        print(f"iteration: {iter_idx}")
        print(f"LLM input dir: {paths['llm_input_dir']}")
        print(f"LLM output dir: {paths['llm_output_dir']}")
        print(f"Feedback dir: {paths['feedback_dir']}")
        print(f"MAE: {history_item.get('overall_mae')}")
        print(f"RMSE: {history_item.get('overall_rmse')}")
        print(f"WMAPE: {history_item.get('overall_wmape')}")
        print(f"SMAPE: {history_item.get('overall_smape')}")
        print(f"[selection] val_rmse: {history_item.get('val_rmse')}  val_mae: {history_item.get('val_mae')}  val_loss: {history_item.get('val_loss')}")
        print(f"Composite score: {stop_result.get('composite_score')}")
        print(f"Stop judgement: {stop_result.get('stop_reason')}")

        if stop_result.get("should_stop"):
            print(stop_result.get("stop_reason", "Stop condition reached; ending iteration."))
            break

    # Phase 2 & 3: Stability re-runs
    # Take Top-K candidate models from iteration_history (ranked by composite score),
    # re-train/test each with different seeds, compute mean/std metrics,
    # and re-rank by comprehensive composite score.
    _run_stability_phase(npy_path, npy_overrides)

    # ── Increment processed counter; exit if limit reached ──
    _npy_processed_count += 1
    max_npy = cfg.MAX_NPY_FILES
    if max_npy > 0:
        print(f"[config] Files processed this run: {_npy_processed_count}/{max_npy}")
        if _npy_processed_count >= max_npy:
            print(f"[config] MAX_NPY_FILES={max_npy} reached. Stopping.")
            sys.exit(0)

    return True


def _run_stability_phase(npy_path, npy_overrides):
    """Run Phase 2 stability re-runs and Phase 3 ranking for the current dataset."""
    stability_top_k = int(getattr(cfg, "STABILITY_TOP_K", 3))
    stability_seeds = list(getattr(cfg, "STABILITY_SEEDS", [2026, 2027, 2028]))

    history_path = getattr(cfg, "ITERATION_HISTORY_PATH", "")
    if not history_path or not os.path.exists(history_path):
        print("[stability] iteration_history.json not found; skipping stability re-runs.")
        return

    history = load_json(history_path, default=[])
    # Only keep successful rounds that have validation metrics (selection signal).
    successful = [h for h in history if h.get("val_rmse") is not None]
    if not successful:
        print("[stability] No successful rounds; skipping stability re-runs.")
        return

    # Dedup by architecture signature (same logic as Top-K) so stability re-runs
    # cover diverse architectures rather than near-duplicate variants.
    if bool(getattr(cfg, "TOP_K_DEDUP_BY_ARCH", True)):
        for h in successful:
            h["_module_categories"] = classify_modules_from_md(h.get("module_reason_file", ""))

        def _val_v(h):
            v = safe_float(h.get("val_rmse"))
            return v if v is not None else float("inf")

        best_per_sig = {}
        for h in successful:
            sig = _arch_signature(h.get("_module_categories", {}))
            prev = best_per_sig.get(sig)
            if prev is None or _val_v(h) < _val_v(prev):
                best_per_sig[sig] = h
        dropped = len(successful) - len(best_per_sig)
        successful = list(best_per_sig.values())
        if dropped:
            print(f"[stability] Dedup by architecture signature: dropped {dropped} near-duplicate candidate(s).")

    # Select Top-K by VALIDATION RMSE (test set reserved for the final stability ranking).
    successful.sort(key=lambda h: safe_float(h.get("val_rmse", float("inf"))) or float("inf"))
    top_models = successful[:stability_top_k]

    print("\n" + "█" * 80)
    print(f"█ Phase 2: Stability re-runs — Top-{stability_top_k} models × {len(stability_seeds)} seeds")
    print("█" * 80)

    stability_dir = os.path.join(cfg.RESULT_DIR, "stability")
    os.makedirs(stability_dir, exist_ok=True)

    all_results = []

    for rank, item in enumerate(top_models):
        model_path = item.get("generated_model_file", "")
        if not model_path or not os.path.exists(model_path):
            print(f"[stability] Model file not found, skipping: {model_path}")
            continue

        iteration = item.get("iteration", "?")
        print(f"\n[stability] Model Rank={rank + 1} (iter {iteration}): {model_path}")

        seed_results = []
        for seed in stability_seeds:
            model_name = f"model_{rank + 1}_iter{iteration}"
            seed_dir = os.path.join(stability_dir, model_name, f"seed_{seed}")
            train_dir_s = os.path.join(seed_dir, "train")
            test_dir_s = os.path.join(seed_dir, "test")
            os.makedirs(train_dir_s, exist_ok=True)
            os.makedirs(test_dir_s, exist_ok=True)

            best_model_path_s = os.path.join(train_dir_s, cfg.BEST_MODEL_FILE)
            train_stats_path_s = os.path.join(train_dir_s, cfg.TRAIN_STATS_FILE)

            try:
                print(f"  Training seed={seed} ...")
                run_step(
                    "train_val_early_stop_iterative.py",
                    cfg_overrides={**npy_overrides,
                                   "TRAIN_OUTPUT_DIR": train_dir_s,
                                   "CANDIDATE_MODEL_PATH": model_path,
                                   "SEARCH_SEED": seed},
                )
            except (StepError, subprocess.CalledProcessError, RuntimeError) as err:
                stderr_detail = getattr(err, "_stderr_text", "")
                print(f"  [stability] Training failed seed={seed}: {stderr_detail[-500:]}")
                seed_results.append({"seed": seed, "error": "train_failed"})
                continue

            try:
                print(f"  Testing seed={seed} ...")
                run_step(
                    "test_metrics_iterative.py",
                    cfg_overrides={**npy_overrides,
                                   "TEST_OUTPUT_DIR": test_dir_s,
                                   "MODEL_WEIGHTS_PATH": best_model_path_s,
                                   "TRAIN_STATS_NPZ_PATH": train_stats_path_s,
                                   "CANDIDATE_MODEL_PATH": model_path,
                                   "SEARCH_SEED": seed},
                )
            except (StepError, subprocess.CalledProcessError, RuntimeError) as err:
                stderr_detail = getattr(err, "_stderr_text", "")
                print(f"  [stability] Testing failed seed={seed}: {stderr_detail[-500:]}")
                seed_results.append({"seed": seed, "error": "test_failed"})
                continue

            test_metrics_path = os.path.join(test_dir_s, cfg.TEST_METRICS_FILE)
            metrics = load_json(test_metrics_path, default={})
            overall = metrics.get("overall_metrics", {}) if isinstance(metrics, dict) else {}
            result = {
                "seed": seed,
                "rmse": safe_float(overall.get("rmse")),
                "mae": safe_float(overall.get("mae")),
                "wmape": safe_float(overall.get("wmape")),
            }
            seed_results.append(result)
            print(f"    seed={seed}: RMSE={result['rmse']}, MAE={result['mae']}, WMAPE={result['wmape']}")

        if not seed_results or all(r.get("error") for r in seed_results):
            print(f"  [stability] All seeds failed for this model; skipping.")
            continue

        # Compute mean/std
        valid = [r for r in seed_results if "error" not in r]
        rmse_vals = [r["rmse"] for r in valid if r["rmse"] is not None]
        mae_vals = [r["mae"] for r in valid if r["mae"] is not None]
        wmape_vals = [r["wmape"] for r in valid if r["wmape"] is not None]

        def _mean_std(vals):
            if not vals:
                return None, None
            arr = np.array(vals, dtype=float)
            # std is undefined for a single seed — return None rather than 0.0
            # so the CSV honestly reflects "no spread estimated".
            return float(np.mean(arr)), (float(np.std(arr, ddof=1)) if len(arr) >= 2 else None)

        mean_rmse, std_rmse = _mean_std(rmse_vals)
        mean_mae, std_mae = _mean_std(mae_vals)
        mean_wmape, std_wmape = _mean_std(wmape_vals)

        all_results.append({
            "rank_phase1": rank + 1,
            "iteration": iteration,
            "model_path": model_path,
            "module_reason_file": item.get("module_reason_file", ""),
            "iteration_folder": item.get("iteration_folder", ""),
            "val_rmse": item.get("val_rmse"),
            "val_mae": item.get("val_mae"),
            "val_loss": item.get("val_loss"),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "mean_mae": mean_mae,
            "std_mae": std_mae,
            "mean_wmape": mean_wmape,
            "std_wmape": std_wmape,
            "total_params": item.get("total_params"),
            "total_flops": item.get("total_flops"),
            "seed_results": seed_results,
        })

    if not all_results:
        print("[stability] No models passed stability re-runs; skipping CSV output.")
        return

    # Phase 3: Re-rank by mean_RMSE
    all_results.sort(key=lambda r: r["mean_rmse"] if r["mean_rmse"] is not None else float("inf"))

    _write_stability_csv(npy_path, all_results, stability_seeds, stability_dir)
    print(f"\n[stability] Stability re-runs complete; results saved to: {stability_dir}")


def _write_stability_csv(npy_path, results, seeds, output_dir):
    """Output stability re-run results CSV."""
    npy_basename = os.path.splitext(os.path.basename(npy_path))[0]
    meta_features_path = os.path.join(cfg.BASE_RESULT_DIR, "meta_features.csv")
    meta_dict = read_meta_features_dict(meta_features_path)
    meta_order = getattr(cfg, "META_FEATURE_NAMES_ORDERED", [])

    time_lag = int(getattr(cfg, "TIME_LAG", 0))
    pre_len = int(getattr(cfg, "PRE_LEN", 0))

    seed_headers = []
    for s in seeds:
        seed_headers += [f"seed{s}_RMSE", f"seed{s}_MAE", f"seed{s}_WMAPE"]

    fieldnames = (
        ["Dataset", "Original_Iteration", "Stability_Rank", "TIME_LAG", "PRE_LEN"]
        + meta_order
        + ["Selection_val_RMSE", "Selection_val_MAE", "Selection_val_loss",
           "mean_RMSE", "std_RMSE", "mean_MAE", "std_MAE", "mean_WMAPE", "std_WMAPE"]
        + seed_headers
        + ["Params", "FLOPs"]
        + ["A_Input_Preprocessing", "B_Backbone_Modeling", "C_Head_Projection", "D_Output_Inverse", "E_Combination_Paradigm", "Candidate_Model_Dir"]
    )

    rows = []
    for rank, item in enumerate(results, start=1):
        row = {
            "Dataset": npy_basename,
            "Original_Iteration": item["iteration"],
            "Stability_Rank": rank,
            "TIME_LAG": time_lag,
            "PRE_LEN": pre_len,
        }
        for name in meta_order:
            row[name] = meta_dict.get(name, "")

        row["Selection_val_RMSE"] = item.get("val_rmse")
        row["Selection_val_MAE"] = item.get("val_mae")
        row["Selection_val_loss"] = item.get("val_loss")
        row["mean_RMSE"] = item["mean_rmse"]
        row["std_RMSE"] = item["std_rmse"]
        row["mean_MAE"] = item["mean_mae"]
        row["std_MAE"] = item["std_mae"]
        row["mean_WMAPE"] = item["mean_wmape"]
        row["std_WMAPE"] = item["std_wmape"]

        seed_dict = {}
        for sr in item.get("seed_results", []):
            s = sr.get("seed")
            if s is None or "error" in sr:
                continue
            seed_dict[f"seed{s}_RMSE"] = sr.get("rmse")
            seed_dict[f"seed{s}_MAE"] = sr.get("mae")
            seed_dict[f"seed{s}_WMAPE"] = sr.get("wmape")
        for h in seed_headers:
            row[h] = seed_dict.get(h)

        cats = classify_modules_from_md(item.get("module_reason_file", ""))
        row["A_Input_Preprocessing"] = cats.get("A", "none")
        row["B_Backbone_Modeling"] = cats.get("B", "none")
        row["C_Head_Projection"] = cats.get("C", "none")
        row["D_Output_Inverse"] = cats.get("D", "none")
        row["E_Combination_Paradigm"] = cats.get("E", "none")
        row["Params"] = item.get("total_params")
        row["FLOPs"] = item.get("total_flops")
        row["Candidate_Model_Dir"] = item.get("iteration_folder", "")
        rows.append(row)

    csv_path = os.path.join(output_dir, getattr(cfg, "STABILITY_CSV_FILENAME", "stability_topk_results.csv"))
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[stability] CSV saved: {csv_path}")
    for i, row in enumerate(rows, 1):
        print(f"  #{i} iter={row['Original_Iteration']} mean_RMSE={row['mean_RMSE']} std_RMSE={row['std_RMSE']}")


def main():
    npy_files = discover_npy_files(cfg.DATASET_NPY_DIR)
    # The shared directory contains both modes; retain the configured mode.
    selected_files = []
    for npy_path in npy_files:
        with open(npy_path, "rb") as stream:
            version = np.lib.format.read_magic(stream)
            shape, _, _ = np.lib.format._read_array_header(stream, version)
        channels = shape[1] if len(shape) >= 2 else 1
        if (channels == 1) == cfg.UNIVARIATE:
            selected_files.append(npy_path)
    npy_files = selected_files
    npy_total = len(npy_files)

    # ── Limit number of npy files ACTUALLY PROCESSED per run (skip already-completed) ──
    max_npy = int(getattr(cfg, "MAX_NPY_FILES", 0) or 0)
    if max_npy > 0:
        print(f"[config] MAX_NPY_FILES={max_npy}: will process up to {max_npy} .npy files (skipping already-completed).")

    processed = 0
    display_idx = 1
    for npy_path in npy_files:
        result = run_experiment_for_npy(npy_path, display_idx, npy_total)
        display_idx += 1
        if result:
            processed += 1
            if max_npy > 0 and processed >= max_npy:
                print(f"\n[config] Reached MAX_NPY_FILES={max_npy} processed files; stopping.")
                break

    print("\n" + "█" * 80)
    print(f"█ Processed {processed} datasets this run ({npy_total} total in directory).")
    print("█" * 80)


if __name__ == "__main__":
    main()
