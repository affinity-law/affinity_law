import ast
import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import project_config as cfg
from candidate_utils import validate_candidate_code

cfg.apply_proxy_settings()

MODULE_LIBRARY_MAX_CHARS = 130000
SEARCH_GRAMMAR_MAX_CHARS = 60000
FEEDBACK_MAX_CHARS = 50000


PROMPT = PromptTemplate.from_template("""
You are a time-series forecasting model design expert. Generate a runnable PyTorch CandidateModel from the statistical features, task hyperparameters, previous iteration results, candidate module library, and search grammar below.

[7 Abstract Indicators]
{abstract_series_info}

[Task Hyperparameters]
{task_info}

[Previous Iteration Results]
{iteration_feedback}

[Candidate Module Library]
{module_library}

[Search Grammar & Hard Constraints]
{search_grammar}

# Design rules
1. Architecture selection: pick modules ONLY from the candidate library (never fabricate names) and comply with the search grammar. Choose inductive biases based on the [7 Abstract Indicators] and justify the choice with specific indicator values in the evidence field (e.g. "noise_volatility_score=X>0.6 → TCN Block over Linear Block"). Prioritize reducing MAE — consider residual prediction, last-value add-back, local de-meaning, short-term trend branches, or multi-scale temporal convolution. You may fully rewrite the model when prior errors are high, residuals show structure, or the previous structure is unsuitable.
2. Diversity (binding): follow the [Diversity Constraint — *] sections and the [Already-Tried Architecture Signatures] list. You MUST (a) select at least one UNEXPLORED module in each of the A/C/E categories until every module in that category has been tried once, and (b) pick a backbone bias type from the UNEXPLORED types until all 6 are tried. Once all are explored you may reuse, BUT your full A/B/C/D/E signature MUST differ from every already-tried signature — vary at least one stage's module set. Merely changing hyperparameters (hidden dim, layer count) of a tried architecture is forbidden.
3. Inverse modules are first-class: every preprocessing module (Z-score / RevIN / Differencing / Series Decomposition) needs its matching D-category inverse module, each listed in selected_modules with its own rationale. Do not omit them.
4. Code quality: complete PyTorch code for CandidateModel, correct dimensions, training-stable. Self-check syntax (no duplicate kwargs like keepdim=). Avoid needless stacking but keep effective structure. The file must contain only necessary imports and one top-level class CandidateModel — no training / main / argparse, no private packages.

# Dimension & runtime safety (all must hold)
- CandidateModel(time_lag, pre_len, station_num); forward(x): x is [B, time_lag, station_num] → [B, pre_len, station_num]. Use the actual station_num from [Task Hyperparameters], never a hardcoded constant.
- nn.Linear(in_features) must match the last dim of the tensor actually passed; do not confuse time_lag / pre_len / station_num / hidden_dim.
- Preserve batch dim B at all times; never hardcode B to 1/16/32/BATCH_SIZE. reshape/view/permute must not lose B.
- Both operands of torch.bmm share the same batch dim; a learnable weight in bmm needs expand(B,-1,-1) or matmul/einsum — no batch=1 broadcasting.
- Normalization (Z-score/RevIN): use local mean/std only; never store as self.mean / self.std.
- "same" padding to keep time dim equal (stride=1): Conv1d padding = dilation*(kernel_size-1)//2; Pool1d padding = (kernel_size-1)//2.
- Before output, this pseudo-test must pass: for B in [1,2,16]: y=model(torch.randn(B,time_lag,station_num)); assert y.shape==(B,pre_len,station_num).

# Output
Write all explanations, code comments and docstrings in English.
Output valid JSON only (no Markdown, no code fences):
{{
  "selected_modules": [{{"module_name": "module or structure name", "reason": "selection rationale"}}],
  "combination_strategy": "description of module combination approach",
  "model_summary": "brief model architecture summary",
  "evidence": "specific indicator values + previous-round feedback + constraint basis",
  "changes_vs_previous_iteration": "changes vs previous round; write 'first-round generation' if first round",
  "ablation_plan": ["at least one ablation, e.g. remove the newly added module or replace a key module"],
  "candidate_code": "complete Python code string"
}}
""")


def build_llm():
    api_key = str(cfg.LLM_API_KEY).strip()
    if not api_key or api_key.startswith("Please fill in your"):
        raise RuntimeError(f"Please set the {cfg.LLM_PROVIDER} API Key in project_config.py or its environment variable.")

    options = {
        "model": cfg.LLM_MODEL,
        "api_key": api_key,
        "base_url": cfg.LLM_BASE_URL,
        "streaming": cfg.LLM_STREAMING,
    }
    if cfg.LLM_MODEL == "gpt-5.6-luna":
        options["reasoning_effort"] = cfg.LLM_REASONING_EFFORT
    elif cfg.LLM_PROVIDER != "gemini":
        options["temperature"] = cfg.LLM_TEMPERATURE
    return ChatOpenAI(**options)


def read_feature_dict(csv_path):
    if not os.path.exists(csv_path):
        print(f"[warn] CSV file not found, skipping: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[warn] CSV file is empty: {csv_path}")
        return {}

    print(f"[ok] Successfully read CSV: {csv_path}, shape={df.shape}")
    columns = {str(col).strip().lower(): col for col in df.columns}
    if {"indicator_name", "value"} <= columns.keys():
        return dict(zip(df[columns["indicator_name"]].astype(str), df[columns["value"]]))
    if df.shape[1] == 2:
        return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1]))
    return df.iloc[0].to_dict()


def format_value(value):
    if pd.isna(value):
        return "NaN"
    return f"{value:.8g}" if isinstance(value, float) else str(value)


def build_series_info(feature_dict, title):
    if not feature_dict:
        return f"{title}\nNot found or not provided."
    return "\n".join([title] + [f"{name}: {format_value(value)}" for name, value in feature_dict.items()])

def inspect_npy_shape_for_prompt(npy_path, univariate=False, time_lag=None, pre_len=None):
    """Read the actual npy shape and derive CandidateModel input/output shapes. Keep only key info."""
    if not os.path.exists(npy_path):
        return "\n".join([
            "[NPY Data Shape]",
            f"File not found: {npy_path}",
        ])

    try:
        data = np.load(npy_path, mmap_mode="r")
        raw_shape = tuple(data.shape)
        raw_ndim = data.ndim

        # Derive station_num
        if raw_ndim == 1:
            channel_num = 1
        elif raw_ndim == 2:
            channel_num = raw_shape[1]
        elif raw_ndim == 3:
            if raw_shape[2] == 1:
                channel_num = raw_shape[1]
            else:
                return "\n".join([
                    "[NPY Data Shape]",
                    f"Original shape: {raw_shape}, 3D data but F!=1, not supported by this project.",
                ])
        else:
            return "\n".join([
                "[NPY Data Shape]",
                f"Original shape: {raw_shape}, unsupported dimensionality.",
            ])

        T = raw_shape[0]
        station_num = 1 if univariate else channel_num
        var_note = "Univariate (using only variable 0)" if univariate else f"Multivariate (using all {station_num} variables)"

        window_count = None
        if time_lag is not None and pre_len is not None:
            window_count = T - int(time_lag) - int(pre_len) + 1

        lines = [
            "[NPY Data Shape]",
            f"Time length T={T}, variable count entering model station_num={station_num} ({var_note})",
            f"CandidateModel input: [B, {time_lag}, {station_num}]",
            f"CandidateModel output: [B, {pre_len}, {station_num}]",
        ]

        if window_count is not None:
            lines.append(f"Sliding window sample count: {window_count}")
            if window_count <= 0:
                lines.append("[warn] Insufficient data length: T - time_lag - pre_len + 1 <= 0.")

        lines += [
            "Hard constraints:",
            "- station_num must equal the actual variable count above; do not assume other constants.",
            "- nn.Linear(in_features) must match the last dimension of the actually passed tensor.",
            "- Batch dimension B must not be hardcoded to a specific value.",
        ]

        return "\n".join(lines)

    except Exception as exc:
        return "\n".join([
            "[NPY Data Shape]",
            f"Read failed: {type(exc).__name__}: {exc}",
        ])

def build_task_info():
    mode = "Univariate" if cfg.UNIVARIATE else "Multivariate"

    npy_shape_info = inspect_npy_shape_for_prompt(
        npy_path=cfg.NPY_PATH,
        univariate=cfg.UNIVARIATE,
        time_lag=cfg.TIME_LAG,
        pre_len=cfg.PRE_LEN,
    )

    return "\n".join([
        f"Prediction horizon pre_len: {cfg.PRE_LEN}",
        f"Input window time_lag: {cfg.TIME_LAG}",
        f"Variable mode: {mode}",
        f"batch_size: {cfg.BATCH_SIZE}",
        f"train/val/test: {cfg.TRAIN_RATIO}/{cfg.VAL_RATIO}/{cfg.TEST_RATIO}",
        "",
        npy_shape_info,
    ])


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_if_exists(path):
    return load_json(path) if os.path.exists(path) else None


def _extract_selected_modules_section(reason_md_path):
    """Extract the 'Selected Modules' section from module_selection_reason.md."""
    if not os.path.exists(reason_md_path):
        return ""
    with open(reason_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"##\s*Selected Modules\s*\n(.*?)(?=\n##\s|\Z)", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_combination_strategy_section(reason_md_path):
    """Extract the 'Combination Strategy' section from module_selection_reason.md."""
    if not os.path.exists(reason_md_path):
        return ""
    with open(reason_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r"##\s*Combination Strategy\s*\n(.*?)(?=\n##\s|\Z)", content, re.DOTALL)
    return m.group(1).strip() if m else ""


def _format_val_metrics_line(val):
    """Format the validation metrics that drive model selection into a single line."""
    def _v(key):
        v = val.get(key)
        return f"{v:.6g}" if isinstance(v, (int, float)) and v == v else "N/A"
    return f"val_rmse={_v('val_rmse')}  val_mae={_v('val_mae')}  val_loss={_v('val_loss')}"


def _format_residual_line(residual):
    """Format bias/residual_std/lag1_autocorr/residual_target_corr from residual_diagnostics dict into a single line."""
    def _v(key):
        val = residual.get(key)
        return f"{val:.6g}" if isinstance(val, (int, float)) and val == val else "N/A"
    return f"bias={_v('bias_mean_error')}  residual_std={_v('residual_std')}  lag1_ac={_v('lag1_autocorr')}  res_tgt_corr={_v('residual_target_corr')}"


def build_iteration_history():
    """Scan the result directory for all previous iterations, collecting selected modules + combination strategy + overall_metrics + residual_diagnostics for each round."""
    result_dir = getattr(cfg, "RESULT_DIR", "")
    current_iter = int(getattr(cfg, "EXPERIMENT_ITERATION", 1))
    if not result_dir or not os.path.isdir(result_dir):
        return ""

    lines = []
    template = getattr(cfg, "ITERATION_DIR_TEMPLATE", "iter_{:03d}")
    llm_out_name = getattr(cfg, "LLM_OUTPUT_DIR_NAME", "llm_output")
    test_dir_name = getattr(cfg, "TEST_DIR_NAME", "test")

    for idx in range(1, current_iter):
        iter_dir = os.path.join(result_dir, template.format(idx))
        if not os.path.isdir(iter_dir):
            continue
        reason_path = os.path.join(iter_dir, llm_out_name, cfg.MODULE_REASON_FILE)
        metrics_path = os.path.join(iter_dir, test_dir_name, cfg.TEST_METRICS_FILE)

        selected_modules = _extract_selected_modules_section(reason_path)
        combination_strategy = _extract_combination_strategy_section(reason_path)
        test_metrics = load_json_if_exists(metrics_path)
        test_metrics = test_metrics if isinstance(test_metrics, dict) else {}
        val = test_metrics.get("val_metrics", {})
        residual = test_metrics.get("residual_diagnostics", {})
        # Selection signal is validation metrics; residual diagnostics come from
        # the test set and serve as design feedback only (not a selection signal).
        metrics_str = _format_val_metrics_line(val)
        residual_str = _format_residual_line(residual)

        if selected_modules:
            entry = (
                f"Round {idx} | [val] {metrics_str} | [test residual] {residual_str}\n"
                f"  Selected Modules:\n"
                f"    {selected_modules}"
            )
            if combination_strategy:
                entry += f"\n  Combination Strategy:\n    {combination_strategy}"
            lines.append(entry)
        else:
            lines.append(
                f"Round {idx} | [val] {metrics_str} | [test residual] {residual_str}\n"
                f"  (no module selection record found)"
            )

    return "\n".join(lines) if lines else ""


# ── Module category & bias-type definitions ──
# Five categories: A=Input Preprocessing, B=Backbone Modeling, C=Head Projection,
# D=Output Inverse, E=Combination Paradigm.
_MODULE_CATEGORY = {
    "Identity": "A", "Z-score": "A", "RevIN": "A", "Differencing": "A",
    "Series Decomposition": "A", "Patch": "A", "Mask": "A",
    "Linear Block": "B", "MLP Block": "B", "CNN Block": "B", "TCN Block": "B",
    "GRU Block": "B", "LSTM Block": "B",
    "State Space Model Block": "B", "Transformer Encoder Block": "B", "Basis Block": "B",
    "Linear Head": "C", "MLP Head": "C", "Basis Projection": "C",
    "Inverse Z-score": "D", "Inverse RevIN": "D",
    "Inverse Differencing": "D", "Decomposition Recomposition": "D",
    "Single Block": "E", "Sequential Composition": "E",
    "Parallel Additive Branches": "E", "Residual Modeling": "E", "Gated Fusion": "E",
}
_CAT_LABEL = {"A": "Input Preprocessing", "B": "Backbone Modeling",
              "C": "Head Projection", "D": "Output Inverse", "E": "Combination Paradigm"}
_CAT_MODULES = {c: sorted([m for m, cat in _MODULE_CATEGORY.items() if cat == c])
                for c in "ABCDE"}

# B-category bias types (for diversity constraint)
_B_TO_BIAS = {
    "Linear Block": "Linear projection", "MLP Block": "Linear projection",
    "CNN Block": "Convolution", "TCN Block": "Convolution",
    "GRU Block": "Recurrent", "LSTM Block": "Recurrent",
    "Transformer Encoder Block": "Self-Attention",
    "State Space Model Block": "State-Space",
    "Basis Block": "Basis functions",
}
_ALL_BIAS = ["Linear projection", "Convolution", "Recurrent",
             "Self-Attention", "State-Space", "Basis functions"]


def _normalize_module_name(name):
    """Normalize LLM-generated module names to canonical form."""
    name = name.strip()
    if name in _MODULE_CATEGORY:
        return name
    # Try case-insensitive exact match first
    name_lower = name.lower()
    for canonical in _MODULE_CATEGORY:
        if canonical.lower() == name_lower:
            return canonical
    # Try collapsing whitespace
    collapsed = "".join(name.split()).lower()
    for canonical in _MODULE_CATEGORY:
        if "".join(canonical.split()).lower() == collapsed:
            return canonical
    return name


def _extract_module_names_from_md(md_path):
    """Parse module_selection_reason.md, return list of canonical module names."""
    if not os.path.exists(md_path):
        return []
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    m = re.search(r'##\s+Selected Modules\s*\n(.*?)(?=\n##\s|\Z)', content, re.DOTALL)
    if not m:
        return []
    names = re.findall(r'\d+\.\s+\*\*(.+?)\*\*', m.group(1))
    seen = set()
    result = []
    for n in names:
        c = _normalize_module_name(n.strip())
        if c in _MODULE_CATEGORY and c not in seen:
            seen.add(c)
            result.append(c)
    return result


# Preprocessing module -> its required D-category inverse (mirrors
# classify_modules_from_md in run_iterative_loop.py, so signatures match).
_A_TO_D_INVERSE = {
    "Z-score": "Inverse Z-score",
    "RevIN": "Inverse RevIN",
    "Differencing": "Inverse Differencing",
    "Series Decomposition": "Decomposition Recomposition",
}


def _arch_signature_from_modules(modules):
    """Build an order-invariant A/B/C/D/E signature string from canonical module names.

    D-category inverses are auto-inferred from A-category preprocessing modules to
    stay consistent with the selection-side signature (run_iterative_loop._arch_signature).
    """
    cats = {"A": [], "B": [], "C": [], "D": [], "E": []}
    for mod in modules:
        cat = _MODULE_CATEGORY.get(mod)
        if cat and mod not in cats[cat]:
            cats[cat].append(mod)
    for a_mod in cats["A"]:
        d_mod = _A_TO_D_INVERSE.get(a_mod)
        if d_mod and d_mod not in cats["D"]:
            cats["D"].append(d_mod)
    return " | ".join(
        f"{c}={'+'.join(sorted(cats[c])) if cats[c] else 'none'}"
        for c in "ABCDE"
    )


def build_tried_arch_signatures():
    """List the architecture signatures already tried in previous rounds, each with its
    best RMSE, so the LLM is steered away from proposing a near-duplicate architecture
    (same A/B/C/D/E composition, differing only in hyperparameters)."""
    if not bool(getattr(cfg, "LLM_AVOID_DUPLICATE_SIGNATURE", True)):
        return ""

    result_dir = getattr(cfg, "RESULT_DIR", "")
    current_iter = int(getattr(cfg, "EXPERIMENT_ITERATION", 1))
    if not result_dir or not os.path.isdir(result_dir) or current_iter <= 1:
        return ""

    template = getattr(cfg, "ITERATION_DIR_TEMPLATE", "iter_{:03d}")
    llm_out_name = getattr(cfg, "LLM_OUTPUT_DIR_NAME", "llm_output")
    test_dir_name = getattr(cfg, "TEST_DIR_NAME", "test")

    sig_best = {}  # {signature: {"best_rmse": inf, "rounds": set()}}
    for idx in range(1, current_iter):
        iter_dir = os.path.join(result_dir, template.format(idx))
        if not os.path.isdir(iter_dir):
            continue
        reason_path = os.path.join(iter_dir, llm_out_name, cfg.MODULE_REASON_FILE)
        modules = _extract_module_names_from_md(reason_path)
        if not modules:
            continue
        sig = _arch_signature_from_modules(modules)
        rmse = float("inf")
        tm = load_json_if_exists(os.path.join(iter_dir, test_dir_name, cfg.TEST_METRICS_FILE))
        if isinstance(tm, dict):
            # Use validation RMSE (the selection signal), not test RMSE.
            v = tm.get("val_metrics", {}).get("val_rmse")
            if isinstance(v, (int, float)) and v == v:
                rmse = float(v)
        entry = sig_best.setdefault(sig, {"best_rmse": float("inf"), "rounds": set()})
        entry["rounds"].add(idx)
        if rmse < entry["best_rmse"]:
            entry["best_rmse"] = rmse

    if not sig_best:
        return ""

    lines = [
        "[Already-Tried Architecture Signatures (A | B | C | D | E)]",
        "Do NOT propose a candidate whose full A/B/C/D/E signature matches any entry below. "
        "Vary at least one stage's module set so the new signature is genuinely different — "
        "merely changing hyperparameters (hidden dim, layer count) of a tried signature is forbidden.",
        "",
    ]
    for sig, info in sig_best.items():
        rmse_s = f"{info['best_rmse']:.6f}" if info["best_rmse"] != float("inf") else "N/A"
        rnds = ",".join(str(r) for r in sorted(info["rounds"]))
        lines.append(f"- [{sig}]  best val_RMSE={rmse_s}  (round {rnds})")
    return "\n".join(lines)


def build_exploration_status():
    """Scan previous iterations and report exploration status for all 5 module categories A-E.
    A-category tracks per-module diversity for input preprocessing exploration.
    B-category tracks bias-type diversity for the backbone exploration constraint."""
    result_dir = getattr(cfg, "RESULT_DIR", "")
    current_iter = int(getattr(cfg, "EXPERIMENT_ITERATION", 1))
    if not result_dir or not os.path.isdir(result_dir) or current_iter <= 1:
        return ""

    template = getattr(cfg, "ITERATION_DIR_TEMPLATE", "iter_{:03d}")
    llm_out_name = getattr(cfg, "LLM_OUTPUT_DIR_NAME", "llm_output")
    test_dir_name = getattr(cfg, "TEST_DIR_NAME", "test")

    # Collect per-round data: {round: {"modules": [...], "rmse": float}}
    rounds = {}
    for idx in range(1, current_iter):
        iter_dir = os.path.join(result_dir, template.format(idx))
        if not os.path.isdir(iter_dir):
            continue
        reason_path = os.path.join(iter_dir, llm_out_name, cfg.MODULE_REASON_FILE)
        modules = _extract_module_names_from_md(reason_path)
        if not modules:
            continue
        rmse = float("inf")
        test_metrics = load_json_if_exists(
            os.path.join(iter_dir, test_dir_name, cfg.TEST_METRICS_FILE))
        if isinstance(test_metrics, dict):
            # Use validation RMSE (the selection signal), not test RMSE.
            v = test_metrics.get("val_metrics", {}).get("val_rmse")
            if isinstance(v, (int, float)) and v == v:
                rmse = float(v)
        rounds[idx] = {"modules": modules, "rmse": rmse}

    if not rounds:
        return ""

    # Aggregate: per category, per module, track rounds and best RMSE
    trial = {}  # {module: {"rounds": set(), "best_rmse": inf}}
    for idx, r in rounds.items():
        for mod in r["modules"]:
            trial.setdefault(mod, {"rounds": set(), "best_rmse": float("inf")})
            trial[mod]["rounds"].add(idx)
            if r["rmse"] < trial[mod]["best_rmse"]:
                trial[mod]["best_rmse"] = r["rmse"]

    lines = ["[Exploration Status: Module Usage by Category]", ""]

    for cat in "ABCDE":
        lines.append(f"{cat}. {_CAT_LABEL[cat]}")
        for mod in _CAT_MODULES[cat]:
            t = trial.get(mod)
            if t and t["rounds"]:
                rnds = ", ".join(str(r) for r in sorted(t["rounds"]))
                rmse_s = f", best val_RMSE={t['best_rmse']:.6f}" if t["best_rmse"] != float("inf") else ""
                lines.append(f"  ✓ {mod} (Round {rnds}{rmse_s})")
            else:
                lines.append(f"  ✗ {mod} — NOT YET TRIED")
        lines.append("")

    # ── B-category diversity constraint ──
    bias_tried = {}  # {bias_type: {"rounds": set(), "best_rmse": inf, "modules": set()}}
    for mod, t in trial.items():
        if mod in _B_TO_BIAS:
            bt = _B_TO_BIAS[mod]
            bias_tried.setdefault(bt, {"rounds": set(), "best_rmse": float("inf"), "modules": set()})
            bias_tried[bt]["rounds"] |= t["rounds"]
            bias_tried[bt]["modules"].add(mod)
            if t["best_rmse"] < bias_tried[bt]["best_rmse"]:
                bias_tried[bt]["best_rmse"] = t["best_rmse"]

    untried_bias = [b for b in _ALL_BIAS if b not in bias_tried]
    lines.append("[Diversity Constraint — B-category Backbone Bias Types]")
    lines.append(f"Explored: {len(bias_tried)}/{len(_ALL_BIAS)}")

    for bt in _ALL_BIAS:
        if bt in bias_tried:
            s = bias_tried[bt]
            mods = ", ".join(sorted(s["modules"]))
            rmse_s = f"{s['best_rmse']:.6f}" if s["best_rmse"] != float("inf") else "N/A"
            lines.append(f"  ✓ {bt} ({mods}) — best val_RMSE={rmse_s}")
        else:
            examples = {"Linear projection": "Linear Block/MLP Block",
                        "Convolution": "CNN Block/TCN Block",
                        "Recurrent": "GRU Block/LSTM Block",
                        "Self-Attention": "Transformer Encoder Block",
                        "State-Space": "State Space Model Block",
                        "Basis functions": "Basis Block"}
            lines.append(f"  ✗ {bt} ({examples.get(bt, '')}) — NOT YET TRIED")

    if untried_bias:
        lines.append("")
        lines.append("This round you MUST select a backbone from one of the UNEXPLORED bias types above.")
        lines.append("Do NOT reuse an already-explored type until all 6 have been tried at least once.")
    else:
        best_bt = min(bias_tried.items(), key=lambda x: x[1]["best_rmse"])
        lines.append(f"\nAll 6 bias types explored. Best: {best_bt[0]} (RMSE={best_bt[1]['best_rmse']:.6f}). You may freely select.")

    # ── A-category diversity constraint ──
    tried_a = [mod for mod in _CAT_MODULES["A"] if mod in trial and trial[mod]["rounds"]]
    untried_a = [mod for mod in _CAT_MODULES["A"] if mod not in trial or not trial[mod].get("rounds")]

    lines.append("")
    lines.append("[Diversity Constraint — A-category Input Preprocessing]")
    lines.append(f"Explored: {len(tried_a)}/{len(_CAT_MODULES['A'])}")

    for mod in _CAT_MODULES["A"]:
        if mod in trial and trial[mod]["rounds"]:
            rnds = ", ".join(str(r) for r in sorted(trial[mod]["rounds"]))
            rmse_s = f"{trial[mod]['best_rmse']:.6f}" if trial[mod]["best_rmse"] != float("inf") else "N/A"
            lines.append(f"  ✓ {mod} — best val_RMSE={rmse_s} (Round {rnds})")
        else:
            lines.append(f"  ✗ {mod} — NOT YET TRIED")

    if untried_a:
        lines.append("")
        lines.append("This round you MUST select at least one UNEXPLORED A-category preprocessing module above.")
        lines.append("Do NOT reuse only already-explored A modules until all 7 have been tried at least once.")
    else:
        best_a = min(
            ((mod, trial[mod]["best_rmse"]) for mod in _CAT_MODULES["A"] if mod in trial),
            key=lambda x: x[1],
        )
        lines.append(f"\nAll 7 A-category modules explored. Best: {best_a[0]} (RMSE={best_a[1]:.6f}). You may freely select.")

    # ── C-category diversity constraint (Head Projection: 3 modules) ──
    tried_c = [mod for mod in _CAT_MODULES["C"] if mod in trial and trial[mod]["rounds"]]
    untried_c = [mod for mod in _CAT_MODULES["C"] if mod not in trial or not trial[mod].get("rounds")]

    lines.append("")
    lines.append("[Diversity Constraint — C-category Head Projection]")
    lines.append(f"Explored: {len(tried_c)}/{len(_CAT_MODULES['C'])}")

    for mod in _CAT_MODULES["C"]:
        if mod in trial and trial[mod]["rounds"]:
            rnds = ", ".join(str(r) for r in sorted(trial[mod]["rounds"]))
            rmse_s = f"{trial[mod]['best_rmse']:.6f}" if trial[mod]["best_rmse"] != float("inf") else "N/A"
            lines.append(f"  ✓ {mod} — best val_RMSE={rmse_s} (Round {rnds})")
        else:
            lines.append(f"  ✗ {mod} — NOT YET TRIED")

    if untried_c:
        lines.append("")
        lines.append("This round you MUST select at least one UNEXPLORED C-category module above.")
        lines.append("Do NOT reuse only already-explored C modules until all 3 have been tried at least once.")
    else:
        best_c = min(
            ((mod, trial[mod]["best_rmse"]) for mod in _CAT_MODULES["C"] if mod in trial),
            key=lambda x: x[1],
        )
        lines.append(f"\nAll 3 C-category modules explored. Best: {best_c[0]} (RMSE={best_c[1]:.6f}). You may freely select.")

    # ── E-category diversity constraint (Combination Paradigm: 5 modules) ──
    tried_e = [mod for mod in _CAT_MODULES["E"] if mod in trial and trial[mod]["rounds"]]
    untried_e = [mod for mod in _CAT_MODULES["E"] if mod not in trial or not trial[mod].get("rounds")]

    lines.append("")
    lines.append("[Diversity Constraint — E-category Combination Paradigm]")
    lines.append(f"Explored: {len(tried_e)}/{len(_CAT_MODULES['E'])}")

    for mod in _CAT_MODULES["E"]:
        if mod in trial and trial[mod]["rounds"]:
            rnds = ", ".join(str(r) for r in sorted(trial[mod]["rounds"]))
            rmse_s = f"{trial[mod]['best_rmse']:.6f}" if trial[mod]["best_rmse"] != float("inf") else "N/A"
            lines.append(f"  ✓ {mod} — best val_RMSE={rmse_s} (Round {rnds})")
        else:
            lines.append(f"  ✗ {mod} — NOT YET TRIED")

    if untried_e:
        lines.append("")
        lines.append("This round you MUST select at least one UNEXPLORED E-category module above.")
        lines.append("Do NOT reuse only already-explored E modules until all 5 have been tried at least once.")
    else:
        best_e = min(
            ((mod, trial[mod]["best_rmse"]) for mod in _CAT_MODULES["E"] if mod in trial),
            key=lambda x: x[1],
        )
        lines.append(f"\nAll 5 E-category modules explored. Best: {best_e[0]} (RMSE={best_e[1]:.6f}). You may freely select.")

    return "\n".join(lines)


def build_iteration_feedback():
    parts = []

    # Model structure + error metrics + residual diagnostics for all previous rounds
    history = build_iteration_history()
    if history:
        parts.append(f"[Iteration History (excluding current round)]\n{history}")

    # Structured exploration status: which bias types have been tried / are unexplored
    exploration = build_exploration_status()
    if exploration:
        parts.append(exploration)

    # Already-tried full architecture signatures (avoid near-duplicate proposals)
    tried_sigs = build_tried_arch_signatures()
    if tried_sigs:
        parts.append(tried_sigs)

    # Failure feedback
    failure_feedback_path = getattr(cfg, "LATEST_FAILURE_FEEDBACK_PATH", "")
    if failure_feedback_path:
        failure_feedback = load_json_if_exists(failure_feedback_path)
        if failure_feedback:
            parts += [
                "The previous round's candidate model failed runtime validation. The following errors must be fixed one by one:",
                json.dumps(failure_feedback, ensure_ascii=False, indent=2),
            ]

    return "\n\n".join(parts)[-FEEDBACK_MAX_CHARS:] if parts else "First iteration: no historical results available yet."


def format_json_for_prompt(data, max_chars, overflow_note):
    text = json.dumps(data, ensure_ascii=False, indent=2)
    return text if len(text) <= max_chars else text[:max_chars] + overflow_note


def format_module_library(module_library):
    if isinstance(module_library, dict):
        for key in ["modules", "module_library", "module_set"]:
            if isinstance(module_library.get(key), list):
                module_library = module_library[key]
                break
    return format_json_for_prompt(module_library, MODULE_LIBRARY_MAX_CHARS, "\n...[Content too long; truncated. Do not fabricate modules not shown.]")


def format_search_grammar(search_grammar):
    return format_json_for_prompt(search_grammar, SEARCH_GRAMMAR_MAX_CHARS, "\n...[Search Grammar content too long; truncated.]")




def extract_json_text(text):
    match = re.search(r"```json\s*(.*?)```", text, re.S | re.I)
    if match:
        return match.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1].strip() if start >= 0 and end > start else text.strip()


def parse_llm_json(text) -> dict[str, Any]:
    json_text = extract_json_text(text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        payload = ast.literal_eval(json_text)
        if isinstance(payload, dict):
            return payload
        raise ValueError("LLM output is not valid JSON.")


def extract_python_code(payload):
    code = str(payload.get("candidate_code", "")).strip()
    if not code:
        raise ValueError("LLM JSON is missing the candidate_code field.")

    match = re.search(r"```(?:python)?\s*(.*?)```", code, re.S | re.I)
    return match.group(1).strip() if match else code


def print_llm_input_paths():
    items = [
        ("Time-series npy data", cfg.NPY_PATH),
        ("7 abstract indicators CSV", cfg.ABSTRACT_STATS_CSV_PATH),
        ("Candidate module library JSON", cfg.MODULE_LIBRARY_PATH),
        ("Search grammar constraints JSON", cfg.SEARCH_GRAMMAR_PATH),
        ("Previous test feedback", cfg.LATEST_TEST_METRICS_PATH),
        ("Previous train feedback", cfg.LATEST_TRAIN_HISTORY_PATH),
    ]
    print("=" * 80)
    print("Data file paths provided to LLM for this round")
    print("=" * 80)
    for name, path in items:
        print(f"{name}: {path} [{'exists' if os.path.exists(path) else 'not found / may not be used this round'}]")
    print("=" * 80)


def build_reason_report(payload):
    lines = ["# LLM Module Selection Rationale", ""]
    if payload.get("model_summary"):
        lines += ["## Model Architecture Summary", str(payload["model_summary"]), ""]

    lines.append("## Selected Modules")
    for idx, item in enumerate(payload.get("selected_modules", []), start=1):
        if isinstance(item, dict):
            lines += [f"{idx}. **{item.get('module_name', f'Module {idx}')}**", f"   - Rationale: {item.get('reason', 'No rationale provided')}"]
        else:
            lines += [f"{idx}. **Module {idx}**", f"   - Rationale: {item}"]

    lines += ["", "## Combination Strategy", str(payload.get("combination_strategy", "No combination strategy provided.")), ""]
    lines += ["## Evidence / Basis from Papers or Previous Results", str(payload.get("evidence", "No evidence provided.")), ""]
    return "\n".join(lines)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ── JSON fix prompt template for retry ──
_JSON_FIX_PROMPT = """Your previous JSON output could not be parsed due to a formatting error.

=== Parse Error ===
{error_message}

=== Your Previous Output (last 4000 chars) ===
{previous_output}

Please fix the JSON formatting issue (common causes: missing comma between fields, unescaped double-quote inside a string value, trailing comma, or unmatched braces) and output ONLY the corrected valid JSON. Do NOT change the model design or content — only fix the JSON syntax so it can be parsed by json.loads()."""


def _llm_call_with_retry(llm, prompt, llm_output_dir, raw_output_filename):
    """Call LLM with retry-on-JSON-parse-failure.

    Returns (payload_dict, raw_llm_text). Retries up to LLM_JSON_RETRY_MAX times
    when parse_llm_json fails, sending the parse error back to the LLM for correction.
    """
    retry_max = int(getattr(cfg, "LLM_JSON_RETRY_MAX", 3))
    llm_raw_text = ""
    last_error = None

    for retry_idx in range(retry_max + 1):
        if retry_idx == 0:
            current_prompt = prompt
            print(f"Starting LLM call to generate candidate model; prompt char count: {len(current_prompt)}")
        else:
            current_prompt = _JSON_FIX_PROMPT.format(
                error_message=str(last_error),
                previous_output=llm_raw_text[-4000:],
            )
            print(f"\n[JSON retry {retry_idx}/{retry_max}] LLM JSON parse failed: {last_error}")
            print(f"[JSON retry {retry_idx}/{retry_max}] Sending fix request to LLM...")

        chunks = []
        for chunk in llm.stream(current_prompt):
            if chunk.content:
                chunks.append(chunk.content)
        llm_raw_text = "".join(chunks)

        # Save raw output (overwrite on retry with _retryN suffix)
        raw_llm_output_path = os.path.join(llm_output_dir, raw_output_filename)
        if retry_idx > 0:
            raw_llm_output_path = os.path.join(
                llm_output_dir,
                raw_output_filename.replace(".txt", f"_retry{retry_idx}.txt"),
            )
        save_text(raw_llm_output_path, llm_raw_text)
        print("LLM raw output saved to:", raw_llm_output_path)

        if not llm_raw_text.strip():
            last_error = "LLM returned an empty response"
            if retry_idx >= retry_max:
                raise RuntimeError(
                    "LLM returned an empty response after all retries. Possible causes:\n"
                    "1. Invalid API Key or quota exhausted\n"
                    "2. Network connectivity issues / proxy configuration issues\n"
                    "3. API server rate limiting or temporarily unavailable\n"
                    "4. Prompt too long, exceeding model context window\n"
                    f"Current prompt length: {len(current_prompt)} characters\n"
                    "Please check and retry."
                )
            continue

        try:
            payload = parse_llm_json(llm_raw_text)
            print("LLM generation complete. JSON parsed successfully.")
            if retry_idx > 0:
                print(f"[JSON retry] Successfully fixed after {retry_idx} retry(ies).")
            return payload, llm_raw_text
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if retry_idx >= retry_max:
                raise RuntimeError(
                    f"LLM JSON parsing failed after {retry_max} retry(ies). "
                    f"Last error: {last_error}\n"
                    f"Last raw output saved to: {raw_llm_output_path}\n"
                    "Please check the raw output for JSON formatting issues."
                ) from exc

    # Should not reach here
    raise RuntimeError("LLM JSON parsing failed: unexpected exit from retry loop.")


def main():
    os.makedirs(cfg.RUNTIME_DIR, exist_ok=True)
    print_llm_input_paths()

    # Read abstract indicators, task info, and previous-round feedback
    abstract_info = build_series_info(read_feature_dict(cfg.ABSTRACT_STATS_CSV_PATH), "[Abstract Statistical Features: 7 Abstract Indicators]")
    task_info = build_task_info()
    feedback = build_iteration_feedback()

    # Always provide module library and search grammar
    module_library_text = format_module_library(load_json(cfg.MODULE_LIBRARY_PATH))
    search_grammar_text = format_search_grammar(load_json(cfg.SEARCH_GRAMMAR_PATH))

    final_prompt = PROMPT.format(
        abstract_series_info=abstract_info,
        task_info=task_info,
        iteration_feedback=feedback,
        module_library=module_library_text,
        search_grammar=search_grammar_text,
    )

    prompt_save_path = os.path.join(cfg.RUNTIME_DIR, getattr(cfg, "LLM_PROMPT_FILE", "llm_prompt.txt"))
    save_text(prompt_save_path, final_prompt)
    print("Prompt for this round saved to:", prompt_save_path)

    llm = build_llm()
    llm_output_dir = os.path.dirname(cfg.SELECTION_REASON_SAVE_PATH)
    raw_output_filename = getattr(cfg, "LLM_RAW_OUTPUT_FILE", "llm_raw_output.txt")

    payload, _llm_raw_text = _llm_call_with_retry(
        llm, final_prompt, llm_output_dir, raw_output_filename,
    )

    print("Parsing and validating code...")
    candidate_code = extract_python_code(payload)
    candidate_code = validate_candidate_code(
        candidate_code,
        cfg.CANDIDATE_MODEL_PATH,
        auto_repair=bool(getattr(cfg, "AUTO_REPAIR_CANDIDATE_SYNTAX", True)),
    )

    save_text(cfg.CANDIDATE_MODEL_PATH, candidate_code)
    save_text(cfg.SELECTION_REASON_SAVE_PATH, build_reason_report(payload))

    print("Model code saved to:", cfg.CANDIDATE_MODEL_PATH)
    print("Module selection rationale saved to:", cfg.SELECTION_REASON_SAVE_PATH)


if __name__ == "__main__":
    main()
