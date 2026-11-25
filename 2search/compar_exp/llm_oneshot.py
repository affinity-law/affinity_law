# -*- coding: utf-8 -*-
"""
Generate unrestricted PyTorch model code with an LLM.
=====================================================================
Design
    Give meta-features to the LLM and request the full source of a PyTorch CandidateModel.
    Designs may extend beyond the 24 assembler motifs. The prompt contains
    dataset features, the CandidateModel signature, input/output shapes, shape-safety requirements and
    a short request to"avoid unnecessary complexity"as brief complexity guidance.
    Validate generated code with syntax and runtime smoke checks from 1experiment/candidate_utils.
    Check import, forward shapes, NaN/Inf, backward and support for variable batch sizes.

Follow LLM_with_RAG_iterative: the model returns JSON with source code
    in candidate_code; _extract_code removes any code fences.

Generate code in one call. On JSON, syntax or runtime failure,
    retry with error feedback, up to three attempts.

Each seed calls llm_generate_code independently to generate and train its model.
    The prompt permits repeated architectures across seeds.

Save the full system and user messages as text before each API call.
    Require prompt_dir and prompt_tag. Remove prompts from failed network connections
    and number the remaining attempts consecutively to distinguish design retries.
"""

import ast
import json
import re
import sys
import tempfile
from pathlib import Path

import config

_FEAT_CN = {
    "distribution_shape_score": "distribution shape",
    "trend_strength_score": "trend strength",
    "stationarity_score": "stationarity",
    "periodicity_score": "periodicity",
    "temporal_dependence_score": "temporal dependence",
    "noise_volatility_score": "noise volatility",
    "multivariate_structure_score": "multivariate structure",
}

FEATURE_ORDER = [
    "distribution_shape_score", "trend_strength_score", "stationarity_score",
    "periodicity_score", "temporal_dependence_score", "noise_volatility_score",
    "multivariate_structure_score",
]

SYSTEM_PROMPT = (
    "You are a concise time-series model designer. "
    "Write descriptions, code comments and docstrings in English. "
    "Do NOT show reasoning or thinking. "
    "Output ONLY the requested JSON object, no extra text, no markdown fences."
)


def _load_candidate_utils():
    """Lazily load runtime validation and parameter counting from 1experiment/candidate_utils."""
    run_dir = str(config.RUN_PIPELINE_DIR)
    if run_dir not in sys.path:
        sys.path.insert(0, run_dir)
    from candidate_utils import (validate_candidate_model_runtime,
                                 count_model_params)
    return validate_candidate_model_runtime, count_model_params


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(meta, horizon, station_num, param_target, feedback=None):
    feats = "\n".join(f"  - {_FEAT_CN[k]}: {meta[k]:.3f}" for k in FEATURE_ORDER
                      if k in meta)
    fb = f"\n[Previous attempt feedback] {feedback}\n" if feedback else ""

    simplicity = "Avoid unnecessary complexity and do not stack excessive parameters."
    param_lo = max(1, int(round(param_target * config.LLM_PARAM_LO)))
    param_hi = max(param_lo, int(round(param_target * config.LLM_PARAM_HI)))

    return f"""You are a time-series forecasting model design expert. Generate a runnable PyTorch model.

Data: {station_num}-variable series, input window {config.TIME_LAG} steps, predict next {horizon} steps.
Meta-features of the series:
{feats}
Capacity reference: the comparison model has {param_target} trainable parameters.
As a soft capacity constraint, aim for approximately {param_target} parameters and preferably stay
within [{param_lo}, {param_hi}]. Do not increase complexity merely to approach the upper bound.
Generate a complete PyTorch nn.Module class CandidateModel with:
  - __init__(self, time_lag, pre_len, station_num)
  - forward(self, x): x is [B, time_lag, station_num] → return [B, pre_len, station_num]

Rules (all must hold):
  - Use the actual station_num argument; never hardcode it.
  - Preserve the batch dim B at all times; never hardcode B.
  - Keep the time dim consistent: Conv1d padding = dilation*(kernel_size-1)//2 when stride=1.
  - The file contains ONLY necessary imports and ONE class CandidateModel — no training/main/argparse.

{simplicity}
{fb}
Output valid JSON only (no Markdown, no code fences):
{{"candidate_code": "complete Python code string"}}
"""


# ---------------------------------------------------------------------------
# Save one prompt per call for inspection and reproduction.
# ---------------------------------------------------------------------------

def _save_prompt(prompt_dir, prompt_tag, attempt, system_text, human_text):
    """Save both the system and user messages before calling the API."""
    p = Path(prompt_dir) / f"prompt_{prompt_tag}_a{attempt + 1}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    content = f"[SYSTEM]\n{system_text}\n\n[HUMAN]\n{human_text}"
    p.write_text(content, encoding="utf-8")
    return p


def _next_prompt_attempt(prompt_dir, prompt_tag):
    """Return the next available attempt number across resumed processes."""
    prompt_dir = Path(prompt_dir)
    numbers = []
    pattern = re.compile(rf"^prompt_{re.escape(prompt_tag)}_a(\d+)\.txt$")
    for path in prompt_dir.glob(f"prompt_{prompt_tag}_a*.txt"):
        match = pattern.match(path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0)


def _renumber_prompt_attempts(prompt_dir, prompt_tag):
    """Renumber retained prompts safely as a1, a2, ... after removing failed calls."""
    prompt_dir = Path(prompt_dir)
    pattern = re.compile(rf"^prompt_{re.escape(prompt_tag)}_a(\d+)\.txt$")
    files = []
    for path in prompt_dir.glob(f"prompt_{prompt_tag}_a*.txt"):
        match = pattern.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    temporary = []
    for index, (_, path) in enumerate(sorted(files), 1):
        tmp = path.with_name(f".{path.name}.renaming")
        path.replace(tmp)
        temporary.append((index, tmp))
    for index, tmp in temporary:
        tmp.replace(prompt_dir / f"prompt_{prompt_tag}_a{index}.txt")


# ---------------------------------------------------------------------------
# Parse the JSON response and extract model code.
# ---------------------------------------------------------------------------

def _extract_json_text(text):
    m = re.search(r"```json\s*(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1].strip() if start >= 0 and end > start else text.strip()


def _parse_json(text):
    json_text = _extract_json_text(text)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        payload = ast.literal_eval(json_text)
        if isinstance(payload, dict):
            return payload
        raise ValueError("LLM output is not valid JSON.")


def _extract_code(payload):
    code = str(payload.get("candidate_code", "")).strip()
    if not code:
        raise ValueError("LLM JSON is missing the candidate_code field.")
    m = re.search(r"```(?:python)?\s*(.*?)```", code, re.S | re.I)
    return m.group(1).strip() if m else code


# ---------------------------------------------------------------------------
# Generate model code
# ---------------------------------------------------------------------------

def llm_generate_code(meta, horizon, station_num, param_target, max_retries=3,
                      prompt_dir=None, prompt_tag=""):
    """
    Generate CandidateModel and return (code, n_params).
    Return (None, None) after all attempts fail.
    meta: dict{seven feature names: values}; station_num: variable count; param_target: reference L parameter count.
    prompt_dir and prompt_tag are required to archive each complete prompt before its API call.
    """
    if prompt_dir is None:
        raise ValueError("prompt_dir is required to save the full prompt before the LLM call")
    if not str(prompt_tag).strip():
        raise ValueError("prompt_tag is required to uniquely identify each prompt file")

    from langchain_openai import ChatOpenAI

    api_key = config.LLM_API_KEY
    if not api_key:
        import os
        api_key = os.environ.get("LLM_API_KEY", "")

    llm = ChatOpenAI(model=config.LLM_MODEL, api_key=api_key,
                     base_url=config.LLM_BASE_URL, temperature=config.LLM_TEMPERATURE,
                     timeout=120, max_retries=1, max_tokens=config.LLM_MAX_TOKENS)

    from langchain_core.messages import SystemMessage, HumanMessage
    system = SystemMessage(content=SYSTEM_PROMPT)

    validate_runtime, count_params = _load_candidate_utils()

    feedback = None
    network_failed_prompts = []
    prompt_attempt_offset = _next_prompt_attempt(prompt_dir, prompt_tag)
    for attempt in range(max_retries + 1):
        prompt = _build_prompt(meta, horizon, station_num, param_target, feedback)
        prompt_path = _save_prompt(
            prompt_dir, prompt_tag, prompt_attempt_offset + attempt,
            SYSTEM_PROMPT, prompt)
        try:
            resp = llm.invoke([system, HumanMessage(content=prompt)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            payload = _parse_json(text)
            code = _extract_code(payload)
        except Exception as exc:
            is_network_error = (type(exc).__name__ == "APIConnectionError" or
                                "Connection error" in str(exc))
            if is_network_error:
                network_failed_prompts.append(prompt_path)
                feedback = None
            else:
                feedback = (f"response parse failed ({type(exc).__name__}: {str(exc)[:120]}). "
                            f"Return ONLY valid JSON with a candidate_code field.")
            continue

        # Check syntax and runtime; reload any code modified by automatic repair.
        try:
            with tempfile.TemporaryDirectory(prefix="llmgen_") as tmp:
                code_path = Path(tmp) / "candidate_model.py"
                code_path.write_text(code, encoding="utf-8")
                validate_runtime(str(code_path), config.TIME_LAG, horizon, station_num)
                code = code_path.read_text(encoding="utf-8")
                n_params = count_params(str(code_path), config.TIME_LAG, horizon, station_num)
        except Exception as exc:
            feedback = (f"model code validation failed ({type(exc).__name__}): "
                        f"{str(exc)[:300]}. Fix the code and return valid JSON again.")
            continue

        print(f"  [llm] attempt{attempt + 1}: model code passed validation, params={n_params}")
        for failed_path in network_failed_prompts:
            failed_path.unlink(missing_ok=True)
        _renumber_prompt_attempts(prompt_dir, prompt_tag)
        return code, n_params

    for failed_path in network_failed_prompts:
        failed_path.unlink(missing_ok=True)
    _renumber_prompt_attempts(prompt_dir, prompt_tag)
    print("  [llm] No usable code after retries")
    return None, None
