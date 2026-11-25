import importlib.util
import os
import re

import torch


def validate_candidate_model_runtime(
    model_path,
    time_lag,
    pre_len,
    station_num,
    batch_sizes=(1, 2, 16),
    device="cpu",
    auto_repair=True,
):
    """
    Pre-training smoke test:
    1. Can dynamically import CandidateModel
    2. forward does not error
    3. Output shape must be [B, pre_len, station_num]
    4. Output must not contain NaN/Inf
    5. backward can complete
    6. Batch size must not be hardcoded
    """
    CandidateModel = load_candidate_model_class(model_path, auto_repair=auto_repair)

    errors = []
    for batch_size in batch_sizes:
        try:
            model = CandidateModel(time_lag, pre_len, station_num).to(device)
            model.train()

            x = torch.randn(batch_size, time_lag, station_num, device=device)
            y = torch.randn(batch_size, pre_len, station_num, device=device)

            pred = model(x)

            expected_shape = (batch_size, pre_len, station_num)
            if tuple(pred.shape) != expected_shape:
                raise ValueError(
                    f"Output shape error: got {tuple(pred.shape)}, expected {expected_shape}"
                )

            if not torch.isfinite(pred).all():
                raise ValueError("Output contains NaN or Inf")

            loss = torch.mean(torch.abs(pred - y))
            loss.backward()

            has_grad = any(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.requires_grad
            )
            if not has_grad:
                raise ValueError("No valid gradients after backward")

        except Exception as exc:
            errors.append(f"batch_size={batch_size}: {type(exc).__name__}: {exc}")

    if errors:
        raise RuntimeError(
            "Candidate model runtime validation failed, training is blocked.\n" + "\n".join(errors)
        )

    print("Candidate model runtime smoke test passed.")
    return True

def repair_conv1d_padding(code):
    """Fix incorrect "same" padding in nn.Conv1d / nn.AvgPool1d / nn.MaxPool1d.

    Common LLM mistake: arbitrarily setting padding=2/3/4 for Conv1d without using the formula,
    causing the output time dimension to inflate (12->14->18), breaking residual connections.
    This function corrects padding to dilation*(kernel_size-1)//2 (Conv1d/MaxPool1d)
    or (kernel_size-1)//2 (AvgPool1d), only intervening when stride=1.
    """

    def _get_int_arg(body, name, default):
        m = re.search(rf'\b{name}\s*=\s*(\d+)', body)
        return int(m.group(1)) if m else default

    def _get_pos_kernel_size(body):
        """Try to extract kernel_size from positional arguments (the 3rd argument of Conv1d)."""
        m = re.match(r'\s*\(?\s*\d+\s*,\s*\d+\s*,\s*(\d+)', body)
        return int(m.group(1)) if m else None

    def _fix_call(match, dilation_default=1):
        full = match.group(0)
        body = match.group(1)

        kernel_size = _get_int_arg(body, 'kernel_size', None)
        if kernel_size is None:
            kernel_size = _get_pos_kernel_size(body)
        if kernel_size is None or kernel_size <= 1:
            return full

        stride = _get_int_arg(body, 'stride', 1)
        if stride != 1:
            return full

        dilation = _get_int_arg(body, 'dilation', dilation_default)
        current_padding = _get_int_arg(body, 'padding', 0)

        correct_padding = dilation * (kernel_size - 1) // 2
        if correct_padding == current_padding:
            return full

        if re.search(r'\bpadding\s*=', body):
            return re.sub(r'\bpadding\s*=\s*\d+', f'padding={correct_padding}', full)
        else:
            return full[:-1] + f', padding={correct_padding})'

    # Match nn.Conv1d(..) / nn.AvgPool1d(..) / nn.MaxPool1d(..), supporting one level of nested parentheses
    call_re = r'\(([^()]*(?:\([^()]*\)[^()]*)*)\)'

    code = re.sub(r'nn\.Conv1d' + call_re, lambda m: _fix_call(m, dilation_default=1), code)
    code = re.sub(r'nn\.AvgPool1d' + call_re, lambda m: _fix_call(m, dilation_default=1), code)
    code = re.sub(r'nn\.MaxPool1d' + call_re, lambda m: _fix_call(m, dilation_default=1), code)

    return code


def repair_common_candidate_code_errors(code):
    """Fix common, safely auto-correctable minor syntax errors in LLM-generated models."""
    code = re.sub(r"keepdim\s*=\s*False\s*,\s*keepdim\s*=\s*True", "keepdim=True", code)
    code = re.sub(r"keepdim\s*=\s*True\s*,\s*keepdim\s*=\s*False", "keepdim=True", code)
    return repair_conv1d_padding(code)


def validate_candidate_code(code, save_path, auto_repair=True):
    """Perform syntax check before saving or importing candidate model; auto-repair common errors if needed."""
    for attempt in range(2):
        try:
            compile(code, save_path, "exec")
            print("Candidate model code syntax check passed." if attempt == 0 else "Candidate model code auto-repaired and passed syntax check.")
            return code
        except SyntaxError as err:
            print(f"\nCandidate model code syntax check failed: {save_path}:{err.lineno} {err.msg}")
            if err.text:
                print("Error line:", err.text.strip())
            if not auto_repair or attempt == 1:
                raise

            repaired = repair_common_candidate_code_errors(code)
            if repaired == code:
                raise
            print("Detected common syntax issues eligible for auto-repair, repairing now.")
            code = repaired
    return code


def ensure_candidate_model_syntax(model_path, auto_repair=True):
    """Check candidate model file syntax; overwrite the file if repairable."""
    with open(model_path, "r", encoding="utf-8") as f:
        code = f.read()

    repaired = validate_candidate_code(code, model_path, auto_repair=auto_repair)
    if repaired != code:
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(repaired)


def load_candidate_model_class(model_path, auto_repair=True):
    """Dynamically load CandidateModel from an arbitrary .py file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Candidate model file not found: {model_path}")

    ensure_candidate_model_syntax(model_path, auto_repair=auto_repair)

    spec = importlib.util.spec_from_file_location("candidate_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CandidateModel"):
        raise AttributeError(f"No CandidateModel class found in {model_path}")
    return module.CandidateModel


def count_model_params(model_path, time_lag, pre_len, station_num):
    """Count total number of trainable parameters in CandidateModel."""
    CandidateModel = load_candidate_model_class(model_path, auto_repair=False)
    model = CandidateModel(time_lag, pre_len, station_num)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_model_flops(model_path, time_lag, pre_len, station_num, device="cpu"):
    """
    Calculate total FLOPs for a single forward pass of CandidateModel.

    Prefers PyTorch >= 2.4 built-in FlopCounterMode (no warnings, full coverage);
    falls back to fvcore (requires pip install fvcore);
    returns None if neither is available, without affecting the main workflow.
    """
    CandidateModel = load_candidate_model_class(model_path, auto_repair=False)
    model = CandidateModel(time_lag, pre_len, station_num).to(device)
    model.eval()
    x = torch.randn(1, time_lag, station_num, device=device)

    # Preferred: PyTorch built-in FlopCounterMode (2.4+, supports all operators)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(model, display=False) as fcm:
            _ = model(x)
        return fcm.get_total_flops()
    except ImportError:
        pass

    # Fallback: fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, x)
        return flops.total()
    except ImportError:
        print("[flops] Requires PyTorch >= 2.4 or pip install fvcore, skipping FLOPs calculation.")
        return None
    except Exception as exc:
        print(f"[flops] FLOPs calculation failed: {type(exc).__name__}: {exc}")
        return None
