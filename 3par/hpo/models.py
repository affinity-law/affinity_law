"""Parameterize only the selected L source without rerunning model selection."""

import ast
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from .losses import DEFAULT_LOSS_PROFILE

PATCH = {"p2_s1": (2, 1), "p4_s2": (4, 2), "p6_s3": (6, 3), "p8_s4": (8, 4)}

# Tune each module type independently while preserving the frozen identities and graph.
WIDTH_PARAMETERS = {
    "MLP Block": ("mlp_width_scale", "_B_MLP", "hidden"),
    "CNN Block": ("cnn_width_scale", "_B_CNN", "hidden"),
    "TCN Block": ("tcn_width_scale", "_B_TCN", "hidden"),
    "GRU Block": ("gru_width_scale", "_B_GRU", "hidden"),
    "LSTM Block": ("lstm_width_scale", "_B_LSTM", "hidden"),
    "State Space Model Block": ("ssm_width_scale", "_B_SSM", "hidden"),
    "Transformer Encoder Block": ("transformer_width_scale", "_B_Transformer", "hidden"),
    "Basis Block": ("basis_width_scale", "_B_Basis", "n_basis"),
    "MLP Head": ("mlp_head_width_scale", "_C_MLPHead", "hidden"),
    "Basis Projection": ("basis_head_width_scale", "_C_BasisProjection", "n_basis"),
}
CALL_WIDTH_PARAMETERS = {call: (parameter, keyword)
                         for parameter, call, keyword in WIDTH_PARAMETERS.values()}


def extract_spec(source):
    """Recover module identities and order from the final CandidateModel source."""
    tree = ast.parse(source)
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CandidateModel")
    init = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    spec = {"A": [], "B": []}
    for node in init.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name):
            continue
        if call.func.id in ("_build_a", "_build_b"):
            spec["A" if call.func.id == "_build_a" else "B"].append(ast.literal_eval(call.args[0]))
        elif call.func.id == "_build_c":
            spec["C"] = ast.literal_eval(call.args[0])
        elif call.func.id == "_E_Composition":
            spec["E"] = ast.literal_eval(call.args[1])
    if not spec["B"] or not {"C", "E"}.issubset(spec):
        raise ValueError("Could not recover a complete module specification from the saved L source")
    return spec


def active_space(task, settings):
    spec, configured = task["spec"], settings["space"]
    result = {name: configured[name] for name in ("lr", "weight_decay", "batch_size")}
    if "loss_profile" in configured:
        result["loss_profile"] = list(configured["loss_profile"])
    if "time_lag" in configured:
        result["time_lag"] = list(configured["time_lag"])
    for module in spec["B"] + [spec["C"]]:
        if module in WIDTH_PARAMETERS:
            result[WIDTH_PARAMETERS[module][0]] = list(configured["width_scales"])
    if "LSTM Block" in spec["B"]:
        result["lstm_layers"] = list(configured["lstm_layers"])
    if "TCN Block" in spec["B"]:
        result["tcn_kernel"] = configured["tcn_kernel"]
    if "Patch" in spec["A"]:
        result["patch_profile"] = configured["patch_profile"]
    return result


def default_config(task, settings):
    values = {"lr": 1e-3, "weight_decay": 0.0, "batch_size": 16,
              "loss_profile": DEFAULT_LOSS_PROFILE, "time_lag": 12,
              "lstm_layers": 1, "tcn_kernel": 3, "patch_profile": "p4_s2"}
    values.update({parameter: 1.0 for parameter, _, _ in WIDTH_PARAMETERS.values()})
    return {name: values[name] for name in active_space(task, settings)}


def validate_config(config, task, settings):
    space = active_space(task, settings)
    if set(config) != set(space):
        raise ValueError("Parameter fields must exactly match the active search space for this L model.")
    output = dict(config)
    for name, choices in space.items():
        value = config[name]
        if isinstance(value, bool):
            raise ValueError(f"Parameter {name} cannot be a boolean.")
        if isinstance(choices, dict):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Parameter {name} must be a finite number.")
            if not choices["low"] <= value <= choices["high"]:
                raise ValueError(f"Parameter {name} is outside the allowed range.")
            output[name] = float(value)
        else:
            if value not in choices or (isinstance(choices[0], int) and not isinstance(value, int)):
                raise ValueError(f"Parameter {name} must use an allowed categorical value with the correct type.")
            output[name] = next(choice for choice in choices if choice == value)
    return output


def candidate_code(task, config, root):
    original = (Path(root) / task["model_path"]).read_text(encoding="utf-8-sig")
    if extract_spec(original) != task["spec"]:
        raise ValueError("L source does not match the frozen module specification")
    layers = config.get("lstm_layers", 1)
    kernel = config.get("tcn_kernel", 3)
    patch = PATCH[config.get("patch_profile", "p4_s2")]
    time_lag = config.get("time_lag", 12)
    width_changed = any(config.get(parameter, 1.0) != 1.0
                        for parameter, _, _ in WIDTH_PARAMETERS.values())
    if not width_changed and layers == 1 and kernel == 3 and patch == (4, 2) and time_lag == 12:
        return original  # Preserve default source verbatim for strict compatibility checks.
    tree = ast.parse(original)
    input_positions = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        init = next((method for method in node.body
                     if isinstance(method, ast.FunctionDef) and method.name == "__init__"), None)
        if init is None:
            continue
        parameters = [arg.arg for arg in [*init.args.posonlyargs, *init.args.args]][1:]
        if "time_lag" in parameters:
            input_positions[node.name] = parameters.index("time_lag")
    applied = set()
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in {"_build_a", "_build_b", "_build_c"}:
            continue
        for call in ast.walk(function):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            name = call.func.id
            if time_lag != 12 and name in input_positions:
                # Locate the semantic constructor argument, never all literals equal to 12.
                # Generic input models may already pass a dynamic variable; leave it intact.
                lag_keyword = next((kw for kw in call.keywords if kw.arg == "time_lag"), None)
                index = input_positions[name]
                value = (lag_keyword.value if lag_keyword is not None
                         else call.args[index] if index < len(call.args) else None)
                if isinstance(value, ast.Constant) and value.value == 12:
                    if lag_keyword is not None:
                        lag_keyword.value = ast.Constant(time_lag)
                    else:
                        call.args[index] = ast.Constant(time_lag)
            width = CALL_WIDTH_PARAMETERS.get(name)
            if width and width[0] in config:
                for keyword in call.keywords:
                    if keyword.arg == width[1]:
                        keyword.value = ast.Constant(max(1, round(ast.literal_eval(keyword.value) * config[width[0]])))
                        applied.add(width[0])
            extra = {}
            if name == "_B_LSTM" and "LSTM Block" in task["spec"]["B"]:
                extra["num_layers"] = layers
            if name == "_B_TCN" and "tcn_kernel" in config:
                extra["kernel"] = kernel
            if name == "_A_Patch" and "patch_profile" in config:
                extra.update(patch_len=patch[0], step=patch[1])
            for key, value in extra.items():
                call.keywords = [kw for kw in call.keywords if kw.arg != key]
                call.keywords.append(ast.keyword(arg=key, value=ast.Constant(value)))
    changed_widths = {parameter for parameter in config
                      if parameter.endswith("_width_scale") and config[parameter] != 1.0}
    if changed_widths - applied:
        raise ValueError(f"Model source does not expose tunable widths: {sorted(changed_widths - applied)}")
    code = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    if extract_spec(code) != task["spec"]:
        raise ValueError("Parameterization changed the module composition")
    return code


def candidate_class(code):
    namespace = {"__name__": "frozen_l_candidate"}
    exec(compile(code, "<frozen_l_candidate>", "exec"), namespace)
    return namespace["CandidateModel"]


def instantiate(code, task, config=None):
    return candidate_class(code)((config or {}).get("time_lag", 12), task["horizon"], task["channels"])


def inspect_candidate(task, config, settings, root, check_forward=False, batch_sizes=(1, 2), preserve_rng=True):
    config = validate_config(config, task, settings)
    time_lag = config.get("time_lag", 12)
    code = candidate_code(task, config, root)
    with torch.random.fork_rng(devices=[]) if preserve_rng else nullcontext():
        CandidateModel = candidate_class(code)
        model = CandidateModel(time_lag, task["horizon"], task["channels"])
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = count / task["original_params"]
        if (settings["space"].get("enforce_param_ratio", True)
                and not settings["space"]["param_ratio_min"] <= ratio <= settings["space"]["param_ratio_max"]):
            raise ValueError(f"Model parameter ratio is outside the allowed range: {ratio:.6f}.")
        if check_forward:
            for index, batch in enumerate(batch_sizes):
                if index:
                    model = CandidateModel(time_lag, task["horizon"], task["channels"])
                model.train()
                x = torch.randn(batch, time_lag, task["channels"])
                y = torch.randn(batch, task["horizon"], task["channels"])
                output = model(x)
                if output.shape != (batch, task["horizon"], task["channels"]):
                    raise ValueError("Invalid output shape")
                if not torch.isfinite(output).all():
                    raise FloatingPointError("Model output contains NaN/Inf")
                (output - y).abs().mean().backward()
                if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Model gradients contain NaN/Inf")
    return {"code": code, "params": count, "param_ratio": ratio}
