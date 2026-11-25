"""LLM review context, decision validation, and API client for staged search."""

import json

from .config import require_llm
from .coordinator import TRAINING_PARAMETERS, record_complete, record_score
from .losses import loss_details
from .models import active_space, default_config, inspect_candidate, validate_config


class AgentAPIError(RuntimeError):
    """Pause when the API is unavailable instead of silently skipping a review."""


DIRECT_TRANSPORT = {"mode": "direct", "trust_env": False, "proxy": None}


def api_error_details(exc, api_key=None):
    """Capture a redacted exception chain for network diagnostics."""
    chain, current = [], exc
    while current is not None and len(chain) < 4:
        message = str(current)
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
        chain.append({"type": type(current).__name__, "message": message[:1000]})
        current = current.__cause__ or current.__context__
    response = getattr(exc, "response", None)
    body = getattr(exc, "body", None)
    if body is None and response is not None:
        try:
            body = response.text
        except Exception:
            body = None
    text = json.dumps(body, ensure_ascii=False) if isinstance(body, (dict, list)) else str(body or "")
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return {"type": type(exc).__name__, "status_code": getattr(exc, "status_code", None),
            "chain": chain, "response_body": text[:8000]}


def summary(record, defaults):
    """Compact one historical slot; full training curves remain in local artifacts."""
    result = record["result"]
    config = record.get("executed_config") or {}
    item = {"slot": record["slot"], "status": record.get("screening_status", result.get("status")),
            "round": record.get("search_round"), "parameter": record.get("parameter"),
            "search_priority": record.get("search_priority"),
            "tested_value": record.get("probe_value"),
            "changes_from_default": {name: value for name, value in config.items()
                                     if defaults.get(name) != value},
            **{key: result.get(key) for key in ("val_rmse", "val_loss", "best_epoch",
                                               "epochs_run", "params", "param_ratio")}}
    if "screening_results" in record:
        item.update(val_rmse=record_score(record),
                    val_rmse_std=record.get("screening_val_rmse_std"),
                    ranking_metric="mean_validation_rmse_across_screening_seeds",
                    seed_results={str(seed): {key: fit["result"].get(key)
                                             for key in ("status", "val_rmse", "val_loss")}
                                  for seed, fit in record["screening_results"].items()})
    curve = result.get("curve") or {}
    first, last = curve.get("first", {}), curve.get("last", {})
    if first or last:
        item["loss_trend"] = {"train_first": first.get("train_loss"),
                              "train_last": last.get("train_loss"),
                              "val_first": first.get("val_loss"),
                              "val_last": last.get("val_loss")}
    if not record_complete(record):
        item["failure"] = result.get("failure_kind") or result.get("error_type") or "training_failed"
    return {key: value for key, value in item.items() if value is not None}


def get_search_state(task, settings, proposal, records, slot):
    defaults = default_config(task, settings)
    space = active_space(task, settings)
    default_time_lag = settings["time_lag"]
    enforce_param_ratio = settings["space"].get("enforce_param_ratio", True)
    selected_loss = loss_details(proposal["config"], settings)
    loss_tunable = "loss_profile" in space
    loss_protocol = ({"loss_tunable": True, "selected_loss": selected_loss,
                      "loss_profiles": {profile: loss_details({"loss_profile": profile}, settings)
                                        for profile in space["loss_profile"]}}
                     if loss_tunable else {"loss_tunable": False, "fixed_loss": selected_loss})
    fixed_loss_fields = ["mse_weight", "mae_weight"] + ([] if loss_tunable else ["loss_profile"])
    complete = [record for record in records if record_complete(record)]
    incumbent = min(complete, key=lambda record: (record_score(record), record["slot"]), default=None)
    incumbent_config = incumbent["executed_config"] if incumbent is not None else defaults
    diff = {key: {"current": incumbent_config[key], "proposal": value}
            for key, value in proposal["config"].items()
            if incumbent_config[key] != value}
    length = task["length"]
    if settings.get("smoke"):
        length = min(int(length * settings["train_ratio"]), settings["smoke_prefix_points"])
    return {"slot": slot, "remaining_slots": settings["search"]["trials"] - slot + 1,
            "executed_slots": [record["slot"] for record in records if record["executed_config"] is not None],
            "task": {"horizon": task["horizon"], "channels": task["channels"],
                     "train_points": int(length * settings["train_ratio"]), "engineering_smoke": settings.get("smoke", False),
                     "time_lag": proposal["config"].get("time_lag", default_time_lag),
                     "default_time_lag": default_time_lag,
                     "time_lag_choices": space.get("time_lag", [default_time_lag]),
                     "input_window_tunable": "time_lag" in space,
                     "spec": task["spec"], "original_params": task["original_params"],
                     "meta_features": None,
                     "meta_feature_note": "Use meta-features only as weak context."},
            "training_protocol": {"max_epochs": settings["training"]["short_epochs"],
                                  "early_stop_patience": settings["training"]["short_patience"],
                                  "early_stop_metric": selected_loss["checkpoint_metric"],
                                  "screening_seeds": settings["search"].get("screening_seeds", [2026, 2027]),
                                  "ranking_metric": "mean_validation_rmse_across_screening_seeds",
                                  **loss_protocol,
                                  "fixed_parameters": ["max_epochs", "early_stop_patience", "screening_seeds",
                                                       "horizon", "train_ratio", "val_ratio", *fixed_loss_fields]},
            "evaluation_protocol": {"forecast_origin_reference_time_lag": default_time_lag,
                                    "comparison": "All input windows use the same validation and test forecast origins and target dates; longer windows use earlier known history."},
            "parameter_count_policy": {"enforce_limits": enforce_param_ratio,
                                       "ratio_limits": ([settings["space"]["param_ratio_min"],
                                                         settings["space"]["param_ratio_max"]]
                                                        if enforce_param_ratio else None),
                                       "usage": "validity_constraint" if enforce_param_ratio else "recording_only"},
            "space": space, "coordinate_proposal": proposal["config"],
            "proposal_source": proposal["source"], "difference_from_incumbent": diff,
            "incumbent_slot": incumbent["slot"] if incumbent is not None else None,
            "search_priority": proposal.get("search_priority"),
            "effective_priority": proposal.get("effective_priority"),
            "review_stage": proposal.get("review_stage"),
            "coordinate_parameter": proposal.get("parameter"),
            "search_round": proposal.get("search_round"),
            "editable_parameters": proposal.get("editable_parameters", []),
            "editable_values": proposal.get("editable_values", {}),
            "protected_parameters": [name for name in space
                                     if name not in proposal.get("editable_parameters", [])] + fixed_loss_fields,
            "history": [summary(record, defaults) for record in records]}


def parse_decision(text, state, task, settings, root, seen):
    decision = json.loads(text)
    fields = {"action", "overrides", "evidence_trials", "hypothesis"}
    if not isinstance(decision, dict) or set(decision) != fields:
        raise ValueError("Return exactly action, overrides, evidence_trials, and hypothesis.")
    if decision["action"] not in {"accept_proposal", "edit_proposal"}:
        raise ValueError("Unknown action. Use accept_proposal or edit_proposal.")
    overrides = decision["overrides"]
    if not isinstance(overrides, dict) or len(overrides) > 2:
        raise ValueError("overrides must be an object containing at most two parameters.")
    if decision["action"] == "accept_proposal" and overrides:
        raise ValueError("accept_proposal requires empty overrides.")
    if decision["action"] == "edit_proposal" and not overrides:
        raise ValueError("edit_proposal requires at least one override.")
    editable = set(state.get("editable_parameters", []))
    protected = set(overrides) - editable
    if protected:
        names = ", ".join(sorted(protected))
        raise ValueError(f"These parameters are protected by the search schedule and cannot be overridden: {names}.")
    editable_values = state.get("editable_values", {})
    invalid_values = [name for name, value in overrides.items()
                      if value not in editable_values.get(name, [])]
    if invalid_values:
        names = ", ".join(sorted(invalid_values))
        raise ValueError(f"Overrides must reuse already tested values listed in editable_values: {names}.")
    if not isinstance(decision["hypothesis"], str) or not 1 <= len(decision["hypothesis"]) <= 500:
        raise ValueError("hypothesis must be a short string of 1-500 characters.")
    if not decision["hypothesis"].isascii():
        raise ValueError("Write hypothesis in English using plain ASCII letters and punctuation.")
    evidence = decision["evidence_trials"]
    if not isinstance(evidence, list) or any(type(value) is not int or value not in state["executed_slots"] for value in evidence):
        raise ValueError("evidence_trials must contain only integer slot IDs that have already been executed.")
    config = validate_config({**state["coordinate_proposal"], **overrides}, task, settings)
    if state.get("search_round") == "global":
        defaults = default_config(task, settings)
        if sum(config[name] != defaults[name] for name in config) < 2:
            raise ValueError("Global exploration must keep at least two parameters changed from default.")
        training_changes = sum(config[name] != defaults[name] for name in config
                               if name in TRAINING_PARAMETERS)
        model_changes = sum(config[name] != defaults[name] for name in config
                            if name not in TRAINING_PARAMETERS)
        priority = "model" if model_changes >= training_changes and model_changes else "training"
        if state.get("effective_priority") and priority != state["effective_priority"]:
            raise ValueError("Global edits must preserve the scheduled model or training priority.")
    parameter = state.get("coordinate_parameter")
    if parameter and config[parameter] != state["coordinate_proposal"][parameter]:
        raise ValueError("The scheduled coordinate value must remain unchanged.")
    changed = [key for key in config if config[key] != state["coordinate_proposal"][key]]
    if decision["action"] == "edit_proposal" and not changed:
        raise ValueError("edit_proposal must actually change at least one parameter.")
    if json.dumps(config, sort_keys=True) in seen:
        raise ValueError("The proposed configuration duplicates an executed trial.")
    inspect_candidate(task, config, settings, root, check_forward=True, batch_sizes=(1,))
    return {**decision, "executed_config": config, "changed_parameters": changed}


class Agent:
    def __init__(self, settings, mock=False, api_key=None):
        self._api_key = require_llm(settings, mock, api_key)
        self.settings, self.mock = settings, mock

    def call(self, request, state):
        if self.mock:
            decision = {"action": "accept_proposal", "overrides": {}, "evidence_trials": [],
                        "hypothesis": "This is an offline interface test, not evidence of LLM reasoning or tuning gains."}
            return {"mock": True, "choices": [{"message": {"content": json.dumps(decision, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        from openai import DefaultHttpxClient, OpenAI
        http_client = DefaultHttpxClient(trust_env=False, proxy=None)
        with OpenAI(api_key=self._api_key, base_url=self.settings["agent"]["base_url"],
                    timeout=self.settings["agent"]["timeout_seconds"],
                    max_retries=self.settings["agent"].get("max_retries", 3),
                    http_client=http_client) as client:
            return client.chat.completions.create(**request).model_dump(mode="json")
