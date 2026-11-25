"""Readable per-slot hyperparameters, using each run's frozen evidence only."""

import csv
import io
import json
import math
import statistics
from pathlib import Path

from .artifacts import atomic_bytes, read_json


TABLE_NAME = "slot_hyperparameters.csv"
IDENTITY_FIELDS = [
    "dataset", "horizon", "slot", "stage", "search_priority", "effective_priority", "proposal_source",
    "execution_source", "base_slot", "parent_slots", "parent_pool_slots", "parent_pool_size",
    "parent_pool_expanded", "parameter", "probe_value",
    "status", "screening_seeds", "screening_seed_count", "screening_val_rmse_mean",
    "screening_val_rmse_std", "params", "param_ratio",
    "llm_action", "llm_fallback", "llm_changed_parameters", "llm_changes",
    "llm_evidence_slots", "llm_hypothesis", "changes_from_default", "changes_from_base",
    "time_lag", "max_epochs", "early_stop_patience", "train_ratio", "val_ratio",
    "checkpoint_metric", "loss_profile_source",
]
PARAMETER_FIRST = ["lr", "weight_decay", "batch_size", "loss_profile", "mse_weight", "mae_weight"]


def _text(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _changes(before, after):
    return "; ".join(f"{key}: {_text(before.get(key))} -> {_text(after.get(key))}"
                     for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key))


def _loss_config(config, settings, result):
    """Never insert a new-run default into a historical configuration."""
    output = dict(config)
    training = settings.get("training", {})
    details = result.get("loss", {}) or result.get("loss_details", {}) or {}
    weights = (result.get("mse_weight", details.get("mse_weight")),
               result.get("mae_weight", details.get("mae_weight")))
    profile = output.get("loss_profile")
    if profile is None:
        weights = (training.get("mse_weight"), training.get("mae_weight"))
        known = {(0.2, 0.8): "mae_heavy", (0.5, 0.5): "balanced", (1.0, 0.0): "mse"}
        profile = known.get(weights) or (f"legacy_{weights[0]:g}_mse_{weights[1]:g}_mae"
                   if all(isinstance(value, (int, float)) for value in weights) else "legacy_unknown")
        source = "frozen_training_weights"
    else:
        source = "executed_config"
        if any(value is None for value in weights):
            # The names identify immutable choices; historical configurations above
            # are deliberately kept out of this mapping.
            from .losses import loss_details
            details = loss_details(config, settings)
            weights = (details["mse_weight"], details["mae_weight"])
    output.update(loss_profile=profile, mse_weight=weights[0], mae_weight=weights[1])
    return output, source


def _seed_results(record, settings):
    results = record.get("screening_results")
    if isinstance(results, dict):
        return {int(seed): value.get("result", value) for seed, value in results.items()}
    if isinstance(results, list):
        return {int(value.get("seed", value.get("result", {}).get("seed"))): value.get("result", value)
                for value in results}
    result = record.get("result", {})
    seed = result.get("seed", settings.get("training", {}).get("search_train_seed", 2026))
    return {int(seed): result} if result else {}


def slot_rows(records, settings, task):
    """Expand executed and original proposed parameters into individual columns."""
    entries = [(item.get("record", item), item.get("proposal", {})) for item in records]
    entries.sort(key=lambda item: item[0]["slot"])
    by_slot = {record["slot"]: record for record, _ in entries}
    default_config = by_slot.get(1, {}).get("executed_config") or {}
    rows = []
    for record, proposal in entries:
        config = record.get("executed_config") or {}
        original = record.get("proposal_config") or proposal.get("config") or {}
        result = record.get("result") or {}
        seeds = _seed_results(record, settings)
        first_result = next(iter(seeds.values()), result)
        effective, loss_source = _loss_config(config, settings, first_result) if config else ({}, "")
        proposed, _ = _loss_config(original, settings, {}) if original else ({}, "")
        decision = record.get("decision") or {}
        values = [item.get("val_rmse") for item in seeds.values()]
        completed = bool(values) and all(item.get("status") == "COMPLETE" for item in seeds.values())
        valid = completed and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
        training = settings.get("training", {})
        base_slot = record.get("base_slot", proposal.get("base_slot"))
        base = by_slot.get(base_slot, {}).get("executed_config") or {}
        row = {
            "dataset": task["dataset"], "horizon": task["horizon"], "slot": record["slot"],
            "stage": record.get("search_round") or record.get("phase", ""),
            "search_priority": record.get("search_priority", proposal.get("search_priority", "unrecorded")),
            "effective_priority": record.get("effective_priority", proposal.get("effective_priority", "unrecorded")),
            "proposal_source": record.get("proposal_source", proposal.get("source")),
            "execution_source": record.get("execution_source"), "base_slot": base_slot,
            "parent_slots": ";".join(map(str, record.get("parent_slots", proposal.get("parent_slots", [])))),
            "parent_pool_slots": ";".join(map(str, record.get("parent_pool_slots", proposal.get("parent_pool_slots")) or [])),
            "parent_pool_size": record.get("parent_pool_size", proposal.get("parent_pool_size")),
            "parent_pool_expanded": record.get("parent_pool_expanded", proposal.get("parent_pool_expanded")),
            "parameter": record.get("parameter"), "probe_value": record.get("probe_value"),
            "status": record.get("screening_status", result.get("status", "PENDING")),
            "screening_seeds": ";".join(map(str, sorted(seeds))), "screening_seed_count": len(seeds),
            "screening_val_rmse_mean": statistics.mean(values) if valid else None,
            "screening_val_rmse_std": statistics.stdev(values) if valid and len(values) > 1 else None,
            "params": result.get("params", first_result.get("params")),
            "param_ratio": result.get("param_ratio", first_result.get("param_ratio")),
            "llm_action": decision.get("action", "not_reviewed"), "llm_fallback": decision.get("fallback"),
            "llm_changed_parameters": ";".join(sorted(key for key in set(config) | set(original)
                                                     if config.get(key) != original.get(key))),
            "llm_changes": _changes(original, config),
            "llm_evidence_slots": ";".join(map(str, decision.get("evidence_trials", []))),
            "llm_hypothesis": decision.get("hypothesis"),
            "changes_from_default": _changes(default_config, config),
            "changes_from_base": _changes(base, config) if base else "",
            "time_lag": settings.get("time_lag"), "max_epochs": training.get("short_epochs"),
            "early_stop_patience": training.get("short_patience"), "train_ratio": settings.get("train_ratio"),
            "val_ratio": settings.get("val_ratio"),
            "checkpoint_metric": first_result.get("checkpoint_metric"), "loss_profile_source": loss_source,
            **effective, **{f"proposal_{key}": value for key, value in proposed.items()},
        }
        for seed, item in sorted(seeds.items()):
            for field in ("status", "val_rmse", "best_epoch", "epochs_run"):
                row[f"seed_{seed}_{field}"] = item.get(field)
        rows.append(row)
    return rows


def _write_table(path, rows):
    keys = {key for row in rows for key in row}
    parameters = [key for key in PARAMETER_FIRST if key in keys]
    parameters += sorted(keys - set(IDENTITY_FIELDS) - set(parameters)
                         - {key for key in keys if key.startswith(("proposal_", "seed_"))})
    fields = IDENTITY_FIELDS + parameters
    fields += [f"proposal_{key}" for key in parameters if f"proposal_{key}" in keys]
    fields += sorted(key for key in keys if key.startswith("proposal_") and key not in fields)
    fields += sorted(key for key in keys if key.startswith("seed_"))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8-sig"))


def _merge_dataset_table(folder, only_missing):
    destination = folder / TABLE_NAME
    if only_missing and destination.exists():
        return []
    rows = []
    for table in sorted(folder.glob(f"h*/{TABLE_NAME}")):
        with table.open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        return []
    rows.sort(key=lambda row: (int(row["horizon"]), int(row["slot"])))
    _write_table(destination, rows)
    return [destination]


def write_slot_tables(folder, records, settings, task, *, only_missing=False):
    """Write a task table and its dataset's combined table; existing trial CSVs are untouched."""
    folder = Path(folder)
    destination = folder / TABLE_NAME
    written = []
    if not only_missing or not destination.exists():
        _write_table(destination, slot_rows(records, settings, task))
        written.append(destination)
    written.extend(_merge_dataset_table(folder.parent, only_missing))
    return written


def backfill_run_slot_tables(run, *, only_missing=True):
    """Generate tables from frozen trial records, without training or editing old evidence."""
    run = Path(run)
    protocol = read_json(run / "tuning_protocol.json")
    settings = protocol["settings"]
    written, dataset_folders = [], set()
    for task in protocol["tasks"]:
        folder = run / "tasks" / task["dataset"] / f"h{task['horizon']}"
        records = [value for path in sorted(folder.glob("trials/trial_*/trial.json"))
                   if "record" in (value := read_json(path))]
        if not records:
            continue
        dataset_folders.add(folder.parent)
        destination = folder / TABLE_NAME
        if not only_missing or not destination.exists():
            _write_table(destination, slot_rows(records, settings, task))
            written.append(destination)
    for folder in sorted(dataset_folders):
        written.extend(_merge_dataset_table(folder, only_missing))
    return written
