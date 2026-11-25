"""Entry point: joint screening, Top-K crossover, local tuning, and export."""

import argparse
import csv
import getpass
import importlib.metadata
import json
import math
import re
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch

from hpo.agent import (DIRECT_TRANSPORT, Agent, AgentAPIError, api_error_details,
                       get_search_state, parse_decision)
from hpo.artifacts import (atomic_bytes, copy_verified, freeze_json, json_text,
                           next_attempt, read_json, run_lock, save_json, update_json)
from hpo.config import (EARLY_STOP_PATIENCE, FEATURES_DIR, MAX_TRAIN_EPOCHS, ROOT, load_settings,
                        require_llm, run_folder, select_tasks, tasks_from_manifest, feature_path)
from hpo.coordinator import Coordinator, GLOBAL_REVIEW_STAGES, record_complete, record_score
from hpo.data import effective_series, series_view, split_bounds, validation_diagnostics
from hpo.evaluation import evaluate_test, read_test_cache
from hpo.models import active_space, default_config, extract_spec, inspect_candidate, instantiate
from hpo.losses import loss_details
from hpo.report import summary_row, write_csv
from hpo.slot_table import backfill_run_slot_tables, write_slot_tables
from hpo.retrain import best_configuration, read_seed_results, retrain_best
from hpo.training import fit
from tuning_features import load_features

PROMPT = ROOT / "tuning_prompt.txt"
EVALUATION_SCHEDULE = "per_dataset_after_all_requested_horizons"
ARTIFACT_FORMAT = "balanced_dynamic_input_window_fixed_loss_dual_seed_top4_default_trial_v12"
SUPPORTED_ARTIFACT_FORMATS = {ARTIFACT_FORMAT, "balanced_dynamic_input_window_dual_seed_top4_default_trial_v11",
                             "balanced_dynamic_dual_seed_top4_default_trial_v10",
                             "joint_crossover_local_top3_default_trial_v9",
                             "joint_crossover_local_top3_trial_v8",
                             "joint_crossover_local_top3_trial_v7",
                             "coordinate_two_round_top3_trial_v6"}


class TuningAgent(Agent):
    """Store requests and responses with each slot and save a readable prompt file."""

    def decide(self, state, task, root, folder, seen, record_path=None):
        folder = Path(folder)
        record_path = Path(record_path) if record_path is not None else folder / "agent.json"
        saved = read_json(record_path) if record_path.exists() else {}
        if "agent" in saved:
            raise ValueError("An LLM record already exists; use a new run name")
        trace = saved.get("agent", {"state": state, "attempts": []})
        feedback = ""
        for number in range(self.settings["agent"]["max_attempts"]):
            attempt = next_attempt(folder)
            user = json_text(state) + ("\nPrevious response error: " + feedback if feedback else "")
            system = PROMPT.read_text(encoding="utf-8")
            request = {"model": "MOCK_NO_API" if self.mock else self.settings["agent"]["model"],
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                       **self.settings["agent"]["request_options"]}
            atomic_bytes(attempt / "prompt.txt", ("[SYSTEM]\n" + system + "\n[USER]\n" + user).encode("utf-8"))
            entry = {"directory": attempt.name, "request": request,
                     "transport": DIRECT_TRANSPORT, "started_unix": time.time()}
            trace["attempts"].append(entry)
            update_json(record_path, agent=trace)
            started = time.perf_counter()
            try:
                raw = json_text(self.call(request, state))
                if self._api_key:
                    raw = raw.replace(self._api_key, "[REDACTED]")
                response = json.loads(raw)
            except Exception as exc:
                details = api_error_details(exc, self._api_key)
                entry.update(api_error=details, elapsed_seconds=time.perf_counter() - started)
                update_json(record_path, agent=trace)
                cause = details["chain"][0]
                explanation = details.get("response_body") or cause["message"]
                raise AgentAPIError(
                    f"LLM API failed ({cause['type']}: {explanation}); request was not completed. "
                    "Preserve this run and diagnose the API response before restarting.") from None
            entry.update(response=response, elapsed_seconds=time.perf_counter() - started)
            save_json(attempt / "response.json", response)
            update_json(record_path, agent=trace)
            finish_reason = None
            try:
                choice = response["choices"][0]
                finish_reason = choice.get("finish_reason")
                content = choice["message"]["content"]
                if finish_reason == "length":
                    raise ValueError("Output was truncated by the provider before a complete decision was returned.")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("The response contained no final JSON answer.")
                decision = parse_decision(content, state, task, self.settings, root, seen)
                decision.update(mock=self.mock, attempts=number + 1, fallback=False)
                trace["decision"] = decision
                save_json(folder / "decision.json", decision)
                update_json(record_path, agent=trace)
                return decision
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                feedback = str(exc)
                if not feedback.isascii():
                    feedback = "Return a complete valid JSON decision with allowed parameter values and an English hypothesis."
                entry["validation_error"] = {"error": feedback, "error_type": type(exc).__name__,
                                             "finish_reason": finish_reason}
                update_json(record_path, agent=trace)
        decision = {"action": "accept_proposal", "overrides": {}, "evidence_trials": [],
                    "hypothesis": "Response validation failed; execute the original coordinate proposal.",
                    "executed_config": state["coordinate_proposal"], "changed_parameters": [],
                    "fallback": True, "fallback_reason": feedback, "mock": self.mock,
                    "attempts": self.settings["agent"]["max_attempts"]}
        trace["decision"] = decision
        save_json(folder / "decision.json", decision)
        update_json(record_path, agent=trace)
        return decision


def task_folder(run, task):
    return Path(run) / "tasks" / task["dataset"] / f"h{task['horizon']}"


def input_tasks(args):
    if bool(args.model) != bool(args.data):
        raise ValueError("Direct input requires both --model and --data")
    if args.model:
        model_path, data_path = Path(args.model).resolve(), Path(args.data).resolve()
        data = series_view(data_path)
        code = model_path.read_text(encoding="utf-8-sig")
        try:
            spec = extract_spec(code)
        except (ValueError, StopIteration, SyntaxError, IndexError, TypeError) as exc:
            raise ValueError("The model must use this project's modular CandidateModel format; a weights-only input is unsupported.") from exc
        tasks = []
        for horizon in args.horizons or [3]:
            task = {"id": f"{data_path.stem}__h{horizon}", "dataset": data_path.stem,
                    "horizon": horizon, "length": len(data), "channels": int(data.shape[1]),
                    "data_path": str(data_path), "model_path": str(model_path), "spec": spec}
            with torch.random.fork_rng(devices=[]):
                model = instantiate(code, task)
                task["original_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if task["original_params"] <= 0:
                raise ValueError("The input model has no trainable parameters")
            tasks.append(task)
        return tasks
    datasets = args.datasets
    if datasets.isdecimal():
        if int(datasets) < 1:
            raise ValueError("--datasets must be a positive integer")
        datasets = "first" + datasets
    return select_tasks(tasks_from_manifest(), datasets, args.horizons or [3, 12])


def load_inputs(tasks, args, settings):
    features = {}
    for task in tasks:
        for field in ("data", "model"):
            if not (ROOT / task[field + "_path"]).is_file():
                raise FileNotFoundError(f"Input {field} is missing: {task['id']}")
        data = effective_series(ROOT / task["data_path"], settings)
        train_end, val_end = split_bounds(len(data), settings)
        if min(train_end, val_end - train_end, len(data) - val_end) < 12 + task["horizon"]:
            raise ValueError(f"Split data is too short to construct windows: {task['id']}")
        path = Path(args.features) if args.features else feature_path(task["dataset"], args.features_dir)
        features[task["id"]] = load_features(path)
        inspect_candidate(task, default_config(task, settings), settings, ROOT, check_forward=True)
    return features


def freeze_run(run, tasks, features, settings, mock):
    if (run / "protocol.json").exists():
        raise ValueError("This directory belongs to the legacy 00-05 workflow; use a new --run name")
    original_path = run / "tuning_protocol.json"
    if original_path.exists():
        raise ValueError("This run already exists; use a new run name")
    protocol = {"method": "fixed-1-16-9-6-balanced-dynamic-input-window-fixed-loss-dual-seed-top4-default-confirmation-agent-v12", "settings": settings, "tasks": tasks,
                "artifact_format": ARTIFACT_FORMAT,
                "evaluation_schedule": EVALUATION_SCHEDULE,
                "features": features, "agent_backend": "mock" if mock else "api",
                "evaluation_scope": "exploratory_provided_features_train_only_unverified",
                "selection_metric": "mean_original_scale_overlap_averaged_validation_RMSE",
                "screening_seeds": settings["search"].get("screening_seeds", [2026]),
                "input_window_choices": settings["space"].get("time_lag", [settings["time_lag"]]),
                "evaluation_target_alignment": "split_start_plus_12_with_past_context_v1" if "time_lag" in settings["space"] else "within_split_fixed_input_window",
                "early_stopping_metric": "normalized_validation_0.2_mse_0.8_mae_loss",
                "fixed_loss": loss_details(),
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "openai")}}
    save_json(original_path, protocol)


def saved_protocol(run):
    return read_json(run / "tuning_protocol.json")


def dataset_groups(tasks):
    groups = {}
    for task in tasks:
        groups.setdefault(task["dataset"], []).append(task)
    return list(groups.values())


def summary_path(run):
    """Reuse the first summary name; timestamp new summaries in local time."""
    paths = sorted(path for path in run.glob("summary_*.csv")
                   if re.fullmatch(r"summary_[0-9]{12}\.csv", path.name))
    if len(paths) > 1:
        raise ValueError(f"Multiple timestamped summaries found; cannot choose a file: {run}")
    if paths:
        return paths[0]
    legacy = run / "summary.csv"
    if legacy.exists():
        return legacy
    return run / time.strftime("summary_%Y%m%d%H%M.csv")


def write_summary_rows(run, rows):
    """Update summary rows while preserving other completed datasets."""
    path = summary_path(run)
    combined = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                combined[(row["dataset"], int(row["horizon"]))] = row
    for row in rows:
        combined[(row["dataset"], int(row["horizon"]))] = row
    write_csv(path, [combined[key] for key in sorted(combined)], list(rows[0]))


def finish_dataset(run, completed, features, settings):
    """Confirm Top K plus defaults, freeze selection, then test the dataset."""
    selected = []
    for task, folder, records in completed:
        confirmation = confirm_top_three(task, settings, folder, records)
        write_slot_tables(folder, records, settings, task)
        selected.append((task, folder, records, *select_best(task, folder, records, confirmation)))
    path = run / "selection_freeze.json"
    frozen = read_json(path) if path.exists() else {"task_ids": []}
    existing = set(frozen.get("task_ids", []))
    additions = {task["id"] for task, folder, *_ in selected}
    if existing & additions:
        raise ValueError("A selection record already exists; use a new run name")
    save_json(path, {"task_ids": sorted(existing | additions)})
    for task, folder, _, _, best in selected:
        freeze_json(folder / "best/config.json", best_configuration(task, settings, folder, best))
    rows = []
    for task, folder, records, baseline, best in selected:
        row = finish_task(task, features[task["id"]], settings, folder, baseline, best, records)
        rows.append(row)
        write_summary_rows(run, [row])
        if row["status"] != "COMPLETE":
            raise RuntimeError(f"Selected-configuration runs are incomplete; three-seed means were not computed. Check {folder / 'best/seed_results.csv'}")
    return rows


def write_trials(folder, records):
    rows = []
    for record in records:
        result, decision = record["result"], record.get("decision") or {}
        rows.append({"slot": record["slot"], "phase": record.get("phase"),
                     "search_round": record.get("search_round"),
                     "search_priority": record.get("search_priority"),
                     "effective_priority": record.get("effective_priority"),
                     "parameter": record.get("parameter"), "probe_value": record.get("probe_value"),
                     "base_slot": record.get("base_slot"), "parent_slots": record.get("parent_slots"),
                     "seed": result.get("seed"),
                     "source": record["execution_source"],
                     "status": result["status"], **(record["executed_config"] or {}),
                     "screening_status": record.get("screening_status", result["status"]),
                     "screening_val_rmse_mean": record.get("screening_val_rmse_mean", result.get("val_rmse")),
                     "screening_val_rmse_std": record.get("screening_val_rmse_std"),
                     **{k: result.get(k) for k in ("val_rmse", "val_loss", "best_epoch", "epochs_run", "params", "elapsed_seconds")},
                     "agent_fallback": decision.get("fallback", False), "hypothesis": decision.get("hypothesis", "")})
    parameter_fields = list(dict.fromkeys(name for row in rows for name in row
                                          if name not in {"slot", "phase", "search_round", "parameter", "probe_value", "base_slot", "parent_slots", "seed",
                                                          "source", "status", "val_rmse", "val_loss",
                                                          "best_epoch", "epochs_run", "params", "elapsed_seconds",
                                                          "agent_fallback", "hypothesis"}))
    fields = ["slot", "phase", "search_round", "parameter", "probe_value", "base_slot", "parent_slots", "seed", "source", "status", *parameter_fields,
              "val_rmse", "val_loss", "best_epoch", "epochs_run", "params", "elapsed_seconds",
              "agent_fallback", "hypothesis"]
    write_csv(folder / "trials.csv", rows, fields)


def screening_entries(record):
    """Keep individual checkpoint metrics distinct from the search ranking mean."""
    if "screening_results" in record:
        return record["screening_results"]
    result = record["result"]
    return {str(result["seed"]): {
        "result": result, "kind": "trial",
        "record": f"trials/trial_{record['slot']:03d}/trial.json",
        "fit_folder": record["fit_folder"]}}


def fit_screening(task, config, settings, folder, trial, proposal):
    seeds = settings["search"].get("screening_seeds", [settings["training"]["search_train_seed"]])
    entries = {}
    for index, seed in enumerate(seeds):
        directory = trial if index == 0 else trial / f"seed_{seed}"
        record_path = directory / ("trial.json" if index == 0 else "record.json")
        result = fit(task, config, settings, ROOT, directory / "training", seed=seed,
                     epochs=settings["training"]["short_epochs"], patience=settings["training"]["short_patience"],
                     purpose=f"coordinate_{proposal['phase']}", record_path=record_path)
        entries[str(seed)] = {"result": result, "kind": "trial" if index == 0 else "screening",
                              "record": str(record_path.relative_to(folder)),
                              "fit_folder": str((directory / "training").relative_to(folder))}
        if result["status"] != "COMPLETE":
            break
        if (result.get("seed") != seed or result.get("config") != config
                or not isinstance(result.get("val_rmse"), (int, float))
                or not math.isfinite(result["val_rmse"]) or result["val_rmse"] < 0):
            raise ValueError(f"Invalid screening result: {task['id']}/seed{seed}")
    complete = len(entries) == len(seeds) and all(e["result"]["status"] == "COMPLETE" for e in entries.values())
    values = [e["result"]["val_rmse"] for e in entries.values()] if complete else []
    return entries, {"screening_status": "COMPLETE" if complete else "FAIL",
                     "screening_val_rmse_mean": statistics.mean(values) if complete else None,
                     "screening_val_rmse_std": statistics.stdev(values) if len(values) > 1 else 0.0 if complete else None}


def tune_task(task, features, settings, folder, agent):
    diagnostics = validation_diagnostics(ROOT / task["data_path"], settings)
    freeze_json(folder / "task.json", {"task": task, "features": features,
                                       "search_space": active_space(task, settings),
                                       "validation_diagnostics": diagnostics})
    copy_verified(features["source"], folder / "input" / "meta_features.csv")
    copy_verified(ROOT / task["model_path"], folder / "input" / "candidate_model.py")
    optimizer, records = Coordinator(task, settings, ROOT, diagnostics["tuning_allowed"]), []
    for slot in range(1, settings["search"]["trials"] + 1):
        proposal = optimizer.propose(slot)
        if proposal is None:
            break
        trial = folder / "trials" / f"trial_{slot:03d}"
        record_path = trial / "trial.json"
        if record_path.exists():
            raise ValueError("A slot record already exists; use a new run name")
        update_json(record_path, proposal=proposal)
        config, decision, source = proposal["config"], None, proposal["source"]
        if proposal.get("review_stage"):
            if agent is None:
                raise RuntimeError("This slot requires LLM review, but the client is not initialized")
            state = get_search_state(task, settings, proposal, records, slot)
            state["task"]["meta_features"] = features["values"]
            state["task"]["meta_feature_note"] = features["note"]
            state["task"]["meta_feature_scope"] = features["scope"]
            decision = agent.decide(state, task, ROOT, trial / "agent", optimizer.seen,
                                    record_path=record_path)
            config = decision["executed_config"]
            source = "agent_fallback_coordinate" if decision["fallback"] else "agent_" + decision["action"]
        save_json(trial / "config.json", {"slot": slot, "phase": proposal["phase"],
                                           "search_round": proposal.get("search_round"),
                                           "proposal": proposal, "executed_config": config,
                                           "execution_source": source, "agent_decision": decision})
        print(f"{task['id']} slot {slot:02d}/{settings['search']['trials']} [{source}]", flush=True)
        entries, screening = fit_screening(task, config, settings, folder, trial, proposal)
        result = entries[str(settings["training"]["search_train_seed"])]["result"]
        record = {"slot": slot, "proposal_config": proposal["config"], "executed_config": config,
                  "proposal_source": proposal["source"], "execution_source": source,
                  "phase": proposal["phase"], "search_round": proposal.get("search_round"),
                  "search_priority": proposal.get("search_priority"),
                  "effective_priority": proposal.get("effective_priority"),
                  "parameter": proposal.get("parameter"),
                  "probe_value": proposal.get("probe_value"), "base_slot": proposal.get("base_slot"),
                  "parent_slots": proposal.get("parent_slots", []),
                  "parent_pool_slots": proposal.get("parent_pool_slots"),
                  "parent_pool_size": proposal.get("parent_pool_size"),
                  "parent_pool_expanded": proposal.get("parent_pool_expanded"),
                  "decision": decision, "result": result, "screening_results": entries, **screening,
                  "fit_folder": str((trial / "training").relative_to(folder))}
        update_json(record_path, record=record)
        records.append(record)
        optimizer.observe(record)
        write_trials(folder, records)
        write_slot_tables(folder, records, settings, task)
        save_json(folder / "coordinate_trace.json", optimizer.trace())
        if slot == 1:
            if not record_complete(record):
                raise RuntimeError(f"Baseline training failed; inspect error.txt for this slot: {task['id']}")
    write_trials(folder, records)
    save_json(folder / "coordinate_trace.json", optimizer.trace())
    return records


def confirm_top_three(task, settings, folder, records):
    """Confirm Top K plus default; keep the historical function/folder names for readers."""
    complete = [record for record in records if record_complete(record)]
    top_k = settings["search"].get("confirmation_top_k", 3)
    top = sorted(complete, key=lambda record: (record_score(record), record["slot"]))[:top_k]
    if not top:
        raise RuntimeError(f"No successful slots are available for Top-{top_k} + default confirmation: {task['id']}")
    default = next((record for record in complete if record["slot"] == 1), None)
    if default is None:
        raise RuntimeError(f"Default slot 1 is unavailable for confirmation: {task['id']}")
    candidates = list(top)
    if all(record["slot"] != default["slot"] for record in candidates):
        candidates.append(default)
    confirmation_path = folder / "top3" / "confirmation.json"
    if confirmation_path.exists():
        raise ValueError(f"A Top-{top_k} + default confirmation record already exists; use a new run name")
    ranking = []
    for record in candidates:
        entries = screening_entries(record)
        expected_screening = settings["search"].get("screening_seeds", [settings["training"]["search_train_seed"]])
        if set(entries) != set(map(str, expected_screening)):
            raise ValueError(f"Screening seeds are incomplete: {task['id']}/slot{record['slot']}")
        values = {seed: entry["result"]["val_rmse"] for seed, entry in entries.items()}
        sources = {seed: {key: entry[key] for key in ("kind", "record", "fit_folder")}
                   for seed, entry in entries.items()}
        for seed in settings["search"]["confirmation_seeds"]:
            if str(seed) in values:
                raise ValueError("Confirmation seeds overlap with completed screening seeds")
            directory = folder / "top3" / f"slot_{record['slot']:03d}" / f"seed_{seed}"
            result = fit(task, record["executed_config"], settings, ROOT, directory / "training", seed=seed,
                         epochs=settings["training"]["short_epochs"], patience=settings["training"]["short_patience"],
                         purpose="top3_validation_confirmation", record_path=directory / "record.json")
            if result["status"] != "COMPLETE" or result.get("seed") != seed:
                raise RuntimeError(f"Top-{top_k} + default confirmation training failed: {task['id']}/slot{record['slot']}/seed{seed}")
            values[str(seed)] = result["val_rmse"]
            sources[str(seed)] = {
                "kind": "confirmation",
                "record": str((directory / "record.json").relative_to(folder)),
                "fit_folder": str((directory / "training").relative_to(folder))}
        ordered = [values[str(seed)] for seed in settings["training"]["final_seeds"]]
        item = {"slot": record["slot"], "config": record["executed_config"],
                "validation_rmse_by_seed": values,
                "validation_rmse_mean": statistics.mean(ordered),
                "validation_rmse_std": statistics.stdev(ordered), "seed_sources": sources}
        ranking.append(item)
    ranking.sort(key=lambda item: (item["validation_rmse_mean"], item["slot"]))
    rows = [{"rank": rank, "slot": item["slot"], "seed": seed,
             "val_rmse": item["validation_rmse_by_seed"][str(seed)],
             "val_rmse_mean": item["validation_rmse_mean"], "val_rmse_std": item["validation_rmse_std"]}
            for rank, item in enumerate(ranking, 1) for seed in settings["training"]["final_seeds"]]
    confirmation = {"task_id": task["id"], "screening_seed": settings["training"]["search_train_seed"],
                    "screening_seeds": settings["search"].get("screening_seeds", [settings["training"]["search_train_seed"]]),
                    "top_k": top_k,
                    "top_slots": [record["slot"] for record in top],
                    "confirmation_slots": [record["slot"] for record in candidates],
                    "default_slot": default["slot"],
                    "seeds": settings["training"]["final_seeds"], "ranking": ranking,
                    "winner_slot": ranking[0]["slot"],
                    "selection_metric": "mean_original_scale_overlap_averaged_validation_RMSE"}
    write_csv(folder / "top3" / "validation_results.csv", rows,
              ["rank", "slot", "seed", "val_rmse", "val_rmse_mean", "val_rmse_std"])
    freeze_json(confirmation_path, confirmation)
    return confirmation


def select_best(task, folder, records, confirmation):
    trace = read_json(folder / "coordinate_trace.json")
    if not trace.get("complete"):
        raise ValueError("Screening search is incomplete; selection is unavailable")
    by_slot = {record["slot"]: record for record in records}
    baseline = by_slot[trace["baseline_slot"]]
    ranked = confirmation["ranking"]
    winner = ranked[0]
    best = dict(by_slot[winner["slot"]])
    best.update(confirmed_val_rmse_mean=winner["validation_rmse_mean"],
                confirmed_val_rmse_std=winner["validation_rmse_std"],
                validation_rmse_by_seed=winner["validation_rmse_by_seed"],
                seed_sources=winner["seed_sources"])
    selection = {"task_id": task["id"], "slot": best["slot"], "config": best["executed_config"],
                 "val_rmse": best["confirmed_val_rmse_mean"],
                 "validation_rmse_by_seed": best["validation_rmse_by_seed"],
                 "validation_rmse_std": best["confirmed_val_rmse_std"],
                 "selection_metric": "mean_original_scale_overlap_averaged_validation_RMSE",
                 "checkpoint_metric": best["result"].get("checkpoint_metric", loss_details(best["executed_config"])["checkpoint_metric"]),
                 "coordinate_trace": "coordinate_trace.json", "top3_confirmation": "top3/confirmation.json",
                 "top3_ranking": ranked, "seed_sources": best["seed_sources"],
                 "test_used_for_selection": False,
                 "source_record": f"trials/trial_{best['slot']:03d}/trial.json"}
    freeze_json(folder / "best" / "selection.json", selection)
    fit_folder, result = folder / best["fit_folder"], best["result"]
    for key, filename in (("candidate_model", "candidate_model.py"), ("checkpoint", "best_model_weights.pth"),
                          ("stats", "train_standardization_stats.npz")):
        copy_verified(fit_folder / result[key], folder / "best" / filename)
    return baseline, best


def finish_task(task, features, settings, folder, baseline, best, records):
    scores = {}
    for label, record in (("default", baseline), ("best", best)):
        if label == "best" and best["slot"] == 1:
            scores[label] = scores["default"]
            continue
        scores[label] = evaluate_test(task, record["executed_config"], settings, ROOT,
                                      folder / record["fit_folder"], record["result"], folder / "test" / label,
                                      settings["training"]["search_train_seed"])
    default_rmse = record_score(baseline)
    best_rmse = record_score(best)
    seed_rows = retrain_best(task, settings, ROOT, folder, best)
    summary = summary_row(task, features, settings, baseline, best, records, scores, seed_rows)
    write_recommendation(task, settings, folder, best, summary, seed_rows)
    print(f"{task['id']} {summary['status']}: selected slot {best['slot']}, screening validation RMSE mean {default_rmse:.6g} -> {best_rmse:.6g}; "
          f"three-seed test RMSE mean={summary['best_test_rmse_mean']}, recommendation={summary['recommendation_source']}", flush=True)
    return summary


def write_recommendation(task, settings, folder, best, summary, seed_rows):
    """Export readable per-task recommendations without listing file hashes."""
    target = folder / "recommended"
    use_default = summary["recommendation_source"] == "L_DEFAULT"
    source = ROOT / task["model_path"] if use_default else folder / "best/candidate_model.py"
    config = default_config(task, settings) if use_default else best["executed_config"]
    copy_verified(source, target / "recommended_model.py")
    selection = read_json(folder / "best/selection.json")
    ranking = selection.get("top3_ranking", [])
    ranking_text = "\n".join(
        f"  {index}. slot {item['slot']}: {item['validation_rmse_mean']:.10g} +/- {item['validation_rmse_std']:.10g}"
        for index, item in enumerate(ranking, 1))
    validation_text = ", ".join(
        f"seed {seed}={value:.10g}" for seed, value in
        ((int(seed), score) for seed, score in selection["validation_rmse_by_seed"].items()))
    test_text = "\n".join(
        f"  seed {row['seed']}: RMSE={row['test_rmse']:.10g}, MAE={row['test_mae']:.10g}"
        for row in seed_rows)
    if use_default:
        reason = "The confirmed winner did not beat historical L on three-seed test RMSE; recommend L defaults."
    elif summary["l_original_test_rmse_mean"] is None:
        reason = "Historical L metrics are unavailable; recommend the configuration with the best three-seed validation mean."
    else:
        reason = "The confirmed winner beat historical L on three-seed test RMSE; recommend its configuration."
    text = (
        f"Recommendation source: {summary['recommendation_source']}\n"
        f"Dataset: {task['dataset']}\nHorizon: {task['horizon']}\n"
        f"Recommended slot: {summary['recommended_slot']}\n"
        f"Recommended parameters:\n{json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True)}\n\n"
        f"Top-{settings['search'].get('confirmation_top_k', 3)} + default validation RMSE ranking across three seeds:\n{ranking_text}\n\n"
        f"Winner validation RMSE by seed: {validation_text}\n"
        f"Winner test metrics by seed:\n{test_text}\n\n"
        f"Confirmed winner test RMSE mean: {summary['best_test_rmse_mean']}\n"
        f"Confirmed winner test MAE mean: {summary['best_test_mae_mean']}\n"
        f"Historical L test RMSE mean: {summary['l_original_test_rmse_mean']}\n"
        f"Historical L test MAE mean: {summary['l_original_test_mae_mean']}\n"
        f"RMSE improvement over historical L: {summary['test_rmse_improvement_vs_l_original']}\n"
        f"MAE improvement over historical L: {summary['test_mae_improvement_vs_l_original']}\n"
        f"Reason: {reason}\n")
    atomic_bytes(target / "recommended_params.txt", text.encode("utf-8-sig"))


def refresh_summary(run):
    """Refresh CSV from frozen selections and caches without training or reselection."""
    protocol = saved_protocol(run)
    if protocol.get("artifact_format") not in SUPPORTED_ARTIFACT_FORMATS:
        raise ValueError("Unsupported run artifact format")
    freeze_path = run / "selection_freeze.json"
    if not freeze_path.exists():
        raise ValueError("No completed selection records are available")
    frozen = read_json(freeze_path)
    tasks, settings = protocol["tasks"], protocol["settings"]
    task_ids = set(frozen.get("task_ids", []))
    if task_ids - {task["id"] for task in tasks}:
        raise ValueError("Frozen selection contains tasks outside the plan")
    for group in dataset_groups(tasks):
        present = [task["id"] in task_ids for task in group]
        if any(present) and not all(present):
            raise ValueError("Selections for this dataset are not fully frozen")
    rows = []
    for task in tasks:
        if task["id"] not in task_ids:
            continue
        folder = task_folder(run, task)
        selection_path = folder / "best/selection.json"
        if not selection_path.is_file():
            raise ValueError(f"Best selection is missing: {task['id']}")
        selection = read_json(selection_path)
        if selection["task_id"] != task["id"] or selection["test_used_for_selection"]:
            raise ValueError("Selection record uses an incompatible protocol")
        records = [read_json(p)["record"] for p in sorted(folder.glob("trials/trial_*/trial.json"))]
        if (not read_json(folder / "coordinate_trace.json").get("complete")
                or [r["slot"] for r in records] != list(range(1, len(records) + 1))):
            raise ValueError(f"Slots are incomplete: {task['id']}")
        by_slot = {record["slot"]: record for record in records}
        baseline, best = records[0], by_slot[selection["slot"]]
        if selection.get("seed_sources"):
            best = dict(best)
            best.update(confirmed_val_rmse_mean=selection["val_rmse"],
                        confirmed_val_rmse_std=selection["validation_rmse_std"],
                        validation_rmse_by_seed=selection["validation_rmse_by_seed"],
                        seed_sources=selection["seed_sources"])
        if not (folder / selection["source_record"]).is_file():
            raise ValueError("Best-slot record is missing")
        test_labels = ("default",) if best["slot"] == 1 else ("default", "best")
        if any(not (folder / "test" / label / "metrics.json").exists() for label in test_labels):
            continue
        scores = {}
        for label, record in (("default", baseline), ("best", best)):
            cached_label = "default" if record["slot"] == 1 else label
            cached = read_test_cache(task, record["executed_config"], settings, record["result"],
                                     folder / "test" / cached_label, settings["training"]["search_train_seed"])
            scores[label] = cached
        seed_rows = read_seed_results(task, settings, folder, best) if settings["training"].get("final_seeds") else None
        rows.append(summary_row(task, protocol["features"][task["id"]], settings, baseline, best, records, scores, seed_rows))
    if not rows:
        raise ValueError("No complete test results are available; the summary was not changed")
    write_csv(summary_path(run), rows, list(rows[0]))
    return rows


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Balanced dual-seed screening + dynamic crossover/local tuning -> Top-4 + default confirmation")
    parser.add_argument("--run", default="fixed_loss_tuning_v12")
    parser.add_argument("--report-only", action="store_true", help="Refresh the summary from a completed run; no API key required")
    parser.add_argument("--tables-only", action="store_true", help="Add missing per-slot hyperparameter tables to an existing run; no training or summary changes")
    parser.add_argument("--model", help="Path to a modular CandidateModel source; omit to use shared L models")
    parser.add_argument("--data", help="Path to the corresponding NPY time series")
    parser.add_argument("--datasets", default="1", help="Shared data: N/firstN, inclusive range 3:10, all, or comma-separated names")
    parser.add_argument("--horizons", "--horizon", type=int, nargs="+", choices=[3, 12])
    parser.add_argument("--features-dir", default=str(FEATURES_DIR))
    parser.add_argument("--features", help="Direct meta_features.csv input for --model/--data mode")
    parser.add_argument("--trials", type=int, default=32, help="Maximum different configurations (1-32), two seeds each; Top-4 + default adds at most 5 fits")
    parser.add_argument("--epochs", type=int, choices=[MAX_TRAIN_EPOCHS], help="Formal runs use at most 300 epochs; not tunable")
    parser.add_argument("--patience", type=int, choices=[EARLY_STOP_PATIENCE], help="Formal runs use patience 15; not tunable")
    parser.add_argument("--api-key", nargs="?", const="", help="Prompt securely when passed without a value; otherwise use DEEPSEEK_API_KEY")
    parser.add_argument("--smoke", action="store_true", help="Run an engineering smoke test on a short training prefix")
    parser.add_argument("--mock-agent", action="store_true", help="Use an offline mock LLM; allowed only with --smoke")
    args = parser.parse_args(argv)
    if args.report_only and args.tables_only:
        parser.error("Use only one of --report-only and --tables-only")
    if args.report_only or args.tables_only:
        run = run_folder(args.run)
        if not (run / "tuning_protocol.json").exists():
            parser.error("Saved protocol for this run was not found")
        with run_lock(run):
            if args.tables_only:
                paths = backfill_run_slot_tables(run, only_missing=True)
                print(f"Added missing slot hyperparameter tables: {len(paths)} files; existing results unchanged.", flush=True)
                return
            rows = refresh_summary(run)
            backfill_run_slot_tables(run, only_missing=True)
        print(f"Refreshed saved test summary without training or retesting: {len(rows)} items; no LLM calls.", flush=True)
        if rows:
            print(f"Summary: {summary_path(run)}", flush=True)
        return
    if not 1 <= args.trials <= 32:
        parser.error("trials must be between 1 and 32")
    if args.mock_agent and not args.smoke:
        parser.error("mock-agent is allowed only for smoke tests")
    if args.smoke and not args.run.startswith("smoke_"):
        parser.error("Smoke run names must start with smoke_")
    if args.smoke and (args.epochs is not None or args.patience is not None):
        parser.error("Smoke tests use one epoch; omit --epochs and --patience")
    if args.features and not args.model:
        parser.error("--features is only for direct model input; use --features-dir for copied datasets")
    settings = load_settings(smoke=args.smoke)
    settings["search"]["trials"] = args.trials
    tasks = input_tasks(args)
    features = load_inputs(tasks, args, settings)
    run = run_folder(args.run)
    print(f"Run contains {len(tasks)} tasks with at most {args.trials} configurations each; allocation is 1 baseline + 16 joint + 9 crossover + 6 local, with equal model/training priorities.", flush=True)
    print("Each configuration is ranked after seeds 2026 and 2027; Top-4 plus default add seed 2028 (at most 69 fits per task).", flush=True)
    print("Input window is tunable over 12, 32, and 64; prediction horizons remain 3/12. Validation and test targets use the same dates for every window.", flush=True)
    print("All configurations use the fixed default loss: 0.2*MSE + 0.8*MAE.", flush=True)
    print("The LLM client uses a direct connection and ignores HTTP_PROXY/HTTPS_PROXY/ALL_PROXY.", flush=True)
    print(f"Each slot uses at most {settings['training']['short_epochs']} epochs with patience {settings['training']['short_patience']}; neither is tunable.", flush=True)
    print("Meta-features come from user-provided CSV files with unverified training-period provenance; results are exploratory.", flush=True)
    with run_lock(run):
        freeze_run(run, tasks, features, settings, args.mock_agent)
        agent = None
        needs_agent = args.trials >= min(GLOBAL_REVIEW_STAGES) + 1 and any(
            validation_diagnostics(ROOT / task["data_path"], settings)["tuning_allowed"]
            for task in tasks)
        if needs_agent:
            key = args.api_key
            if key == "" and not args.mock_agent:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    try:
                        key = getpass.getpass("DeepSeek API key: ")
                    except (getpass.GetPassWarning, EOFError):
                        parser.error("This terminal cannot hide input; use the DEEPSEEK_API_KEY environment variable")
            key = require_llm(settings, args.mock_agent, key)
            agent = TuningAgent(settings, args.mock_agent, api_key=key)
        for group in dataset_groups(tasks):
            completed = []
            for task in group:
                folder = task_folder(run, task)
                records = tune_task(task, features[task["id"]], settings, folder, agent)
                completed.append((task, folder, records))
            finish_dataset(run, completed, features, settings)
        path = summary_path(run)
        if path.exists():
            print(f"Summary: {path}", flush=True)


if __name__ == "__main__":
    main()
