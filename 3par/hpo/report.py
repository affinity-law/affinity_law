"""Summarize selected results; historical L metrics never enter the search."""

import csv
import io
import json
import math
import statistics
from pathlib import Path

from .artifacts import atomic_bytes
from .config import PROJECT_ROOT
from .models import default_config


def write_csv(path, rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8-sig"))


def l_original_reference(task, settings, path=None):
    """Read current L seed rows; explicit legacy aggregate files remain readable."""
    result = {"l_original_status": "unavailable", "l_original_seed_count": None,
              **{f"l_original_test_{metric}_{stat}": None
                 for metric in ("rmse", "mae") for stat in ("mean", "std")}}
    if settings["smoke"]:
        result["l_original_status"] = "smoke_excluded"
        return result
    path = Path(path) if path is not None else PROJECT_ROOT / "2search/compar_result/exp_L.csv"
    if not path.exists():
        return result
    with path.open(encoding="utf-8-sig", newline="") as handle:
        matches = [row for row in csv.DictReader(handle)
                   if row["dataset"] == task["dataset"] and int(row["horizon"]) == task["horizon"]]
    if not matches:
        return result
    current_seed_rows = "seed" in matches[0]
    if current_seed_rows:
        seeds = [int(row["seed"]) for row in matches]
        if sorted(seeds) != [2026, 2027, 2028]:
            raise ValueError(f"Current L baseline must contain exactly three unique seeds: {task['id']}")
        aggregate = {"seed_count": 3, "seeds": "2026;2027;2028"}
        for metric in ("rmse", "mae"):
            values = [float(row[metric]) for row in matches]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"Invalid current L metrics: {task['id']}/{metric}")
            aggregate[f"{metric}_mean"] = statistics.mean(values)
            aggregate[f"{metric}_std"] = statistics.stdev(values)
        matches = [aggregate]
    if len(matches) != 1:
        raise ValueError(f"Duplicate historical L baseline: {task['id']}")
    row = matches[0]
    if (settings["time_lag"], settings["train_ratio"], settings["val_ratio"]) != (12, 0.6, 0.2):
        result["l_original_status"] = "split_mismatch"
        return result
    if int(row["seed_count"]) != 3 or sorted(row["seeds"].split(";")) != ["2026", "2027", "2028"]:
        raise ValueError(f"Historical L baseline does not contain exactly three seeds: {task['id']}")
    for metric in ("rmse", "mae"):
        for stat in ("mean", "std"):
            value = float(row[f"{metric}_{stat}"])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Historical L baseline contains invalid metrics: {task['id']}")
            result[f"l_original_test_{metric}_{stat}"] = value
    result.update(l_original_status="historical_3seed_reference", l_original_seed_count=3)
    if current_seed_rows:
        result["l_original_status"] = "current_L_3seed_reference"
    return result


def summary_row(task, features, settings, baseline, best, records, scores, seed_rows=None):
    default_rmse = baseline["result"]["val_rmse"]
    best_seed_rmse = best["result"]["val_rmse"]
    best_rmse = best.get("confirmed_val_rmse_mean", best_seed_rmse)
    decisions = [r["decision"] for r in records if r.get("decision")]
    reference = l_original_reference(task, settings)
    summary = {"dataset": task["dataset"], "horizon": task["horizon"], "status": "COMPLETE", "slots": len(records),
               "best_slot": best["slot"], **reference,
               "best_test_rmse": scores["best"]["overall_metrics"]["rmse"],
               "best_test_mae": scores["best"]["overall_metrics"]["mae"],
               "best_seed": settings["training"]["search_train_seed"],
               "max_epochs": settings["training"]["short_epochs"],
               "early_stop_patience": settings["training"]["short_patience"]}
    seeds = settings["training"].get("final_seeds")
    summary["test_comparison"] = "best_3seed_mean_vs_l_original" if seeds else "search_best_single_seed_vs_l_original"
    if seeds:
        seed_rows = seed_rows or []
        if seed_rows and [row["seed"] for row in seed_rows] != seeds:
            raise ValueError("Retraining seeds or order do not match the frozen plan")
        complete = [row for row in seed_rows if row["status"] == "COMPLETE"]
        summary.update(best_seed_count=len(complete), best_seeds=";".join(map(str, seeds)))
        summary["status"] = ("COMPLETE" if len(complete) == len(seeds) else
                             "RETRAIN_FAILED" if any(row["status"] == "FAILED" for row in seed_rows) else "PENDING_RETRAIN")
        for metric in ("rmse", "mae", "wmape"):
            values = [row[f"test_{metric}"] for row in complete]
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
                raise ValueError("Retraining results contain invalid metrics")
            summary[f"best_test_{metric}_mean"] = statistics.mean(values) if summary["status"] == "COMPLETE" else "N/A"
            summary[f"best_test_{metric}_std"] = statistics.stdev(values) if summary["status"] == "COMPLETE" else "N/A"
    for metric in ("rmse", "mae"):
        original = reference[f"l_original_test_{metric}_mean"]
        comparison = summary[f"best_test_{metric}_mean"] if seeds else summary[f"best_test_{metric}"]
        summary[f"test_{metric}_improvement_vs_l_original"] = (
            (original - comparison) / original if original is not None and original > 1e-8 and comparison != "N/A" else "N/A")
    summary.update(
        search_default_test_rmse=scores["default"]["overall_metrics"]["rmse"],
        search_default_test_mae=scores["default"]["overall_metrics"]["mae"],
        search_default_val_rmse=default_rmse, search_best_seed2026_val_rmse=best_seed_rmse,
        best_val_rmse=best_rmse, best_val_rmse_std=best.get("confirmed_val_rmse_std", "N/A"),
        best_val_rmse_by_seed=json.dumps(best.get("validation_rmse_by_seed", {}), sort_keys=True),
        val_improvement_ratio=(default_rmse - best_seed_rmse) / default_rmse if default_rmse > 1e-8 else "N/A",
        best_hyperparameters=json.dumps(best["executed_config"], ensure_ascii=False, sort_keys=True),
        params=best["result"]["params"],
        agent_decisions=len(decisions), agent_edits=sum(d["action"] == "edit_proposal" for d in decisions),
        agent_fallbacks=sum(d["fallback"] for d in decisions), feature_scope=features["scope"], smoke=settings["smoke"])
    if settings["search"].get("screening_seeds"):
        default_mean = baseline.get("screening_val_rmse_mean", default_rmse)
        best_mean = best.get("screening_val_rmse_mean", best_seed_rmse)
        summary.update(screening_seeds=";".join(map(str, settings["search"]["screening_seeds"])),
                       search_default_screening_val_rmse=default_mean,
                       search_best_screening_val_rmse=best_mean,
                       screening_val_improvement_ratio=(default_mean - best_mean) / default_mean if default_mean > 1e-8 else "N/A",
                       confirmation_top_k=settings["search"].get("confirmation_top_k", 3))
    original = summary["l_original_test_rmse_mean"]
    tuned = summary.get("best_test_rmse_mean")
    if summary["status"] != "COMPLETE":
        source = "PENDING"
    elif original is not None and isinstance(tuned, (int, float)) and tuned >= original:
        source = "L_DEFAULT"
    else:
        source = "TUNED_TOP3"
    recommended = default_config(task, settings) if source == "L_DEFAULT" else best["executed_config"]
    summary.update(recommendation_source=source,
                   recommended_slot="L_DEFAULT" if source == "L_DEFAULT" else best["slot"],
                   recommended_hyperparameters=json.dumps(recommended, ensure_ascii=False, sort_keys=True),
                   test_used_for_recommendation=True)
    if "time_lag" in settings["space"]:
        summary.update(default_time_lag=settings["time_lag"],
                       selected_time_lag=best["executed_config"]["time_lag"],
                       recommended_time_lag=recommended["time_lag"],
                       input_window_choices=";".join(map(str, settings["space"]["time_lag"])),
                       evaluation_target_start_offset=settings["time_lag"])
    return summary
