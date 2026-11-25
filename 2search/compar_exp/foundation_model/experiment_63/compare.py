"""Compare completed context-64 foundation runs with validation-selected tuned L."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import common
from portable_paths import RESULT_ROOT, L_SEEDS, load_manifest, l_run_paths, index_tuned_l

MODELS = ("timesfm3_0", "moirai2_0")


def model_result(model_name, run_name, task, settings):
    folder = MODEL_ROOT / model_name / "outputs" / run_name / "tasks" / task["dataset"] / f"h{task['horizon']}"
    required = [folder / "metrics.json", folder / "averaged_predictions.npy",
                folder / "averaged_targets.npy", folder / "target_time_indices.npy"]
    if not all(path.is_file() for path in required):
        return None
    record = common.read_json(required[0])
    if (record.get("status") != "COMPLETE"
            or record.get("task_id") != task["id"]
            or record.get("context_length") != settings["context_length"]
            or record.get("target_start_offset") != settings["target_start_offset"]
            or record.get("target_alignment") != settings["target_alignment"]):
        raise ValueError(f"Invalid foundation result metadata: {folder}")
    return {
        "folder": folder, "record": record,
        "prediction": np.load(required[1]), "target": np.load(required[2]),
        "indices": np.load(required[3]),
    }


def l_result(entry, task, seeds):
    config = entry["config"]
    if config.get("task_id") != task["id"]:
        raise ValueError(f"L config/task mismatch: {entry['folder']}")
    time_lag = config.get("time_lag", config.get("hyperparameters", {}).get("time_lag"))
    if time_lag not in (12, 32, 64):
        raise ValueError(f"L selected an unsupported time_lag for {task['id']}: {time_lag}")
    predictions, targets, records = [], [], []
    for seed, prediction_path, target_path, metric_path in zip(
            seeds, entry["prediction_paths"], entry["target_paths"], entry["metric_paths"]):
        prediction, target = np.load(prediction_path), np.load(target_path)
        record = common.read_json(metric_path)
        if (record.get("status") != "COMPLETE" or record.get("seed") != seed
                or record.get("task_id") != task["id"]):
            raise ValueError(f"Invalid tuned L result: {metric_path}")
        if (record.get("time_lag") != time_lag
                or record.get("target_start_offset") != 12
                or record.get("target_alignment") != "split_start_plus_12_with_past_context_v1"):
            raise ValueError(f"Tuned L result uses a different target protocol: {metric_path}")
        predictions.append(prediction)
        targets.append(target)
        records.append(record)
    return time_lag, predictions, targets, records


def binomial_p(wins, total):
    if not total:
        return 1.0
    return float(sum(math.comb(total, k) for k in range(wins, total + 1)) / (2 ** total))


def outcome(foundation, l_value):
    if np.isclose(foundation, l_value, rtol=1e-9, atol=0.0):
        return "draw"
    return "foundation" if foundation < l_value else "L"


def summary_rows(rows):
    output = []
    scopes = [("overall", lambda row: True),
              ("h3", lambda row: row["horizon"] == 3),
              ("h12", lambda row: row["horizon"] == 12),
              ("uni", lambda row: row["data_type"] == "uni"),
              ("multi", lambda row: row["data_type"] == "multi")]
    for model in MODELS:
        for scope, predicate in scopes:
            selected = [row for row in rows if predicate(row)]
            results = [row[f"{model}_vs_L"] for row in selected]
            wins = results.count("foundation")
            draws = results.count("draw")
            losses = results.count("L")
            effective = wins + losses
            relative = [row[f"{model}_rmse_improvement_vs_L"] for row in selected]
            output.append({
                "model": model, "scope": scope, "pairs": len(selected),
                "wins": wins, "draws": draws, "losses": losses,
                "win_rate_excluding_draws": wins / effective if effective else None,
                "one_sided_sign_test_p": binomial_p(wins, effective),
                "mean_rmse_improvement_vs_L": float(np.mean(relative)) if relative else None,
                "median_rmse_improvement_vs_L": float(np.median(relative)) if relative else None,
            })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-run", default="r_aligned_context64_v1")
    parser.add_argument("--l-run", action="append",
                        help="3par output run name/path; repeat for shards. Default scans all runs.")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--horizons", nargs="+", type=int, choices=(3, 12), default=[3, 12])
    parser.add_argument("--output", default="context64_vs_tuned_L_v1")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.foundation_run):
        parser.error("Invalid foundation run name")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output):
        parser.error("Invalid output name")
    settings = common.load_settings()
    _, all_tasks = load_manifest()
    tasks = common.select_tasks(all_tasks, args.datasets, args.horizons)
    l_paths = l_run_paths(args.l_run)
    l_index, duplicates = index_tuned_l(tasks, l_paths, L_SEEDS)
    if duplicates:
        details = "; ".join(f"{task}: {runs}" for task, runs in sorted(duplicates.items()))
        raise ValueError(f"Duplicate complete L tasks across runs; pass explicit --l-run values: {details}")
    rows, missing = [], {"timesfm3_0": [], "moirai2_0": [], "L": []}
    for task in tasks:
        foundations = {model: model_result(model, args.foundation_run, task, settings)
                       for model in MODELS}
        for model, result in foundations.items():
            if result is None:
                missing[model].append(task["id"])
        if task["id"] not in l_index:
            missing["L"].append(task["id"])
        if any(result is None for result in foundations.values()) or task["id"] not in l_index:
            continue
        times_target = foundations["timesfm3_0"]["target"]
        moirai_target = foundations["moirai2_0"]["target"]
        np.testing.assert_array_equal(
            foundations["timesfm3_0"]["indices"], foundations["moirai2_0"]["indices"])
        np.testing.assert_allclose(times_target, moirai_target, rtol=4e-6, atol=4e-6)
        l_lag, l_predictions, l_targets, l_records = l_result(
            l_index[task["id"]], task, L_SEEDS)
        l_metrics = []
        for prediction, target in zip(l_predictions, l_targets):
            if prediction.shape != times_target.shape:
                raise ValueError(f"L/foundation output-shape mismatch: {task['id']}")
            np.testing.assert_allclose(target, times_target, rtol=4e-6, atol=4e-6)
            l_metrics.append(common.metrics(prediction, times_target))
        l_rmse = np.array([metrics["rmse"] for metrics in l_metrics])
        l_mae = np.array([metrics["mae"] for metrics in l_metrics])
        l_wmape = np.array([metrics["wmape"] for metrics in l_metrics])
        for model, result in foundations.items():
            if result["prediction"].shape != times_target.shape:
                raise ValueError(f"{model} prediction/target mismatch: {task['id']}")
        model_metrics = {
            model: common.metrics(result["prediction"], times_target)
            for model, result in foundations.items()
        }
        row = {
            "dataset": task["dataset"], "data_type": common.data_type(task["channels"]),
            "station_num": task["channels"], "horizon": task["horizon"],
            "target_start_offset": settings["target_start_offset"],
            "target_points": len(times_target), "L_selected_time_lag": l_lag,
            "L_seed_count": len(l_rmse), "L_rmse_mean": float(l_rmse.mean()),
            "L_rmse_std": float(l_rmse.std(ddof=1)), "L_mae_mean": float(l_mae.mean()),
            "L_mae_std": float(l_mae.std(ddof=1)), "L_wmape_mean": float(l_wmape.mean()),
            "L_params_mean": float(np.mean([record.get("params", np.nan) for record in l_records])),
            "L_source_run": l_index[task["id"]]["run"].name,
        }
        for model in MODELS:
            values = model_metrics[model]
            row.update({
                f"{model}_context_length": settings["context_length"],
                f"{model}_rmse": values["rmse"], f"{model}_mae": values["mae"],
                f"{model}_wmape": values["wmape"],
                f"{model}_params": foundations[model]["record"]["params"],
                f"{model}_rmse_improvement_vs_L": 1 - values["rmse"] / l_rmse.mean(),
                f"{model}_mae_improvement_vs_L": 1 - values["mae"] / l_mae.mean(),
                f"{model}_vs_L": outcome(values["rmse"], l_rmse.mean()),
            })
        candidates = {"L": row["L_rmse_mean"],
                      "timesfm3_0": row["timesfm3_0_rmse"],
                      "moirai2_0": row["moirai2_0_rmse"]}
        row["rmse_winner"] = min(candidates, key=candidates.get)
        rows.append(row)
    missing_counts = {name: len(values) for name, values in missing.items()}
    if any(missing_counts.values()) and not args.allow_partial:
        print(json.dumps({"status": "INCOMPLETE", "missing_counts": missing_counts,
                          "missing_tasks": missing}, ensure_ascii=False, indent=2))
        raise RuntimeError("Results are incomplete; finish runs or add --allow-partial")
    if not rows:
        raise RuntimeError("No complete L/TimesFM/Moirai task triplets are available")
    output = RESULT_ROOT / "foundation_comparison" / args.output
    if output.exists():
        raise FileExistsError(f"Use a new --output name: {output}")
    summaries = summary_rows(rows)
    common.write_csv(output / "comparison_pairs.csv", rows)
    common.write_csv(output / "comparison_summary.csv", summaries)
    common.write_json(output / "comparison_summary.json", {
        "status": "COMPLETE" if not any(missing_counts.values()) else "PARTIAL",
        "foundation_run": args.foundation_run,
        "l_selection": "validation_selected_best_only",
        "l_runs": [str(path) for path in l_paths],
        "target_alignment": settings["target_alignment"],
        "target_start_offset": settings["target_start_offset"],
        "foundation_context_length": settings["context_length"],
        "l_seed_count": len(L_SEEDS),
        "fake_foundation_seed_repetition": False,
        "pair_count": len(rows), "expected_pair_count": len(tasks),
        "missing_counts": missing_counts, "missing_tasks": missing,
        "summary": summaries,
    })
    print(f"Compared {len(rows)}/{len(tasks)} task triplets")
    for item in summaries:
        if item["scope"] == "overall":
            print(f"{item['model']} vs L: win/draw/loss = "
                  f"{item['wins']}/{item['draws']}/{item['losses']}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
