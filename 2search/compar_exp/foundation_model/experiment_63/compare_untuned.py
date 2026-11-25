"""Compare current L (fixed training hyperparameters) with TimesFM 3.0 and Moirai 2.0 on aligned targets."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common
from compare import MODELS, binomial_p, model_result
from portable_paths import RESULT_ROOT, L_MODEL_ROOT, L_SEEDS, load_manifest, load_l_reference

L_MODEL = "L"
ALL_MODELS = (L_MODEL,) + MODELS
METRICS = ("rmse", "mae", "wmape")
COMPARISONS = (
    ("timesfm3_0", L_MODEL),
    ("moirai2_0", L_MODEL),
    ("timesfm3_0", "moirai2_0"),
)
SCOPES = (
    ("overall", lambda row: True),
    ("h3", lambda row: row["horizon"] == 3),
    ("h12", lambda row: row["horizon"] == 12),
    ("uni", lambda row: row["data_type"] == "uni"),
    ("multi", lambda row: row["data_type"] == "multi"),
)


def current_l_result(task, seeds):
    task_folder = L_MODEL_ROOT / task["dataset"] / f"h{task['horizon']}"
    predictions, targets, records = [], [], []
    for seed in seeds:
        folder = task_folder / f"law_s{seed}" / "test"
        prediction_path = folder / "averaged_final_predictions.npy"
        target_path = folder / "averaged_final_targets.npy"
        metric_path = folder / "test_metrics.json"
        if not all(path.is_file() for path in (prediction_path, target_path, metric_path)):
            return None
        record = common.read_json(metric_path)
        metadata = record.get("task", {})
        evaluation = record.get("evaluation_space", {})
        if (metadata.get("time_lag") != 12
                or metadata.get("pre_len") != task["horizon"]
                or metadata.get("station_num") != task["channels"]
                or metadata.get("evaluation_data_space") != "original"
                or evaluation.get("data_space") != "original"):
            raise ValueError(f"Current L result uses a different protocol: {metric_path}")
        predictions.append(np.load(prediction_path))
        targets.append(np.load(target_path))
        records.append(record)
    return {"folder": task_folder, "predictions": predictions,
            "targets": targets, "records": records}


def validate_l_reference(task, reference, seeds, l_metrics):
    if tuple(reference["seeds"]) != tuple(seeds) or reference["seed_count"] != len(seeds):
        raise ValueError(f"L seed mismatch: {task['id']}")
    for seed, computed in zip(seeds, l_metrics):
        for metric in METRICS:
            if not np.isclose(computed[metric], float(reference["seed_rows"][seed][metric]), rtol=1e-6, atol=1e-8):
                raise ValueError(f"Current exp_L.csv differs from saved predictions: {task['id']}/seed{seed}/{metric}")
    checks = {
        "rmse_mean": np.mean([item["rmse"] for item in l_metrics]),
        "rmse_std": np.std([item["rmse"] for item in l_metrics], ddof=1),
        "mae_mean": np.mean([item["mae"] for item in l_metrics]),
        "mae_std": np.std([item["mae"] for item in l_metrics], ddof=1),
    }
    for name, computed in checks.items():
        if not np.isclose(computed, float(reference[name]), rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"Current L reference {name} does not match saved arrays: {task['id']}")


def metric_value(row, model, metric):
    suffix = f"{metric}_mean" if model == L_MODEL else metric
    return row[f"{model}_{suffix}"]


def outcome(left_model, left_value, right_model, right_value):
    if np.isclose(left_value, right_value, rtol=1e-9, atol=0.0):
        return "draw"
    return left_model if left_value < right_value else right_model


def relative_improvement(left_value, right_value):
    if np.isclose(right_value, 0.0, rtol=0.0, atol=1e-15):
        return None
    return 1 - left_value / right_value


def best_models(values):
    best = min(values.values())
    winners = [name for name, value in values.items()
               if np.isclose(value, best, rtol=1e-9, atol=0.0)]
    return ";".join(winners)


def pairwise_summary(rows):
    output = []
    for metric in METRICS:
        for left_model, right_model in COMPARISONS:
            for scope, predicate in SCOPES:
                selected = [row for row in rows if predicate(row)]
                outcomes, improvements = [], []
                for row in selected:
                    left = metric_value(row, left_model, metric)
                    right = metric_value(row, right_model, metric)
                    outcomes.append(outcome(left_model, left, right_model, right))
                    improvement = relative_improvement(left, right)
                    if improvement is not None:
                        improvements.append(improvement)
                left_wins = outcomes.count(left_model)
                draws = outcomes.count("draw")
                right_wins = outcomes.count(right_model)
                effective = left_wins + right_wins
                output.append({
                    "metric": metric, "comparison": f"{left_model}_vs_{right_model}",
                    "scope": scope, "pairs": len(selected), "left_model": left_model,
                    "right_model": right_model, "left_wins": left_wins,
                    "draws": draws, "right_wins": right_wins,
                    "left_win_rate_excluding_draws": (
                        left_wins / effective if effective else None),
                    "one_sided_sign_test_p": binomial_p(left_wins, effective),
                    "mean_relative_improvement_of_left": (
                        float(np.mean(improvements)) if improvements else None),
                    "median_relative_improvement_of_left": (
                        float(np.median(improvements)) if improvements else None),
                })
    return output


def three_way_summary(rows):
    output = []
    for metric in METRICS:
        for scope, predicate in SCOPES:
            selected = [row for row in rows if predicate(row)]
            winners = [row[f"{metric}_winner"] for row in selected]
            output.append({
                "metric": metric, "scope": scope, "pairs": len(selected),
                "L_wins": winners.count(L_MODEL),
                "timesfm3_0_wins": winners.count("timesfm3_0"),
                "moirai2_0_wins": winners.count("moirai2_0"),
                "tied_best": sum(";" in winner for winner in winners),
            })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-run", default="r_aligned_context64_v1")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--horizons", nargs="+", type=int, choices=(3, 12), default=[3, 12])
    parser.add_argument("--output", default="context64_vs_L_current")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.foundation_run):
        parser.error("Invalid foundation run name")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output):
        parser.error("Invalid output name")

    settings = common.load_settings()
    _, all_tasks = load_manifest()
    tasks = common.select_tasks(all_tasks, args.datasets, args.horizons)
    reference_path, l_reference = load_l_reference()
    seeds = L_SEEDS
    rows, missing = [], {"timesfm3_0": [], "moirai2_0": [], L_MODEL: []}

    for task in tasks:
        foundations = {model: model_result(model, args.foundation_run, task, settings)
                       for model in MODELS}
        for model, result in foundations.items():
            if result is None:
                missing[model].append(task["id"])
        reference = l_reference.get(task["id"])
        l_result = current_l_result(task, seeds) if reference is not None else None
        if l_result is None:
            missing[L_MODEL].append(task["id"])
        if any(result is None for result in foundations.values()) or l_result is None:
            continue

        target = foundations["timesfm3_0"]["target"]
        np.testing.assert_array_equal(
            foundations["timesfm3_0"]["indices"], foundations["moirai2_0"]["indices"])
        np.testing.assert_allclose(
            target, foundations["moirai2_0"]["target"], rtol=4e-6, atol=4e-6)
        for l_target in l_result["targets"]:
            if l_target.shape != target.shape:
                raise ValueError(f"Current L/foundation target-shape mismatch: {task['id']}")
            np.testing.assert_allclose(l_target, target, rtol=4e-6, atol=4e-6)
        for model, result in foundations.items():
            if result["prediction"].shape != target.shape:
                raise ValueError(f"{model} prediction/target mismatch: {task['id']}")
        for prediction in l_result["predictions"]:
            if prediction.shape != target.shape:
                raise ValueError(f"Current L prediction/target mismatch: {task['id']}")

        l_metrics = [common.metrics(prediction, own_target)
                     for prediction, own_target in zip(l_result["predictions"], l_result["targets"])]
        validate_l_reference(task, reference, seeds, l_metrics)
        foundation_metrics = {
            model: common.metrics(result["prediction"], result["target"])
            for model, result in foundations.items()
        }
        row = {
            "dataset": task["dataset"], "data_type": common.data_type(task["channels"]),
            "station_num": task["channels"], "horizon": task["horizon"],
            "target_start_offset": settings["target_start_offset"],
            "target_points": target.shape[0], "L_context_length": 12,
            "L_seed_count": len(seeds),
            "L_params": int(float(reference["params"])),
        }
        for metric in METRICS:
            values = np.array([item[metric] for item in l_metrics])
            row[f"L_{metric}_mean"] = float(values.mean())
            row[f"L_{metric}_std"] = float(values.std(ddof=1))
        for model in MODELS:
            row[f"{model}_context_length"] = settings["context_length"]
            row[f"{model}_params"] = foundations[model]["record"]["params"]
            for metric in METRICS:
                row[f"{model}_{metric}"] = foundation_metrics[model][metric]
        for metric in METRICS:
            values = {model: metric_value(row, model, metric) for model in ALL_MODELS}
            row[f"{metric}_winner"] = best_models(values)
            for left_model, right_model in COMPARISONS:
                left, right = values[left_model], values[right_model]
                name = f"{left_model}_vs_{right_model}_{metric}"
                row[f"{name}_winner"] = outcome(left_model, left, right_model, right)
                row[f"{name}_relative_improvement"] = relative_improvement(left, right)
        rows.append(row)

    missing_counts = {name: len(values) for name, values in missing.items()}
    if any(missing_counts.values()) and not args.allow_partial:
        print(json.dumps({"status": "INCOMPLETE", "missing_counts": missing_counts,
                          "missing_tasks": missing}, ensure_ascii=False, indent=2))
        raise RuntimeError("Results are incomplete; finish runs or add --allow-partial")
    if not rows:
        raise RuntimeError("No complete untuned L/TimesFM/Moirai task triplets are available")

    output = RESULT_ROOT / "foundation_comparison" / args.output
    if output.exists():
        raise FileExistsError(f"Use a new --output name: {output}")
    pairwise = pairwise_summary(rows)
    three_way = three_way_summary(rows)
    common.write_csv(output / "comparison_pairs.csv", rows)
    common.write_csv(output / "pairwise_summary.csv", pairwise)
    common.write_csv(output / "three_way_summary.csv", three_way)
    common.write_json(output / "comparison_summary.json", {
        "status": "COMPLETE" if not any(missing_counts.values()) else "PARTIAL",
        "foundation_run": args.foundation_run,
        "L_root": str(L_MODEL_ROOT),
        "L_reference": str(reference_path),
        "L_definition": (
            "Law-guided L from current compar_result with validation-selected architecture and fixed "
            "training hyperparameters; no 3par hyperparameter tuning"),
        "L_context_length": 12,
        "L_seeds": seeds,
        "foundation_context_length": settings["context_length"],
        "target_alignment": settings["target_alignment"],
        "target_start_offset": settings["target_start_offset"],
        "metrics": list(METRICS), "fake_foundation_seed_repetition": False,
        "pair_count": len(rows), "expected_pair_count": len(tasks),
        "missing_counts": missing_counts, "missing_tasks": missing,
        "pairwise_summary": pairwise, "three_way_summary": three_way,
    })
    print(f"Compared {len(rows)}/{len(tasks)} task triplets")
    for item in three_way:
        if item["scope"] == "overall":
            print(f"{item['metric'].upper()} best counts L/TimesFM/Moirai/tie = "
                  f"{item['L_wins']}/{item['timesfm3_0_wins']}/"
                  f"{item['moirai2_0_wins']}/{item['tied_best']}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
