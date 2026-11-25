"""Check current L inputs and the aligned 63-dataset foundation experiment."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODEL_ROOT / "_runtime"))
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np

import common
from portable_paths import (TUNING_ROOT, L_MODEL_ROOT, L_SEEDS, FOUNDATION_RUN,
                            load_manifest, load_l_reference,
                            normalize_manifest_paths, l_run_paths, index_tuned_l, shared_input_signature)
from compare import model_result
from compare_untuned import current_l_result, validate_l_reference


def load_and_probe_model(model_name, model_settings, settings, device):
    import torch

    adapter = MODEL_ROOT / model_settings["adapter"]
    spec = importlib.util.spec_from_file_location(f"preflight_{model_name}", adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch_size = min(2, model_settings["default_batch_size"])
    model = module.CandidateModel(
        horizon=3, station_num=1,
        checkpoint=MODEL_ROOT / model_settings["checkpoint"],
        device=device, batch_size=batch_size).eval()
    outputs = {}
    for horizon in settings["horizons"]:
        model.horizon = horizon
        x = torch.zeros(1, settings["context_length"], 1, dtype=torch.float32)
        y = model(x)
        if tuple(y.shape) != (1, horizon, 1) or not torch.isfinite(y).all():
            raise ValueError(f"{model_name} failed the output contract for horizon={horizon}")
        outputs[str(horizon)] = list(y.shape)
    params = sum(parameter.numel() for parameter in model.parameters())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"status": "PASS", "output_shapes": outputs, "params": params}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l-run", action="append",
                        help="3par output run name/path; repeat for shards. Default scans all runs.")
    parser.add_argument("--load-models", action="store_true",
                        help="Load both local checkpoints and run real 3/12-step probes")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    settings = common.load_settings()
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required (hashlib.file_digest)")
    manifest, tasks = load_manifest()
    datasets = sorted({task["dataset"] for task in tasks})
    tuning_manifest = TUNING_ROOT / "tasks.json"
    if not tuning_manifest.is_file():
        raise FileNotFoundError(tuning_manifest)
    mapped_tuning = normalize_manifest_paths(common.read_json(tuning_manifest))
    manifest_match = shared_input_signature(manifest) == shared_input_signature(mapped_tuning)
    if not manifest_match:
        raise ValueError("foundation_model and 3par task manifests differ")
    data_checks = []
    reference_path, l_reference = load_l_reference()
    expected_ids = {task["id"] for task in tasks}
    if set(l_reference) != expected_ids:
        raise ValueError("Current exp_L.csv does not match the 126-task manifest")
    aligned_l_target_checks = 0
    tuning_data = {task["dataset"]: Path(task["data_path"]) for task in mapped_tuning["tasks"]}
    seen = set()
    for task in tasks:
        if task["dataset"] in seen:
            continue
        seen.add(task["dataset"])
        foundation_path = Path(task["data_path"])
        tuning_path = tuning_data[task["dataset"]]
        if not foundation_path.is_file() or not tuning_path.is_file():
            raise FileNotFoundError(f"Missing paired data file for {task['dataset']}")
        foundation_hash = common.sha256_file(foundation_path)
        tuning_hash = common.sha256_file(tuning_path)
        if foundation_hash != tuning_hash:
            raise ValueError(f"Foundation/tuning data differ: {task['dataset']}")
        data = common.series_view(foundation_path)
        expected = manifest["tasks"][[item["dataset"] for item in manifest["tasks"]].index(task["dataset"])]
        if len(data) != expected["length"] or data.shape[1] != expected["channels"]:
            raise ValueError(f"Manifest/data shape mismatch: {task['dataset']}")
        normalized, mean, std, _, val_end, total_length = common.aligned_evaluation_data(
            foundation_path, settings)
        for horizon in settings["horizons"]:
            windows = len(normalized) - settings["context_length"] - horizon + 1
            target_points = windows + horizon - 1
            if windows < 1 or target_points != total_length - val_end - settings["target_start_offset"]:
                raise ValueError(f"Aligned window count mismatch: {task['dataset']}/h{horizon}")
            raw_targets = np.stack([
                normalized[index + settings["context_length"]:
                           index + settings["context_length"] + horizon]
                for index in range(windows)])
            aligned_target = common.average_future_steps(raw_targets) * std + mean
            current_l_target = (L_MODEL_ROOT /
                                 task["dataset"] / f"h{horizon}" / "law_s2026" / "test" /
                                 "averaged_final_targets.npy")
            if not current_l_target.is_file():
                raise FileNotFoundError(current_l_target)
            np.testing.assert_allclose(
                aligned_target, np.load(current_l_target), rtol=4e-6, atol=4e-6)
            aligned_l_target_checks += 1
        data_checks.append({
            "dataset": task["dataset"], "length": len(data), "channels": data.shape[1],
            "sha256": foundation_hash, "status": "PASS",
        })
    complete_l_tasks = []
    foundation_complete = {name: [] for name in settings["models"]}
    for task in tasks:
        result = current_l_result(task, L_SEEDS)
        if result is not None:
            metrics = [common.metrics(prediction, target)
                       for prediction, target in zip(result["predictions"], result["targets"])]
            validate_l_reference(task, l_reference[task["id"]], L_SEEDS, metrics)
            complete_l_tasks.append(task["id"])
        for name in settings["models"]:
            if model_result(name, FOUNDATION_RUN, task, settings) is not None:
                foundation_complete[name].append(task["id"])
    l_paths = l_run_paths(args.l_run)
    tuned_l_index, duplicates = index_tuned_l(tasks, l_paths, L_SEEDS)
    device = None
    model_checks = {name: {"status": "NOT_REQUESTED"} for name in settings["models"]}
    if args.load_models:
        import torch
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        for name, model_settings in settings["models"].items():
            model_checks[name] = load_and_probe_model(name, model_settings, settings, device)
    run_status = {}
    for model in settings["models"]:
        path = MODEL_ROOT / model / "outputs" / FOUNDATION_RUN / "status.json"
        run_status[model] = common.read_json(path) if path.is_file() else {"status": "NOT_STARTED"}
    report = {
        "status": "PASS", "python_executable": sys.executable,
        "python_version": sys.version,
        "L_model_root": str(L_MODEL_ROOT), "L_reference": str(reference_path),
        "dataset_count": len(datasets), "task_count": len(tasks),
        "horizons": settings["horizons"], "context_length": settings["context_length"],
        "target_start_offset": settings["target_start_offset"],
        "target_alignment": settings["target_alignment"],
        "manifest_matches_3par": manifest_match,
        "data_match_count": len(data_checks), "data_checks": data_checks,
        "aligned_L_target_checks": aligned_l_target_checks,
        "L_complete_tasks": len(complete_l_tasks), "L_expected_tasks": len(tasks),
        "L_missing_tasks": sorted(expected_ids - set(complete_l_tasks)),
        "foundation_complete_tasks": {name: len(ids) for name, ids in foundation_complete.items()},
        "comparison_ready": len(complete_l_tasks) == len(tasks) and all(
            len(ids) == len(tasks) for ids in foundation_complete.values()),
        "tuning_root": str(TUNING_ROOT),
        "tuned_L_run_paths": [str(path) for path in l_paths],
        "tuned_L_complete_tasks": len(tuned_l_index),
        "tuned_L_missing_tasks": sorted(expected_ids - set(tuned_l_index)),
        "tuned_L_duplicate_tasks": duplicates,
        "tuned_comparison_ready": len(tuned_l_index) == len(tasks) and not duplicates,
        "models_loaded": args.load_models, "model_device": device,
        "model_checks": model_checks, "formal_run_status": run_status,
    }
    if args.report:
        report_path = args.report if args.report.is_absolute() else HERE / args.report
        if report_path.exists():
            raise FileExistsError(f"Use a new report name: {report_path}")
        common.write_json(report_path, report)
    print(json.dumps({
        "status": report["status"], "dataset_count": report["dataset_count"],
        "task_count": report["task_count"], "manifest_matches_3par": manifest_match,
        "data_match_count": report["data_match_count"],
        "aligned_L_target_checks": report["aligned_L_target_checks"],
        "L_complete_tasks": report["L_complete_tasks"],
        "L_expected_tasks": report["L_expected_tasks"],
        "foundation_complete_tasks": report["foundation_complete_tasks"],
        "tuned_L_complete_tasks": report["tuned_L_complete_tasks"],
        "tuned_comparison_ready": report["tuned_comparison_ready"],
        "comparison_ready": report["comparison_ready"],
        "models": model_checks,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
