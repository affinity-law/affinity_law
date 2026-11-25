"""Run frozen configurations by seed without selecting on test results."""

from .artifacts import freeze_json, read_json
from .evaluation import evaluate_test, read_test_cache
from .report import write_csv
from .training import fit, training_request

PURPOSE = "fixed_best_config_three_seed_retraining"
TOP3_PURPOSE = "top3_validation_confirmation"


def best_configuration(task, settings, folder, best):
    return {"task_id": task["id"], "selected_slot": best["slot"],
            "hyperparameters": best["executed_config"], "spec": task["spec"],
            "time_lag": best["executed_config"].get("time_lag", settings["time_lag"]), "horizon": task["horizon"], "channels": task["channels"],
            "training": settings["training"], "train_ratio": settings["train_ratio"], "val_ratio": settings["val_ratio"],
            "params": best["result"]["params"],
            "initialization": "fresh_for_each_seed", "test_used_for_selection": False, "smoke": settings["smoke"]}


def seed_row(task, seed, result=None, score=None, checkpoint=None):
    result = result or {}
    status = "COMPLETE" if score is not None else "FAILED" if result.get("status") == "FAIL" else "PENDING"
    return {"dataset": task["dataset"], "horizon": task["horizon"], "seed": seed, "status": status,
            **{f"test_{metric}": score["overall_metrics"][metric] if score is not None else "N/A"
               for metric in ("rmse", "mae", "wmape")},
            **{key: result.get(key, "N/A") for key in ("val_rmse", "best_epoch", "epochs_run", "params")},
            "checkpoint": checkpoint or (f"seed_{seed}/training/{result['checkpoint']}"
                                           if result.get("checkpoint") else "N/A")}


def confirmed_source(task, folder, best, seed):
    selection = read_json(folder / "best/selection.json")
    source = selection["seed_sources"][str(seed)]
    saved = read_json(folder / source["record"])
    result = (saved["record"]["result"] if source["kind"] == "trial"
              else saved["training"]["attempts"][-1].get("result"))
    if result is None or result["status"] != "COMPLETE" or result["seed"] != seed:
        raise ValueError(f"Incomplete confirmed seed record: {task['id']}/seed{seed}")
    if result["config"] != best["executed_config"]:
        raise ValueError("Confirmed configuration differs from the selected configuration")
    fit_folder = folder / source["fit_folder"]
    for key in ("checkpoint", "candidate_model", "stats"):
        if not (fit_folder / result[key]).is_file():
            raise ValueError(f"Missing confirmed seed artifact: seed{seed}/{key}")
    return fit_folder, result


def read_seed_results(task, settings, folder, best):
    """Rebuild per-seed results; incomplete runs remain PENDING without partial means."""
    rows = []
    config_path = folder / "best/config.json"
    if config_path.exists():
        if read_json(config_path) != best_configuration(task, settings, folder, best):
            raise ValueError(f"Best configuration differs from the frozen selection: {task['id']}")
    selection = read_json(folder / "best/selection.json") if (folder / "best/selection.json").exists() else {}
    if selection.get("seed_sources"):
        for seed in settings["training"]["final_seeds"]:
            fit_folder, result = confirmed_source(task, folder, best, seed)
            score = read_test_cache(task, best["executed_config"], settings, result,
                                    folder / "best" / f"seed_{seed}" / "test", seed)
            checkpoint = str((fit_folder / result["checkpoint"]).relative_to(folder))
            rows.append(seed_row(task, seed, result, score, checkpoint))
        return rows
    for seed in settings["training"]["final_seeds"]:
        directory = folder / "best" / f"seed_{seed}"
        path = directory / "record.json"
        result = score = None
        if path.exists():
            if not config_path.exists():
                raise ValueError("Retraining record is missing the frozen best configuration")
            training = read_json(path)["training"]
            expected = training_request(task, best["executed_config"], settings, seed,
                                        settings["training"]["short_epochs"], settings["training"]["short_patience"], PURPOSE)
            if training["request"] != expected:
                raise ValueError(f"Retraining cache configuration mismatch: {task['id']}/seed{seed}")
            if training["attempts"]:
                result = training["attempts"][-1].get("result")
            if result is not None:
                if result["status"] == "COMPLETE":
                    for key in ("checkpoint", "candidate_model", "stats"):
                        if not (directory / "training" / result[key]).is_file():
                            raise ValueError(f"Missing retraining artifact: seed{seed}/{key}")
                    score = read_test_cache(task, best["executed_config"], settings, result, directory / "test", seed)
        rows.append(seed_row(task, seed, result, score))
    return rows


def retrain_best(task, settings, root, folder, best):
    """Reuse confirmed seed models in the new protocol; retain legacy retraining support."""
    freeze_json(folder / "best/config.json", best_configuration(task, settings, folder, best))
    rows = read_seed_results(task, settings, folder, best)
    output = folder / "best/seed_results.csv"
    selection = read_json(folder / "best/selection.json")
    if selection.get("seed_sources"):
        for index, seed in enumerate(settings["training"]["final_seeds"]):
            if rows[index]["status"] == "COMPLETE":
                continue
            fit_folder, result = confirmed_source(task, folder, best, seed)
            score = evaluate_test(task, best["executed_config"], settings, root, fit_folder, result,
                                  folder / "best" / f"seed_{seed}" / "test", seed)
            checkpoint = str((fit_folder / result["checkpoint"]).relative_to(folder))
            rows[index] = seed_row(task, seed, result, score, checkpoint)
            write_csv(output, rows, list(rows[0]))
        write_csv(output, rows, list(rows[0]))
        return rows
    for index, seed in enumerate(settings["training"]["final_seeds"]):
        if rows[index]["status"] == "COMPLETE":
            continue
        print(f"{task['id']} retraining selected configuration with seed={seed}", flush=True)
        directory = folder / "best" / f"seed_{seed}"
        result = fit(task, best["executed_config"], settings, root, directory / "training", seed=seed,
                     epochs=settings["training"]["short_epochs"], patience=settings["training"]["short_patience"],
                     purpose=PURPOSE, record_path=directory / "record.json")
        score = None
        if result["status"] == "COMPLETE":
            score = evaluate_test(task, best["executed_config"], settings, root,
                                  directory / "training", result, directory / "test", seed)
        rows[index] = seed_row(task, seed, result, score)
        write_csv(output, rows, list(rows[0]))
        if score is None:
            break
    write_csv(output, rows, list(rows[0]))
    return rows
