"""Regression checks for the compact workflow using local mock LLM responses."""

import argparse
import csv
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import tune
import hpo.retrain as retrain
from hpo.agent import AgentAPIError, get_search_state, parse_decision
from hpo.artifacts import read_json, save_json
from hpo.config import FEATURES_DIR, ROOT, load_settings, tasks_from_manifest
from hpo.models import default_config
from hpo.report import l_original_reference, summary_row, write_csv
from tuning_features import FEATURE_NAMES, load_features


def legacy_screening_settings():
    settings = load_settings()
    settings["space"].pop("time_lag", None)
    settings["search"]["probe_values"].pop("time_lag", None)
    settings["search"].pop("screening_seeds")
    settings["search"].pop("confirmation_top_k")
    settings["search"]["confirmation_seeds"] = [2027, 2028]
    return settings


class SimpleTuningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".simple_test_", dir=ROOT)
        self.directory = Path(self.temp.name).resolve()
        self.run = self.directory / "run"
        self.task = tasks_from_manifest()[0]
        self.model = self.directory / "candidate_model.py"
        self.model.write_text('''import torch
from torch import nn

class _B_MLP(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.layer = nn.Linear(channels, hidden)
    def forward(self, x):
        return torch.relu(self.layer(x))

class _C_Linear(nn.Module):
    def __init__(self, hidden, horizon, channels):
        super().__init__()
        self.layer = nn.Linear(hidden, channels)
        self.horizon = horizon
    def forward(self, x):
        return self.layer(x[:, -1:, :]).expand(-1, self.horizon, -1)

def _build_b(name, channels):
    return _B_MLP(channels, hidden=8)

def _build_c(name, horizon, channels):
    return _C_Linear(hidden=8, horizon=horizon, channels=channels)

class _E_Composition(nn.Module):
    def __init__(self, count, name):
        super().__init__()

class CandidateModel(nn.Module):
    def __init__(self, time_lag, pre_len, station_num):
        super().__init__()
        self.b = _build_b("MLP Block", station_num)
        self.c = _build_c("Linear Head", pre_len, station_num)
        self.e = _E_Composition(1, "Single Block")
    def forward(self, x):
        return self.c(self.b(x))
''', encoding="utf-8")
        self.data = self.directory / "series.npy"
        x = np.arange(384)
        np.save(self.data, np.column_stack((np.sin(x / 5), 0.005 * x + np.cos(x / 11))).astype(np.float32))
        self.features = self.directory / "meta_features.csv"
        self.features.write_text("indicator_name,value\n" + "".join(f"{name}_score,{(i + 1) / 10}\n" for i, name in enumerate(FEATURE_NAMES)), encoding="utf-8")

    def tearDown(self):
        self.assertTrue(self.directory.is_relative_to(ROOT.resolve()))
        self.temp.cleanup()

    def invoke(self, extra=()):
        argv = ["--model", str(self.model), "--data", str(self.data), "--features", str(self.features),
                "--run", "smoke_simple_test", "--horizons", "3", "12", "--smoke", "--mock-agent", "--trials", "13", *extra]
        with patch("tune.run_folder", return_value=self.run), redirect_stdout(io.StringIO()):
            tune.main(argv)

    def test_one_command_exports_complete_results_and_rejects_run_reuse(self):
        original_evaluate = tune.evaluate_test
        def evaluate_after_freeze(*args, **kwargs):
            self.assertTrue((self.run / "selection_freeze.json").exists())
            self.assertEqual(len(list(self.run.glob("tasks/*/h*/best/selection.json"))), 2)
            self.assertLessEqual(len(list(self.run.glob("tasks/*/h*/trials/*/trial.json"))), 26)
            return original_evaluate(*args, **kwargs)
        with patch("tune.evaluate_test", side_effect=evaluate_after_freeze):
            self.invoke()
        frozen = read_json(self.run / "selection_freeze.json")
        self.assertEqual(len(frozen["task_ids"]), 2)
        summary = tune.summary_path(self.run)
        self.assertRegex(summary.name, r"^summary_[0-9]{12}\.csv$")
        self.assertEqual(list(self.run.glob("summary*.csv")), [summary])
        with summary.open(encoding="utf-8-sig", newline="") as handle:
            summaries = list(csv.DictReader(handle))
        self.assertEqual(len(summaries), 2)
        for folder in sorted(self.run.glob("tasks/*/h*")):
            saved_trials = [read_json(p) for p in sorted(folder.glob("trials/*/trial.json"))]
            records = [saved["record"] for saved in saved_trials]
            selection = read_json(folder / "best/selection.json")
            confirmation = read_json(folder / "top3/confirmation.json")
            self.assertIn(selection["slot"], confirmation["confirmation_slots"])
            self.assertIn(1, confirmation["confirmation_slots"])
            self.assertEqual(selection["slot"], confirmation["winner_slot"])
            expected = records[selection["slot"] - 1]
            self.assertEqual({r["executed_config"]["time_lag"] for r in records}, {12, 32, 64})
            self.assertEqual(read_json(folder / "best/config.json")["time_lag"], selection["config"]["time_lag"])
            source = folder / expected["fit_folder"]
            self.assertEqual((folder / "best/best_model_weights.pth").read_bytes(),
                             (source / expected["result"]["checkpoint"]).read_bytes())
            self.assertEqual((folder / "best/candidate_model.py").read_bytes(),
                             (source / expected["result"]["candidate_model"]).read_bytes())
            self.assertEqual(len(saved_trials[0]["training"]["attempts"]), 1)
            requests = [saved["training"]["request"] for saved in saved_trials]
            self.assertTrue(all((r["epochs"], r["patience"]) == (1, 1) for r in requests))
            self.assertTrue(all(r["purpose"].startswith("coordinate_") for r in requests))
            reviewed = [saved for saved in saved_trials if "agent" in saved]
            self.assertTrue(reviewed)
            state = reviewed[0]["agent"]["state"]
            self.assertEqual(state["task"]["meta_features"], load_features(self.features)["values"])
            self.assertIn(state["review_stage"], {"training_parameters", "structural_midpoint", "structural_parameters"})
            self.assertNotIn("test_rmse", json.dumps(state))
            self.assertNotIn("l_original", json.dumps(state))
            prompt = next(folder.glob("trials/trial_*/agent/attempt_001/prompt.txt"))
            self.assertTrue(prompt.exists())
            request = reviewed[0]["agent"]["attempts"][0]["request"]
            self.assertNotIn("max_tokens", request)
            self.assertNotIn("max_completion_tokens", request)
            self.assertTrue((folder / "trials.csv").exists())
            self.assertTrue((folder / "slot_hyperparameters.csv").exists())
            self.assertTrue((folder.parent / "slot_hyperparameters.csv").exists())
            self.assertEqual(records[0]["execution_source"], "default")
            expected_json = {folder / "task.json", folder / "coordinate_trace.json",
                             folder / "best/selection.json", folder / "best/config.json",
                             folder / "test/default/metrics.json",
                             *(folder / "trials" / f"trial_{slot:03d}" / "trial.json" for slot in range(1, len(records) + 1))}
            for seed in (2026, 2027, 2028):
                expected_json.add(folder / f"best/seed_{seed}/test/metrics.json")
            for slot in confirmation["confirmation_slots"]:
                for seed in (2028,):
                    expected_json.add(folder / f"top3/slot_{slot:03d}/seed_{seed}/record.json")
            for record in records:
                expected_json.add(folder / f"trials/trial_{record['slot']:03d}/seed_2027/record.json")
                self.assertEqual(set(record["screening_results"]), {"2026", "2027"})
                self.assertNotIn("loss_profile", record["executed_config"])
                for entry in record["screening_results"].values():
                    result = entry["result"]
                    self.assertEqual((result["mse_weight"], result["mae_weight"]), (0.2, 0.8))
                self.assertAlmostEqual(record["screening_val_rmse_mean"],
                                       np.mean([entry["result"]["val_rmse"] for entry in record["screening_results"].values()]))
            if expected["slot"] != 1:
                expected_json.add(folder / "test/best/metrics.json")
            all_json = set(folder.rglob("*.json"))
            self.assertTrue(expected_json.issubset(all_json))
            for saved, trial in zip(saved_trials, sorted(folder.glob("trials/trial_*"))):
                self.assertTrue((trial / "config.json").exists())
                attempt = saved["training"]["attempts"][-1]["directory"]
                self.assertTrue((trial / "training" / attempt / "metrics.json").exists())
        with patch("tune.fit", side_effect=AssertionError("A completed run must not train again")), \
             patch.object(tune.TuningAgent, "call", side_effect=AssertionError("A completed run must not call the LLM")), \
             self.assertRaisesRegex(ValueError, "already exists"):
            self.invoke()

    def test_formal_default_and_candidates_use_fixed_training_limits(self):
        argv = ["--model", str(self.model), "--data", str(self.data), "--features", str(self.features),
                "--run", "fixed_training_test", "--horizons", "3", "12", "--trials", "3"]
        for index, extra in enumerate(([], ["--epochs", "300", "--patience", "15"])):
            with self.subTest(explicit_limits=bool(extra)), redirect_stdout(io.StringIO()), \
                 patch("tune.run_folder", return_value=self.directory / f"formal_{index}"), \
                 patch("tune.fit", side_effect=lambda *a, **kw: {"status": "COMPLETE", "val_rmse": 1.0,
                                                              "seed": kw["seed"], "config": a[1]}) as fit, \
                 patch("tune.finish_dataset"), \
                 patch("tune.TuningAgent", side_effect=AssertionError("This check must not call the LLM")):
                tune.main(argv + extra)
                self.assertEqual(fit.call_count, 12)
                for call in fit.call_args_list:
                    self.assertEqual((call.kwargs["epochs"], call.kwargs["patience"]), (300, 15))
                    self.assertFalse(call.args[2]["smoke"])

    def test_fixed_training_limits_cannot_be_changed_by_cli_or_settings(self):
        for extra in (["--epochs", "40"], ["--patience", "5"], ["--patience", "0"],
                      ["--smoke", "--run", "smoke_limits", "--epochs", "300"]):
            with self.subTest(argv=extra), redirect_stderr(io.StringIO()), \
                 patch("tune.input_tasks", side_effect=AssertionError("Invalid arguments must stop before the run starts")), \
                 self.assertRaises(SystemExit) as error:
                tune.main(extra)
            self.assertEqual(error.exception.code, 2)
        settings = load_settings()
        path = self.directory / "settings.json"
        for epochs, patience in ((40, 15), (300, 5), (40, 5)):
            settings["training"].update(short_epochs=epochs, short_patience=patience)
            save_json(path, settings)
            with self.subTest(limits=(epochs, patience)), self.assertRaisesRegex(ValueError, "300 maximum epochs and patience 15"):
                load_settings(path)
        smoke = load_settings(smoke=True)
        self.assertEqual((smoke["training"]["short_epochs"], smoke["training"]["short_patience"]), (1, 1))

    def test_screening_budget_defaults_to_32_and_accepts_cli_boundaries(self):
        self.assertEqual(load_settings()["search"]["trials"], 32)
        for index, (extra, expected) in enumerate((([], 32), (["--trials", "1"], 1),
                                                  (["--trials", "24"], 24), (["--trials", "32"], 32))):
            with self.subTest(argv=extra), redirect_stdout(io.StringIO()), \
                 patch("tune.input_tasks", return_value=[]), \
                 patch("tune.load_inputs", return_value={}), \
                 patch("tune.run_folder", return_value=self.directory / f"budget_{index}"), \
                 patch("tune.freeze_run") as freeze, \
                 patch("tune.fit", side_effect=AssertionError("Budget validation must not train")), \
                 patch("tune.TuningAgent", side_effect=AssertionError("Budget validation must not call the LLM")):
                tune.main(extra)
            self.assertEqual(freeze.call_args.args[3]["search"]["trials"], expected)

    def test_screening_budget_rejects_out_of_range_cli_and_settings(self):
        for value in (-1, 0, 33):
            with self.subTest(cli=value), redirect_stderr(io.StringIO()), \
                 patch("tune.input_tasks", side_effect=AssertionError("Invalid budget must stop before loading tasks")), \
                 self.assertRaises(SystemExit) as error:
                tune.main(["--trials", str(value)])
            self.assertEqual(error.exception.code, 2)
        settings = load_settings()
        path = self.directory / "budget_settings.json"
        for value in (1, 24, 32, -1, 0, 33):
            settings["search"]["trials"] = value
            save_json(path, settings)
            with self.subTest(settings=value):
                if 1 <= value <= 32:
                    self.assertEqual(load_settings(path)["search"]["trials"], value)
                else:
                    with self.assertRaises(ValueError):
                        load_settings(path)

    def test_agent_cannot_override_fixed_training_limits(self):
        settings = load_settings()
        proposal = {"config": default_config(self.task, settings), "source": "coordinate_probe",
                    "phase": "coordinate", "parameter": "lr", "base_slot": None,
                    "review_stage": "training_parameters"}
        state = get_search_state(self.task, settings, proposal, [], 9)
        self.assertEqual(state["training_protocol"]["max_epochs"], 300)
        self.assertEqual(state["training_protocol"]["early_stop_patience"], 15)
        self.assertEqual(state["training_protocol"]["screening_seeds"], [2026, 2027])
        self.assertNotIn("loss_profile", state["space"])
        self.assertFalse(state["training_protocol"]["loss_tunable"])
        self.assertEqual(state["training_protocol"]["early_stop_metric"], "normalized_validation_0.2_mse_0.8_mae_loss")
        for name in ("epochs", "patience", "short_epochs", "short_patience", "max_epochs", "early_stop_patience"):
            with self.subTest(parameter=name):
                self.assertNotIn(name, state["space"])
                decision = {"action": "edit_proposal", "overrides": {name: 10},
                            "evidence_trials": [], "hypothesis": "Attempt to change a fixed setting."}
                with self.assertRaises(ValueError):
                    parse_decision(json.dumps(decision), state, self.task, settings, ROOT, set())

    def test_features_read_user_files_long_and_wide_without_recomputation(self):
        expected = load_features(self.features)
        self.assertEqual(set(expected["values"]), set(FEATURE_NAMES))
        self.assertIn("unverified", expected["scope"])
        wide = self.directory / "wide.csv"
        wide.write_text(",".join(FEATURE_NAMES) + "\n" + ",".join(str(v) for v in expected["values"].values()) + "\n", encoding="utf-8")
        self.assertEqual(load_features(wide)["values"], expected["values"])
        for value in ("NaN", "1.5"):
            invalid = self.directory / "bad.csv"
            invalid.write_text(self.features.read_text().replace("0.1", value), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_features(invalid)
        paths = list(FEATURES_DIR.rglob("meta_features.csv"))
        self.assertEqual(len(paths), 63)
        for path in paths:
            self.assertEqual(len(load_features(path)["values"]), 7)

    def test_existing_run_rejects_reuse(self):
        tasks, features, settings = [self.task], {}, load_settings(smoke=True)
        tune.freeze_run(self.run, tasks, features, settings, True)
        path = self.run / "tuning_protocol.json"
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            tune.freeze_run(self.run, tasks, features, settings, True)
        incompatible = copy.deepcopy(settings)
        incompatible["search"]["probe_values"]["lr"][0] = 0.0001
        with self.assertRaisesRegex(ValueError, "already exists"):
            tune.freeze_run(self.run, tasks, features, incompatible, True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            tune.freeze_run(self.run, [], features, settings, True)
        self.assertEqual(path.read_bytes(), original)

    def test_default_winner_is_exported_and_retrained_without_llm(self):
        with patch.object(tune.TuningAgent, "call", side_effect=AssertionError("A default-only run must not call the LLM")):
            self.invoke(["--trials", "1"])
        with tune.summary_path(self.run).open(encoding="utf-8-sig", newline="") as handle:
            summaries = {f"h{row['horizon']}": row for row in csv.DictReader(handle)}
        for folder in self.run.glob("tasks/*/h*"):
            selection = read_json(folder / "best/selection.json")
            summary = summaries[folder.name]
            self.assertEqual(selection["slot"], 1)
            self.assertEqual(summary["search_default_test_rmse"], summary["best_test_rmse"])
            self.assertEqual(summary["l_original_status"], "smoke_excluded")
            self.assertEqual(summary["l_original_test_rmse_mean"], "")
            self.assertEqual(summary["agent_decisions"], "0")
            self.assertTrue((folder / "best/best_model_weights.pth").exists())
            self.assertEqual(len(read_json(folder / "trials/trial_001/trial.json")["training"]["attempts"]), 1)

    def test_report_only_uses_saved_settings_and_rejects_wrong_test_cache(self):
        self.invoke(["--trials", "1"])
        summary = tune.summary_path(self.run)
        original = {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        with patch("tune.run_folder", return_value=self.run), redirect_stdout(io.StringIO()), \
             patch("tune.load_settings", side_effect=AssertionError("Report-only must use saved settings")), \
             patch("tune.input_tasks", side_effect=AssertionError("Report-only must use saved tasks")), \
             patch("tune.fit", side_effect=AssertionError("Report-only must not train")), \
             patch("hpo.retrain.fit", side_effect=AssertionError("Report-only must not retrain")), \
             patch("tune.evaluate_test", side_effect=AssertionError("Report-only must not test")), \
             patch("tune.select_best", side_effect=AssertionError("Report-only must not reselect")), \
             patch("tune.TuningAgent", side_effect=AssertionError("Report-only must not create an LLM client")):
            tune.main(["--run", "smoke_simple_test", "--report-only"])
            self.assertEqual(tune.summary_path(self.run), summary)
            self.assertEqual(list(self.run.glob("summary*.csv")), [summary])
            self.assertEqual({p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()}, original)
            path = next(self.run.glob("tasks/*/h*/test/default/metrics.json"))
            cached = read_json(path)
            cached["request"]["seed"] = 999
            save_json(path, cached)
            with self.assertRaisesRegex(ValueError, "Test cache"):
                tune.main(["--run", "smoke_simple_test", "--report-only"])
            self.assertEqual(summary.read_bytes(), original[summary])

    def test_historical_reference_requires_matching_task_and_three_seeds(self):
        path = self.directory / "history.csv"
        row = {"dataset": self.task["dataset"], "horizon": self.task["horizon"],
               "seed_count": 3, "seeds": "2026;2027;2028", "rmse_mean": 10, "rmse_std": 2,
               "mae_mean": 5, "mae_std": 1}
        write_csv(path, [row], list(row))
        settings = load_settings()
        reference = l_original_reference(self.task, settings, path)
        self.assertEqual(reference["l_original_seed_count"], 3)
        self.assertEqual(reference["l_original_test_rmse_mean"], 10)
        mismatch = l_original_reference({**self.task, "dataset": "different"}, settings, path)
        self.assertEqual(mismatch["l_original_status"], "unavailable")
        self.assertIsNone(mismatch["l_original_test_rmse_mean"])
        smoke = l_original_reference(self.task, load_settings(smoke=True), path)
        self.assertIsNone(smoke["l_original_test_rmse_mean"])
        record = {"slot": 1, "executed_config": default_config(self.task, settings),
                  "result": {"val_rmse": 1e-10, "params": 100}}
        scores = {"default": {"overall_metrics": {"rmse": 8, "mae": 4}},
                  "best": {"overall_metrics": {"rmse": 11, "mae": 4}}}
        seed_rows = [{"seed": seed, "status": "COMPLETE", "test_rmse": 7 + index,
                      "test_mae": 3 + index, "test_wmape": 0.1}
                     for index, seed in enumerate(settings["training"]["final_seeds"])]
        with patch("hpo.report.l_original_reference", return_value=reference):
            summary = summary_row(self.task, {"scope": "test"}, settings, record, record, [record], scores, seed_rows)
        self.assertEqual(summary["test_rmse_improvement_vs_l_original"], 0.2)
        self.assertEqual(summary["test_mae_improvement_vs_l_original"], 0.2)
        self.assertEqual(summary["recommendation_source"], "TUNED_TOP3")
        self.assertEqual(summary["search_default_test_rmse"], 8)
        self.assertEqual(summary["val_improvement_ratio"], "N/A")
        self.assertEqual(summary["best_test_rmse_mean"], 8)
        self.assertEqual(summary["best_test_rmse_std"], 1)
        self.assertEqual((summary["max_epochs"], summary["early_stop_patience"]), (300, 15))
        worse_rows = [{**row, "test_rmse": 11 + index} for index, row in enumerate(seed_rows)]
        with patch("hpo.report.l_original_reference", return_value=reference):
            worse = summary_row(self.task, {"scope": "test"}, settings, record, record, [record], scores, worse_rows)
        self.assertEqual(worse["recommendation_source"], "L_DEFAULT")
        self.assertEqual(worse["recommended_slot"], "L_DEFAULT")
        # The report uses its saved settings and cannot relabel an earlier 40/5 run as 300/15.
        legacy = copy.deepcopy(settings)
        legacy["training"].update(short_epochs=40, short_patience=5)
        legacy["training"].pop("final_seeds")
        with patch("hpo.report.l_original_reference", return_value=reference):
            old_summary = summary_row(self.task, {"scope": "test"}, legacy, record, record, [record], scores)
        self.assertEqual((old_summary["max_epochs"], old_summary["early_stop_patience"]), (40, 5))
        self.assertEqual(old_summary["test_comparison"], "search_best_single_seed_vs_l_original")
        self.assertEqual(old_summary["test_rmse_improvement_vs_l_original"], -0.1)
        write_csv(path, [row, row], list(row))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            l_original_reference(self.task, settings, path)
        write_csv(path, [{**row, "seeds": "2026;2026;2028"}], list(row))
        with self.assertRaisesRegex(ValueError, "exactly three seeds"):
            l_original_reference(self.task, settings, path)

    def test_numeric_dataset_count_is_fixed_subset_not_all_128(self):
        args = argparse.Namespace(model=None, data=None, datasets="2", horizons=None)
        self.assertEqual(len(tune.input_tasks(args)), 4)
        args.datasets = "0"
        with self.assertRaises(ValueError):
            tune.input_tasks(args)

    def test_dataset_range_is_inclusive_and_counts_datasets_before_horizons(self):
        tasks = tasks_from_manifest()
        names = sorted({task["dataset"] for task in tasks})
        args = argparse.Namespace(model=None, data=None, datasets="3:10", horizons=None)
        with patch("tune.tasks_from_manifest", return_value=list(reversed(tasks))):
            selected = tune.input_tasks(args)
        self.assertEqual({task["dataset"] for task in selected}, set(names[2:10]))
        self.assertEqual(len(selected), 16)
        args.horizons = [12]
        selected = tune.input_tasks(args)
        self.assertEqual({task["dataset"] for task in selected}, set(names[2:10]))
        self.assertEqual(len(selected), 8)
        self.assertEqual({task["horizon"] for task in selected}, {12})
        for index in (1, 3, len(names)):
            args.datasets = f"{index}:{index}"
            selected = tune.input_tasks(args)
            self.assertEqual([task["dataset"] for task in selected], [names[index - 1]])
        for invalid in ("0:3", "10:3", f"3:{len(names) + 1}", "3:", "3:2:10", "a:10"):
            with self.subTest(datasets=invalid):
                args.datasets = invalid
                with self.assertRaises(ValueError):
                    tune.input_tasks(args)

    def agent_state(self):
        settings = load_settings(smoke=True)
        config = default_config(self.task, settings)
        record = {"slot": 1, "phase": "baseline", "parameter": None,
                  "executed_config": config, "decision": None,
                  "result": {"status": "COMPLETE", "val_rmse": 1.0}}
        proposal = {"config": {**config, "lr": 0.003}, "source": "coordinate_probe",
                    "phase": "coordinate", "parameter": "lr", "probe_value": 0.003,
                    "base_slot": 1, "review_stage": "training_parameters"}
        state = get_search_state(self.task, settings, proposal, [record], 9)
        state["task"]["meta_features"] = load_features(self.features)["values"]
        return settings, state

    def test_llm_prompt_and_history_context_are_compact(self):
        _, state = self.agent_state()
        system = (ROOT / "tuning_prompt.txt").read_text(encoding="utf-8")
        self.assertLess(len(system), 2300)
        self.assertIn("completed validation history", system)
        self.assertEqual(len(state["history"]), 1)
        self.assertEqual(state["history"][0]["slot"], 1)
        self.assertEqual(state["history"][0]["val_rmse"], 1.0)
        for redundant in ("module_notes", "top_trials", "recent_trials", "previous_agent", "failures"):
            self.assertNotIn(redundant, state)

    def test_truncation_is_reported_and_fallback_is_explicit(self):
        settings, state = self.agent_state()
        agent = tune.TuningAgent(settings, mock=True)
        response = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
        folder = self.directory / "agent"
        with patch.object(agent, "call", return_value=response) as call:
            decision = agent.decide(state, self.task, ROOT, folder, set())
        self.assertEqual(call.call_count, 2)
        self.assertTrue(decision["fallback"])
        self.assertIn("truncated", decision["fallback_reason"])
        self.assertEqual(decision["executed_config"], state["coordinate_proposal"])
        self.assertEqual(read_json(folder / "agent.json")["agent"]["attempts"][0]["validation_error"]["finish_reason"], "length")
        with patch.object(agent, "call", side_effect=AssertionError("An existing record must not call the LLM")), \
             self.assertRaisesRegex(ValueError, "already exists"):
            agent.decide(state, self.task, ROOT, folder, set())

    def test_api_failure_is_recorded_without_saving_the_secret(self):
        settings, state = self.agent_state()
        agent = tune.TuningAgent(settings, api_key="unit-placeholder-secret")
        folder = self.directory / "agent"
        with patch.object(agent, "call", side_effect=RuntimeError("unit-placeholder-secret")):
            with self.assertRaises(AgentAPIError):
                agent.decide(state, self.task, ROOT, folder, set())
        first_attempt = read_json(folder / "agent.json")["agent"]["attempts"][0]
        self.assertEqual(first_attempt["transport"], {"mode": "direct", "trust_env": False, "proxy": None})
        self.assertEqual(first_attempt["api_error"]["chain"][0]["message"], "[REDACTED]")
        with self.assertRaisesRegex(ValueError, "already exists"):
            agent.decide(state, self.task, ROOT, folder, set())
        for path in folder.rglob("*"):
            if path.is_file():
                self.assertNotIn("unit-placeholder-secret", path.read_text(encoding="utf-8"))

    def test_top3_confirmation_reuses_three_seed_models_and_exports_recommendation(self):
        original_fit = tune.fit
        with patch("tune.fit", wraps=original_fit) as calls, \
             patch("hpo.retrain.fit", side_effect=AssertionError("Top-3 models must not be trained twice")):
            self.invoke(["--trials", "3"])
        confirmations = [call for call in calls.call_args_list
                         if call.kwargs["purpose"] == retrain.TOP3_PURPOSE]
        self.assertEqual([call.kwargs["seed"] for call in confirmations], [2028] * 6)
        with tune.summary_path(self.run).open(encoding="utf-8-sig", newline="") as handle:
            summaries = {int(row["horizon"]): row for row in csv.DictReader(handle)}
        for folder in self.run.glob("tasks/*/h*"):
            selection = read_json(folder / "best/selection.json")
            config = read_json(folder / "best/config.json")
            self.assertEqual(config["hyperparameters"], selection["config"])
            self.assertEqual(config["training"]["final_seeds"], [2026, 2027, 2028])
            self.assertEqual(config["initialization"], "fresh_for_each_seed")
            self.assertFalse(config["test_used_for_selection"])
            self.assertEqual(set(selection["validation_rmse_by_seed"]), {"2026", "2027", "2028"})
            with (folder / "best/seed_results.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["seed"]) for row in rows], [2026, 2027, 2028])
            summary = summaries[int(folder.name[1:])]
            self.assertEqual(summary["best_seed_count"], "3")
            for row in rows:
                seed = row["seed"]
                self.assertTrue((folder / row["checkpoint"]).exists())
                score = read_json(folder / f"best/seed_{seed}/test/metrics.json")
                for metric in ("rmse", "mae", "wmape"):
                    self.assertEqual(float(row[f"test_{metric}"]), score["overall_metrics"][metric])
            for metric in ("rmse", "mae", "wmape"):
                values = [float(row[f"test_{metric}"]) for row in rows]
                self.assertAlmostEqual(float(summary[f"best_test_{metric}_mean"]), float(np.mean(values)))
                self.assertAlmostEqual(float(summary[f"best_test_{metric}_std"]), float(np.std(values, ddof=1)))
            self.assertTrue((folder / "recommended/recommended_model.py").exists())
            params = (folder / "recommended/recommended_params.txt").read_text(encoding="utf-8-sig")
            self.assertIn("Recommendation source:", params)

    def test_failed_top3_confirmation_stops_before_test_and_summary(self):
        original_fit = tune.fit
        def fail_second_seed(*args, **kwargs):
            if kwargs["purpose"] == retrain.TOP3_PURPOSE and kwargs["seed"] == 2028:
                with patch("hpo.training.train_val_loaders", side_effect=RuntimeError("synthetic training failure")):
                    return original_fit(*args, **kwargs)
            return original_fit(*args, **kwargs)
        with patch("tune.fit", side_effect=fail_second_seed), self.assertRaisesRegex(RuntimeError, r"Top-4 \+ default confirmation training failed"):
            self.invoke(["--trials", "1", "--horizon", "3"])
        folder = next(self.run.glob("tasks/*/h3"))
        self.assertEqual(read_json(folder / "top3/slot_001/seed_2028/record.json")["training"]["attempts"][-1]["result"]["status"], "FAIL")
        self.assertFalse((folder / "best/selection.json").exists())
        self.assertEqual(list(self.run.glob("summary*.csv")), [])

    def test_report_checks_retraining_configuration_metrics_and_weights(self):
        self.invoke(["--trials", "1", "--horizon", "3"])
        folder = next(self.run.glob("tasks/*/h3"))
        summary = tune.summary_path(self.run)
        original_summary = summary.read_bytes()
        config_path = folder / "best/config.json"
        metrics_path = folder / "best/seed_2028/test/metrics.json"
        for path, field, value in ((config_path, "selected_slot", 999), (metrics_path, "seed", 999)):
            content = path.read_bytes()
            modified = read_json(path)
            modified[field] = value
            save_json(path, modified)
            with self.assertRaisesRegex(ValueError, "configuration|cache"):
                tune.refresh_summary(self.run)
            self.assertEqual(summary.read_bytes(), original_summary)
            path.write_bytes(content)

    def test_formal_top3_confirmation_uses_300_15_for_supplemental_seeds(self):
        settings = legacy_screening_settings()
        config = default_config(self.task, settings)
        records = [{"slot": slot, "executed_config": config, "fit_folder": f"trials/trial_{slot:03d}/training",
                    "result": {"status": "COMPLETE", "seed": 2026, "val_rmse": float(slot), "config": config}}
                   for slot in (1, 2, 3)]
        def result_for_seed(*args, **kwargs):
            return {"status": "COMPLETE", "val_rmse": 1.0,
                    "seed": kwargs["seed"], "config": config}
        with patch("tune.fit", side_effect=result_for_seed) as calls:
            tune.confirm_top_three(self.task, settings, self.directory / "formal_limits", records)
        self.assertEqual([call.kwargs["seed"] for call in calls.call_args_list], [2027, 2028] * 3)
        for call in calls.call_args_list:
            self.assertEqual((call.kwargs["epochs"], call.kwargs["patience"]), (300, 15))
            self.assertEqual(call.args[1], config)

    def confirmation_records(self, folder, scores):
        settings = legacy_screening_settings()
        defaults = default_config(self.task, settings)
        records = []
        for slot, (value, lr) in enumerate(zip(scores, (0.001, 0.0003, 0.003, 0.005)), 1):
            config = {**defaults, "lr": lr}
            fit_folder = f"trials/trial_{slot:03d}/training"
            directory = folder / fit_folder
            directory.mkdir(parents=True)
            artifacts = {"candidate_model": "candidate_model.py", "checkpoint": "best_model_weights.pth",
                         "stats": "train_standardization_stats.npz"}
            for name, filename in artifacts.items():
                (directory / filename).write_bytes(f"slot {slot} {name}".encode())
            result = {"status": "COMPLETE", "seed": 2026, "val_rmse": value, "config": config,
                      "params": self.task["original_params"], **artifacts}
            records.append({"slot": slot, "executed_config": config, "fit_folder": fit_folder,
                            "result": result})
        save_json(folder / "coordinate_trace.json", {"complete": True, "baseline_slot": 1})
        return settings, records

    def test_confirmation_adds_default_outside_top3_and_deduplicates_default_inside(self):
        for scores, expected_top, expected_slots in (((4.0, 1.0, 2.0, 3.0), [2, 3, 4], [2, 3, 4, 1]),
                                                     ((1.0, 2.0, 3.0, 4.0), [1, 2, 3], [1, 2, 3])):
            with self.subTest(default_in_top3=1 in expected_top):
                folder = self.directory / f"confirm_{expected_top[0]}"
                settings, records = self.confirmation_records(folder, scores)
                def result_for_seed(*args, **kwargs):
                    return {"status": "COMPLETE", "seed": kwargs["seed"], "val_rmse": 2.0,
                            "config": args[1]}
                with patch("tune.fit", side_effect=result_for_seed) as calls:
                    confirmation = tune.confirm_top_three(self.task, settings, folder, records)
                self.assertEqual(confirmation["top_slots"], expected_top)
                self.assertEqual(confirmation["confirmation_slots"], expected_slots)
                self.assertEqual(confirmation["default_slot"], 1)
                self.assertEqual(calls.call_count, 2 * len(expected_slots))
                observed = [(int(call.args[4].parents[1].name.removeprefix("slot_")), call.kwargs["seed"])
                            for call in calls.call_args_list]
                self.assertEqual(observed, [(slot, seed) for slot in expected_slots for seed in (2027, 2028)])
                self.assertEqual({row["slot"] for row in confirmation["ranking"]}, set(expected_slots))
                for call in calls.call_args_list:
                    self.assertEqual((call.kwargs["epochs"], call.kwargs["patience"]), (300, 15))

    def test_default_outside_top3_can_win_confirmation_and_export(self):
        folder = self.directory / "default_wins"
        settings, records = self.confirmation_records(folder, (4.0, 1.0, 2.0, 3.0))
        defaults = records[0]["executed_config"]
        def result_for_seed(*args, **kwargs):
            return {"status": "COMPLETE", "seed": kwargs["seed"], "val_rmse": 0.1 if args[1] == defaults else 10.0,
                    "config": args[1]}
        row = {"dataset": self.task["dataset"], "horizon": self.task["horizon"], "status": "COMPLETE"}
        with patch("tune.fit", side_effect=result_for_seed), patch("tune.finish_task", return_value=row) as finish:
            tune.finish_dataset(self.run, [(self.task, folder, records)], {self.task["id"]: {}}, settings)
        confirmation = read_json(folder / "top3/confirmation.json")
        self.assertNotIn(1, confirmation["top_slots"])
        self.assertEqual(confirmation["winner_slot"], 1)
        self.assertEqual(read_json(folder / "best/selection.json")["slot"], 1)
        exported = read_json(folder / "best/config.json")
        self.assertEqual(exported["selected_slot"], 1)
        self.assertEqual(exported["hyperparameters"], defaults)
        self.assertFalse(exported["test_used_for_selection"])
        for key, filename in (("candidate_model", "candidate_model.py"), ("checkpoint", "best_model_weights.pth"),
                              ("stats", "train_standardization_stats.npz")):
            self.assertEqual((folder / "best" / filename).read_bytes(), f"slot 1 {key}".encode())
        self.assertEqual(finish.call_args.args[5]["slot"], 1)
        self.assertEqual(set(finish.call_args.args[5]["seed_sources"]), {"2026", "2027", "2028"})

    def test_failed_extra_default_confirmation_blocks_selection_and_test(self):
        folder = self.directory / "default_fails"
        settings, records = self.confirmation_records(folder, (4.0, 1.0, 2.0, 3.0))
        defaults = records[0]["executed_config"]
        def result_for_seed(*args, **kwargs):
            return {"status": "FAIL" if args[1] == defaults else "COMPLETE", "seed": kwargs["seed"],
                    "val_rmse": 2.0, "config": args[1]}
        with patch("tune.fit", side_effect=result_for_seed) as calls, \
             patch("tune.evaluate_test", side_effect=AssertionError("Failed default confirmation must prevent tests")), \
             patch("tune.finish_task", side_effect=AssertionError("Failed default confirmation must prevent export")), \
             self.assertRaisesRegex(RuntimeError, r"confirmation training failed: .*slot1/seed2027"):
            tune.finish_dataset(self.run, [(self.task, folder, records)], {self.task["id"]: {}}, settings)
        self.assertEqual(calls.call_count, 7)
        self.assertFalse((folder / "top3/confirmation.json").exists())
        self.assertFalse((folder / "best/selection.json").exists())
        self.assertFalse((self.run / "selection_freeze.json").exists())
        self.assertEqual(list(self.run.glob("summary*.csv")), [])

    def test_v12_protocol_and_legacy_report_compatibility_without_reconfirmation(self):
        self.invoke(["--trials", "1", "--horizon", "3"])
        path = self.run / "tuning_protocol.json"
        protocol = read_json(path)
        self.assertEqual(protocol["artifact_format"], "balanced_dynamic_input_window_fixed_loss_dual_seed_top4_default_trial_v12")
        self.assertIn("balanced-dynamic-input-window-fixed-loss-dual-seed", protocol["method"])
        self.assertIn("default-confirmation", protocol["method"])
        self.assertTrue(protocol["method"].endswith("v12"))
        protocol["artifact_format"] = "joint_crossover_local_top3_trial_v8"
        protocol["method"] = "fixed-1-15-10-6-coverage-balanced-joint-crossover-local-top3-agent-v8"
        save_json(path, protocol)
        folder = next(self.run.glob("tasks/*/h3"))
        confirmation_path = folder / "top3/confirmation.json"
        confirmation = read_json(confirmation_path)
        confirmation.pop("confirmation_slots")
        confirmation.pop("default_slot")
        save_json(confirmation_path, confirmation)
        legacy_summary = self.run / "summary.csv"
        tune.summary_path(self.run).rename(legacy_summary)
        original = legacy_summary.read_bytes()
        with patch("tune.confirm_top_three", side_effect=AssertionError("Legacy reports must not reselect")), \
             patch("tune.fit", side_effect=AssertionError("Legacy reports must not train")), \
             patch("tune.evaluate_test", side_effect=AssertionError("Legacy reports must not evaluate tests")):
            rows = tune.refresh_summary(self.run)
        self.assertEqual(len(rows), 1)
        self.assertEqual(tune.summary_path(self.run), legacy_summary)
        self.assertEqual(list(self.run.glob("summary*.csv")), [legacy_summary])
        self.assertEqual(legacy_summary.read_bytes(), original)

    def test_old_run_is_not_reused(self):
        save_json(self.run / "protocol.json", {"legacy": True})
        with self.assertRaisesRegex(ValueError, "legacy 00-05"):
            tune.freeze_run(self.run, [], {}, load_settings(), False)
        self.assertEqual(read_json(self.run / "protocol.json"), {"legacy": True})


if __name__ == "__main__":
    unittest.main()
