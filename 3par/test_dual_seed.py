"""Dual-seed screening eligibility, confirmation, and frozen artifact reuse."""

import copy
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tune
import hpo.retrain as retrain
from hpo.artifacts import read_json, save_json
from hpo.config import ROOT, load_settings, tasks_from_manifest
from hpo.coordinator import Coordinator, record_complete, record_score
from hpo.models import default_config


class DualSeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".dual_seed_test_", dir=ROOT)
        self.directory = Path(self.temp.name)
        self.run = self.directory / "run"
        self.task = tasks_from_manifest()[0]
        self.settings = load_settings()

    def tearDown(self):
        self.assertTrue(self.directory.resolve().is_relative_to(ROOT.resolve()))
        self.temp.cleanup()

    def fit_stub(self, scores=None):
        """Write identifiable seed artifacts using the real training record layout."""
        scores = scores or {}

        def execute(task, config, settings, root, fit_folder, *, seed, epochs, patience, purpose, record_path):
            fit_folder = Path(fit_folder)
            fit_folder.mkdir(parents=True, exist_ok=True)
            score = scores.get(seed, 1.0)
            if callable(score):
                score = score(config)
            result = {"status": "FAIL" if score is None else "COMPLETE", "seed": seed,
                      "val_rmse": score, "config": dict(config), "params": task["original_params"]}
            if score is not None:
                for key, filename in (("candidate_model", "candidate_model.py"),
                                      ("checkpoint", "best_model_weights.pth"),
                                      ("stats", "train_standardization_stats.npz")):
                    (fit_folder / filename).write_bytes(f"{task['id']} seed {seed} {key}".encode())
                    result[key] = filename
            save_json(record_path, {"training": {"attempts": [{"result": result}]}})
            return result

        return execute

    def records(self, folder, score_pairs, task=None):
        task = task or self.task
        defaults = default_config(task, self.settings)
        lrs = (0.001, 0.0001, 0.0003, 0.003, 0.005)
        records = []
        for slot, (scores, lr) in enumerate(zip(score_pairs, lrs), 1):
            config = {**defaults, "lr": lr}
            trial = folder / "trials" / f"trial_{slot:03d}"
            phase = "baseline" if slot == 1 else "coordinate"
            proposal = {"phase": phase}
            with patch("tune.fit", side_effect=self.fit_stub(dict(zip((2026, 2027), scores)))):
                entries, aggregate = tune.fit_screening(task, config, self.settings, folder, trial, proposal)
            record = {"slot": slot, "executed_config": config, "proposal_config": config,
                      "phase": phase, "search_round": "baseline" if slot == 1 else "global",
                      "parameter": None, "decision": None, "base_slot": None, "parent_slots": [],
                      "execution_source": "default" if slot == 1 else "joint_global",
                      "result": entries["2026"]["result"], "screening_results": entries,
                      "fit_folder": str((trial / "training").relative_to(folder)), **aggregate}
            saved = read_json(trial / "trial.json")
            save_json(trial / "trial.json", {**saved, "record": record})
            records.append(record)
        save_json(folder / "coordinate_trace.json", {"complete": True, "baseline_slot": 1})
        return records

    def test_mean_ranking_rejects_a_lucky_first_seed_without_relabeling_checkpoint(self):
        folder = self.directory / "mean_ranking"
        records = self.records(folder, ((1.0, 1.0), (0.05, 3.95), (0.9, 0.9)))
        lucky = records[1]
        self.assertEqual(lucky["result"]["seed"], 2026)
        self.assertEqual(lucky["result"]["val_rmse"], 0.05)
        self.assertEqual(record_score(lucky), 2.0)
        self.assertEqual(lucky["screening_val_rmse_std"], statistics.stdev((0.05, 3.95)))
        coordinator = Coordinator(self.task, self.settings, ROOT)
        coordinator.observe(records[0])
        coordinator.observe(lucky)
        self.assertEqual(coordinator.incumbent["slot"], 1)
        coordinator.observe(records[2])
        self.assertEqual(coordinator.incumbent["slot"], 3)
        self.assertEqual([record["slot"] for record in coordinator._global_elites()], [3, 1, 2])
        with patch("tune.fit", side_effect=self.fit_stub()) as fits:
            confirmation = tune.confirm_top_three(self.task, self.settings, folder, records)
        self.assertEqual(confirmation["top_slots"], [3, 1, 2])
        self.assertEqual([call.kwargs["seed"] for call in fits.call_args_list], [2028] * 3)

    def test_second_seed_failure_is_ineligible_for_incumbent_parents_and_confirmation(self):
        folder = self.directory / "partial_failure"
        records = self.records(folder, ((1.0, 1.0), (0.001, None), (0.9, 0.9)))
        failed = records[1]
        self.assertEqual(failed["result"]["status"], "COMPLETE")
        self.assertEqual(failed["screening_results"]["2027"]["result"]["status"], "FAIL")
        self.assertFalse(record_complete(failed))
        self.assertIsNone(record_score(failed))
        coordinator = Coordinator(self.task, self.settings, ROOT)
        for record in records:
            coordinator.observe(record)
        self.assertEqual(coordinator.incumbent["slot"], 3)
        self.assertEqual(coordinator.decisions[0]["reason"], "candidate_failed")
        for elites in (coordinator._global_elites(), coordinator._crossover_elites(), coordinator._local_elites()):
            self.assertNotIn(2, [record["slot"] for record in elites])
        for _, parents in coordinator._crossover_candidates():
            self.assertNotIn(2, parents)
        for _, base_slot, _, _ in coordinator._local_candidates(coordinator.parameters):
            self.assertNotEqual(base_slot, 2)
        with patch("tune.fit", side_effect=self.fit_stub()) as fits:
            confirmation = tune.confirm_top_three(self.task, self.settings, folder, records)
        self.assertEqual(confirmation["confirmation_slots"], [3, 1])
        self.assertEqual(fits.call_count, 2)

    def test_top4_plus_default_deduplicates_and_trains_only_the_third_seed(self):
        cases = [((5, 1, 2, 3, 4), [2, 3, 4, 5], [2, 3, 4, 5, 1]),
                 ((1, 2, 3, 4, 5), [1, 2, 3, 4], [1, 2, 3, 4])]
        for index, (scores, expected_top, expected_candidates) in enumerate(cases):
            with self.subTest(default_in_top4=1 in expected_top):
                folder = self.directory / f"confirmation_{index}"
                records = self.records(folder, [(score, score) for score in scores])
                with patch("tune.fit", side_effect=self.fit_stub()) as fits:
                    confirmation = tune.confirm_top_three(self.task, self.settings, folder, records)
                self.assertEqual(confirmation["top_slots"], expected_top)
                self.assertEqual(confirmation["confirmation_slots"], expected_candidates)
                self.assertEqual(fits.call_count, len(expected_candidates))
                for call in fits.call_args_list:
                    self.assertEqual(call.kwargs["seed"], 2028)
                    self.assertEqual((call.kwargs["epochs"], call.kwargs["patience"]), (300, 15))
                self.assertFalse(list((folder / "top3").glob("*/seed_2027")))
                for ranking in confirmation["ranking"]:
                    record = records[ranking["slot"] - 1]
                    source = ranking["seed_sources"]["2027"]
                    self.assertEqual(source["kind"], "screening")
                    self.assertEqual(source["record"], record["screening_results"]["2027"]["record"])
                    self.assertEqual(source["fit_folder"], record["screening_results"]["2027"]["fit_folder"])

    def test_all_horizons_are_frozen_before_tests_and_screening_artifacts_are_reused(self):
        tasks = [copy.deepcopy(self.task), copy.deepcopy(self.task)]
        tasks[0].update(id="dual_seed__h3", horizon=3)
        tasks[1].update(id="dual_seed__h12", horizon=12)
        completed = []
        for task in tasks:
            folder = tune.task_folder(self.run, task)
            records = self.records(folder, ((2.0, 2.0), (1.0, 1.0)), task)
            completed.append((task, folder, records))
        evaluated = []

        def evaluate_frozen(task, config, settings, root, fit_folder, result, output, seed):
            frozen = read_json(self.run / "selection_freeze.json")
            self.assertEqual(set(frozen["task_ids"]), {item["id"] for item in tasks})
            for _, folder, _ in completed:
                self.assertTrue((folder / "best/config.json").is_file())
                self.assertTrue((folder / "best/selection.json").is_file())
            expected_fragment = (f"trials/trial_002/seed_{seed}/training" if seed == 2027
                                 else "trials/trial_002/training" if seed == 2026
                                 else "top3/slot_002/seed_2028/training")
            self.assertTrue(Path(fit_folder).as_posix().endswith(expected_fragment))
            self.assertEqual((fit_folder / result["checkpoint"]).read_bytes(),
                             f"{task['id']} seed {seed} checkpoint".encode())
            evaluated.append((task["id"], seed))
            return {"overall_metrics": {"rmse": 1.0, "mae": 0.8, "wmape": 0.1}}

        def finish(task, features, settings, folder, baseline, best, records):
            self.assertEqual(best["slot"], 2)
            rows = retrain.retrain_best(task, settings, ROOT, folder, best)
            self.assertEqual([row["status"] for row in rows], ["COMPLETE"] * 3)
            return {"dataset": task["dataset"], "horizon": task["horizon"], "status": "COMPLETE"}

        with patch("tune.fit", side_effect=self.fit_stub()) as confirmation_fits, \
             patch("tune.evaluate_test", side_effect=AssertionError("Selection must never evaluate test data")), \
             patch("tune.finish_task", side_effect=finish), \
             patch("hpo.retrain.fit", side_effect=AssertionError("Confirmed seed models must be reused")), \
             patch("hpo.retrain.read_test_cache", return_value=None), \
             patch("hpo.retrain.evaluate_test", side_effect=evaluate_frozen):
            tune.finish_dataset(self.run, completed, {task["id"]: {} for task in tasks}, self.settings)
        self.assertEqual([call.kwargs["seed"] for call in confirmation_fits.call_args_list], [2028] * 4)
        self.assertEqual(evaluated, [(task["id"], seed) for task in tasks for seed in (2026, 2027, 2028)])


if __name__ == "__main__":
    unittest.main()
