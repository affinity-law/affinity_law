"""Fixed default loss for new runs, with explicit historical loss-choice compatibility."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import tune
from hpo.artifacts import read_json
from hpo.config import ROOT, load_settings, tasks_from_manifest
from hpo.coordinator import Coordinator
from hpo.losses import LOSS_PROFILES, loss_details
from hpo.models import active_space, candidate_code, default_config, validate_config
from hpo.training import evaluate, fit, loss_value, training_request


class LossChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = tasks_from_manifest()[0]
        cls.settings = load_settings(smoke=True)
        # These tests deliberately recreate the older tunable-loss protocol.
        cls.settings["space"]["loss_profile"] = list(LOSS_PROFILES)
        cls.settings["search"]["probe_values"]["loss_profile"] = ["balanced", "mse"]
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".loss_test_", dir=ROOT)
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.assertTrue(self.folder.resolve().is_relative_to(ROOT.resolve()))
        self.temp.cleanup()

    def test_profiles_change_objective_and_gradients_without_changing_rmse(self):
        expected_losses = {"mae_heavy": 2.6, "balanced": 3.5, "mse": 5.0}
        expected_gradients = {"mae_heavy": [0.6, 1.0], "balanced": [0.75, 1.75],
                              "mse": [1.0, 3.0]}
        metrics = []
        for profile in LOSS_PROFILES:
            with self.subTest(profile=profile):
                prediction = torch.tensor([1.0, 3.0], requires_grad=True)
                config = {"loss_profile": profile}
                objective = loss_value(prediction, torch.zeros_like(prediction), self.settings, config)
                self.assertAlmostEqual(objective.item(), expected_losses[profile], places=6)
                objective.backward()
                torch.testing.assert_close(prediction.grad, torch.tensor(expected_gradients[profile]))
                x = prediction.detach().reshape(2, 1, 1)
                result = evaluate(torch.nn.Identity(), [(x, torch.zeros_like(x))], self.settings,
                                  torch.device("cpu"), config=config)
                self.assertAlmostEqual(result["val_loss"], expected_losses[profile], places=6)
                metrics.append(result["val_rmse"])
        self.assertEqual(metrics, [5 ** 0.5] * 3)

    def test_loss_is_an_explicit_config_field_and_never_changes_r_source(self):
        default = default_config(self.task, self.settings)
        original = candidate_code(self.task, default, ROOT)
        self.assertEqual(default["loss_profile"], "mae_heavy")
        self.assertEqual(active_space(self.task, self.settings)["loss_profile"], list(LOSS_PROFILES))
        configurations = set()
        for profile in LOSS_PROFILES:
            config = validate_config({**default, "loss_profile": profile}, self.task, self.settings)
            configurations.add(json.dumps(config, sort_keys=True))
            self.assertEqual(candidate_code(self.task, config, ROOT), original)
        self.assertEqual(len(configurations), 3)
        with self.assertRaisesRegex(ValueError, "loss_profile"):
            validate_config({**default, "loss_profile": "arbitrary"}, self.task, self.settings)

    def test_each_profile_trains_and_records_the_selected_loss(self):
        for profile in LOSS_PROFILES:
            with self.subTest(profile=profile):
                config = {**default_config(self.task, self.settings), "loss_profile": profile}
                folder = self.folder / profile
                result = fit(self.task, config, self.settings, ROOT, folder, seed=2026,
                             epochs=1, patience=1, purpose="unit_loss_smoke")
                self.assertEqual(result["status"], "COMPLETE", result.get("error"))
                details = loss_details(config)
                for key, value in details.items():
                    self.assertEqual(result[key], value)
                request = read_json(folder / "training.json")["training"]["request"]
                self.assertEqual(request["loss"], details)
                self.assertEqual(result["checkpoint_value"], result["val_loss"])

    def test_selected_loss_still_controls_checkpoint_and_early_stopping(self):
        best = {"val_loss": 1.0, "val_mae": 1.0, "val_rmse": 9.0}
        worse = {"val_loss": 2.0, "val_mae": 1.0, "val_rmse": 1.0}
        for profile in ("balanced", "mse"):
            with self.subTest(profile=profile):
                config = {**default_config(self.task, self.settings), "loss_profile": profile}
                with patch("hpo.training.evaluate", side_effect=[best, worse, best]) as evaluation:
                    result = fit(self.task, config, self.settings, ROOT, self.folder / profile,
                                 seed=2026, epochs=10, patience=1, purpose="unit_loss_checkpoint")
                self.assertEqual(result["status"], "COMPLETE", result.get("error"))
                self.assertEqual((result["best_epoch"], result["epochs_run"]), (1, 2))
                self.assertEqual(result["val_rmse"], 9.0)
                self.assertEqual(result["checkpoint_metric"], loss_details(config)["checkpoint_metric"])
                self.assertTrue(all(call.kwargs["config"] == config for call in evaluation.call_args_list))

    def test_legacy_settings_config_and_request_remain_readable(self):
        legacy = load_settings()
        legacy["space"].pop("time_lag", None)
        legacy["search"]["probe_values"].pop("time_lag", None)
        legacy["search"].pop("screening_seeds")
        legacy["search"].pop("confirmation_top_k")
        legacy["search"]["confirmation_seeds"] = [2027, 2028]
        legacy["search"]["probe_values"].pop("loss_profile", None)
        legacy["space"].pop("loss_profile", None)
        path = self.folder / "legacy_settings.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        settings = load_settings(path, smoke=True)
        config = default_config(self.task, settings)
        self.assertNotIn("loss_profile", config)
        self.assertNotIn("loss_profile", active_space(self.task, settings))
        self.assertEqual(validate_config(config, self.task, settings), config)
        request = training_request(self.task, config, settings, 2026, 1, 1, "legacy")
        self.assertNotIn("loss", request)
        expected = {"task_id": self.task["id"], "config": config, "seed": 2026,
                    "epochs": 1, "patience": 1, "purpose": "legacy",
                    "training_settings": settings["training"], "smoke": True,
                    "smoke_prefix_points": 256, "train_ratio": 0.6, "val_ratio": 0.2}
        self.assertEqual(request, expected)
        result = fit(self.task, config, settings, ROOT, self.folder / "legacy", seed=2026,
                     epochs=1, patience=1, purpose="legacy")
        self.assertEqual(result["status"], "COMPLETE", result.get("error"))
        self.assertEqual(result["loss_profile"], "mae_heavy")
        self.assertEqual(result["checkpoint_metric"], "normalized_validation_0.2_mse_0.8_mae_loss")

    def test_new_settings_reject_invalid_losses_and_seed_protocols(self):
        settings = load_settings()
        self.assertEqual(settings["search"]["screening_seeds"], [2026, 2027])
        self.assertEqual(settings["search"]["confirmation_seeds"], [2028])
        self.assertEqual(settings["search"]["confirmation_top_k"], 4)
        path = self.folder / "invalid_settings.json"
        cases = [("space", "loss_profile", []), ("space", "loss_profile", ["mae_heavy", "unknown"]),
                 ("space", "loss_profile", ["mae_heavy", "mae_heavy"]),
                 ("space", "loss_profile", ["mse"]),
                 ("search", "screening_seeds", [2026]),
                 ("search", "confirmation_seeds", [2027, 2028]),
                 ("search", "confirmation_top_k", 3)]
        for section, field, value in cases:
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(settings)
                changed[section][field] = value
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_settings(path)

    def test_saved_loss_choice_settings_remain_readable_without_rewriting(self):
        legacy = load_settings()
        legacy["space"]["loss_profile"] = list(LOSS_PROFILES)
        legacy["search"]["probe_values"]["loss_profile"] = ["balanced", "mse"]
        path = self.folder / "saved_loss_choices.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = load_settings(path)
        self.assertEqual(loaded["space"]["loss_profile"], list(LOSS_PROFILES))
        config = {**default_config(self.task, loaded), "loss_profile": "mse"}
        self.assertEqual(validate_config(config, self.task, loaded), config)
        request = training_request(self.task, config, loaded, 2026, 300, 15, "historical")
        self.assertEqual(request["loss"]["loss_profile"], "mse")
        self.assertEqual((request["loss"]["mse_weight"], request["loss"]["mae_weight"]), (1.0, 0.0))

    def test_new_default_search_never_proposes_a_loss_hyperparameter(self):
        settings = load_settings(smoke=True)
        self.assertNotIn("loss_profile", settings["space"])
        self.assertNotIn("loss_profile", settings["search"]["probe_values"])
        for task in tasks_from_manifest():
            self.assertNotIn("loss_profile", active_space(task, settings))
            self.assertNotIn("loss_profile", default_config(task, settings))
        coordinator = Coordinator(self.task, settings, ROOT)
        proposed = 0
        with patch.object(coordinator, "_valid", return_value=True):
            while not coordinator.complete:
                slot = len(coordinator.records) + 1
                proposal = coordinator.propose(slot)
                if proposal is None:
                    break
                self.assertNotIn("loss_profile", proposal["config"])
                self.assertNotIn("loss_profile", proposal.get("editable_parameters", []))
                self.assertNotEqual(proposal.get("parameter"), "loss_profile")
                value = 1.0 - slot / 1000
                record = {key: proposal.get(key) for key in
                          ("phase", "search_round", "parameter", "probe_value", "base_slot", "parent_slots")}
                record.update(slot=slot, executed_config=proposal["config"], screening_status="COMPLETE",
                              screening_val_rmse_mean=value,
                              result={"status": "COMPLETE", "val_rmse": value, "seed": 2026,
                                      "params": self.task["original_params"]})
                coordinator.observe(record)
                proposed += 1
        self.assertEqual(proposed, 32)

    def test_new_default_rejects_injected_loss_fields_and_nondefault_weights(self):
        settings = load_settings()
        config = default_config(self.task, settings)
        for field, value in (("loss_profile", "mse"), ("mse_weight", 1.0), ("mae_weight", 1.0)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config({**config, field: value}, self.task, settings)
        path = self.folder / "nondefault_weights.json"
        for mse, mae in ((0.5, 0.5), (1.0, 0.0)):
            changed = copy.deepcopy(settings)
            changed["training"].update(mse_weight=mse, mae_weight=mae)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Default training loss weights"):
                load_settings(path)

    def test_new_screening_and_confirmation_cpu_training_use_the_default_loss(self):
        settings = load_settings(smoke=True)
        config = default_config(self.task, settings)
        trial = self.folder / "trials/trial_001"
        entries, screening = tune.fit_screening(self.task, config, settings, self.folder, trial,
                                               {"phase": "baseline"})
        self.assertEqual(screening["screening_status"], "COMPLETE")
        record = {"slot": 1, "executed_config": config, "result": entries["2026"]["result"],
                  "screening_results": entries, "fit_folder": entries["2026"]["fit_folder"], **screening}
        confirmation = tune.confirm_top_three(self.task, settings, self.folder, [record])
        self.assertEqual(confirmation["seeds"], [2026, 2027, 2028])
        confirmed = read_json(self.folder / "top3/slot_001/seed_2028/record.json")
        results = [entry["result"] for entry in entries.values()]
        results.append(confirmed["training"]["attempts"][-1]["result"])
        self.assertEqual([result["seed"] for result in results], [2026, 2027, 2028])
        for result in results:
            self.assertEqual(result["status"], "COMPLETE", result.get("error"))
            self.assertNotIn("loss_profile", result["config"])
            self.assertEqual((result["mse_weight"], result["mae_weight"]), (0.2, 0.8))
            self.assertEqual(result["loss_profile"], "mae_heavy")
            self.assertEqual(result["checkpoint_metric"], "normalized_validation_0.2_mse_0.8_mae_loss")
            expected = 0.2 * result["val_normalized_rmse"] ** 2 + 0.8 * result["val_mae"]
            self.assertTrue(math.isclose(result["val_loss"], expected, rel_tol=1e-5, abs_tol=1e-7))
            self.assertEqual(result["checkpoint_value"], result["val_loss"])

    def test_new_default_checkpoint_and_early_stop_keep_the_original_loss_rule(self):
        settings = load_settings(smoke=True)
        config = default_config(self.task, settings)
        best = {"val_loss": 1.0, "val_mae": 1.0, "val_rmse": 9.0}
        worse = {"val_loss": 2.0, "val_mae": 1.0, "val_rmse": 1.0}
        with patch("hpo.training.evaluate", side_effect=[best, worse, best]):
            result = fit(self.task, config, settings, ROOT, self.folder / "default_checkpoint",
                         seed=2026, epochs=10, patience=1, purpose="unit_fixed_default_loss")
        self.assertEqual(result["status"], "COMPLETE", result.get("error"))
        self.assertEqual((result["best_epoch"], result["epochs_run"]), (1, 2))
        self.assertEqual(result["val_rmse"], 9.0)
        self.assertEqual(result["checkpoint_metric"], "normalized_validation_0.2_mse_0.8_mae_loss")


if __name__ == "__main__":
    unittest.main()
