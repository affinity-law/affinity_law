"""Core training, data isolation, search-space, and coordinate-search checks."""

import ast
import copy
import csv
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from hpo.agent import Agent, get_search_state, parse_decision, summary
from hpo.artifacts import read_json
from hpo.config import PROJECT_ROOT, ROOT, load_settings, require_llm, run_folder, tasks_from_manifest
from hpo.coordinator import TRAINING_PARAMETERS, Coordinator, coordinate_order, record_complete, record_score
from hpo.data import WindowDataset, split_bounds, train_val_loaders, validation_diagnostics
from hpo.evaluation import average_future_steps, metrics
from hpo.models import WIDTH_PARAMETERS, active_space, candidate_class, candidate_code, default_config, extract_spec, inspect_candidate, instantiate, validate_config
from hpo.training import fit, set_seed


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = tasks_from_manifest()
        cls.task = cls.tasks[0]
        cls.settings = load_settings(smoke=True)
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".unit_", dir=ROOT)
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.assertTrue(self.directory.resolve().is_relative_to(ROOT.resolve()))
        self.temp.cleanup()

    def config(self, task=None):
        return default_config(task or self.task, self.settings)

    def fake_record(self, slot, config):
        return {"slot": slot, "executed_config": config, "decision": None,
                "phase": "coordinate", "parameter": None,
                "result": {"status": "COMPLETE", "val_rmse": float(config["lr"] * 100 + slot / 1000),
                           "params": self.task["original_params"], "seed": 2026}}

    def test_prepared_scope_and_original_files(self):
        manifest = read_json(ROOT / "tasks.json")
        datasets = {p.stem: p for p in (PROJECT_ROOT / "0dataset/test").rglob("*.npy")
                    if p.stem in {task["dataset"] for task in self.tasks}}
        self.assertEqual(len(datasets), 63)
        self.assertEqual(len(self.tasks), manifest["task_count"])
        self.assertEqual(set(datasets), {task["dataset"] for task in self.tasks})
        self.assertEqual(len(datasets), manifest["dataset_count"])
        self.assertEqual(len(self.tasks), len(datasets) * len(self.settings["horizons"]))
        self.assertEqual(Counter((task["dataset"], task["horizon"]) for task in self.tasks),
                         Counter({(dataset, horizon): 1 for dataset in datasets
                                  for horizon in self.settings["horizons"]}))
        self.assertNotIn("0107_BDG2_Bobcat_education_Seth_9649_5", {task["dataset"] for task in self.tasks})
        self.assertEqual({task["horizon"] for task in self.tasks}, {3, 12})
        for task in self.tasks:
            dataset = datasets[task["dataset"]]
            self.assertEqual((ROOT / task["data_path"]).resolve(),
                             dataset.resolve())
            array = np.load(dataset, mmap_mode="r")
            self.assertEqual((task["length"], task["channels"]),
                             (len(array), array.shape[1] if array.ndim > 1 else 1))
            self.assertTrue((ROOT / task["model_path"]).is_file())
            self.assertTrue((ROOT / task["data_path"]).is_file())
            self.assertTrue((ROOT / task["weights_path"]).is_file())

    def test_default_source_and_parameter_count(self):
        for task in self.tasks:
            config = self.config(task)
            code = candidate_code(task, config, ROOT)
            self.assertEqual(code, (ROOT / task["model_path"]).read_text(encoding="utf-8-sig"))
            self.assertEqual(inspect_candidate(task, config, self.settings, ROOT)["params"], task["original_params"])

    def test_simplified_layout_and_model_metadata(self):
        for name in ("checks", "manifests", "models_r", "prompts", "reference", "runs", "tests"):
            self.assertFalse((ROOT / name).exists(), name)
        self.assertFalse((ROOT / "data").exists())
        self.assertTrue((PROJECT_ROOT / "0dataset/test").is_dir())
        self.assertTrue((PROJECT_ROOT / "2search/connect/document_summary.csv").is_file())
        self.assertTrue((ROOT / "tuning_prompt.txt").is_file())
        for task in self.tasks:
            model_folder = (PROJECT_ROOT / "2search/compar_result/models_L" / task["dataset"]
                            / f"h{task['horizon']}" / f"law_s{task['source_seed']}")
            for key in ("model_path", "weights_path"):
                self.assertTrue((ROOT / task[key]).resolve().is_relative_to(model_folder))
                self.assertTrue((ROOT / task[key]).is_file())
            self.assertEqual(extract_spec((ROOT / task["model_path"]).read_text(encoding="utf-8-sig")),
                             task["spec"])

    def test_run_output_cannot_shadow_check_reports(self):
        self.assertEqual(run_folder("smoke_example"), ROOT / "outputs" / "smoke_example")
        for name in ("checks", "CHECKS"):
            with self.assertRaises(ValueError):
                run_folder(name)

    def test_parameterization_preserves_macro_modules(self):
        task = next(task for task in self.tasks if "LSTM Block" in task["spec"]["B"])
        config = {**self.config(task), "lstm_width_scale": 0.75, "lstm_layers": 2}
        inspected = inspect_candidate(task, config, self.settings, ROOT, check_forward=True)
        self.assertEqual(extract_spec(inspected["code"]), task["spec"])
        self.assertNotEqual(inspected["params"], task["original_params"])

    def test_capacity_is_split_into_independent_module_parameters(self):
        spaces = [active_space(task, self.settings) for task in self.tasks]
        self.assertTrue(all("capacity_profile" not in space for space in spaces))
        for task, space in zip(self.tasks, spaces):
            spec = task["spec"]
            expected = (set(TRAINING_PARAMETERS) & set(self.settings["space"])) | {WIDTH_PARAMETERS[module][0]
                                                  for module in [*spec["B"], spec["C"]]
                                                  if module in WIDTH_PARAMETERS}
            if "time_lag" in self.settings["space"]:
                expected.add("time_lag")
            expected.update(name for name, active in (("lstm_layers", "LSTM Block" in spec["B"]),
                                                       ("tcn_kernel", "TCN Block" in spec["B"]),
                                                       ("patch_profile", "Patch" in spec["A"])) if active)
            self.assertEqual(set(space), expected, task["id"])
        self.assertTrue(all(len(task["spec"]["B"]) == len(set(task["spec"]["B"])) for task in self.tasks))
        task = next(task for task in self.tasks if len(task["spec"]["B"]) == 2
                    and all(module in WIDTH_PARAMETERS for module in task["spec"]["B"]))
        default = self.config(task)
        changed = dict(default)
        expected_factors = {}
        for module, factor in zip(task["spec"]["B"], (1.25, 0.75)):
            field, constructor, keyword = WIDTH_PARAMETERS[module]
            changed[field] = factor
            expected_factors[constructor] = (keyword, factor)
        def calls(config):
            tree = ast.parse(candidate_code(task, config, ROOT))
            return {node.func.id: {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
                    for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id in expected_factors}
        before, after = calls(default), calls(changed)
        for constructor, (keyword, factor) in expected_factors.items():
            self.assertEqual(after[constructor][keyword], max(1, round(before[constructor][keyword] * factor)))

    def test_each_active_width_can_change_without_changing_other_fields(self):
        for task in self.tasks:
            default = self.config(task)
            width_fields = [name for name in default if name.endswith("_width_scale")]
            for field in width_fields:
                with self.subTest(task=task["id"], field=field):
                    changed = {**default, field: 0.75}
                    inspected = inspect_candidate(task, changed, self.settings, ROOT)
                    self.assertNotEqual(inspected["code"],
                                        (ROOT / task["model_path"]).read_text(encoding="utf-8-sig"))
                    self.assertNotEqual(inspected["params"], task["original_params"])

    def test_inactive_and_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({**self.config(), "attention_heads": 8}, self.task, self.settings)
        with self.assertRaises(ValueError):
            validate_config({**self.config(), "lr": float("nan")}, self.task, self.settings)
        with self.assertRaises(ValueError):
            validate_config({**self.config(), "batch_size": True}, self.task, self.settings)
        canonical = validate_config({**self.config(), "weight_decay": 0}, self.task, self.settings)
        self.assertEqual(canonical, self.config())

    def test_preflight_does_not_consume_rng(self):
        torch.manual_seed(10)
        before = torch.get_rng_state().clone()
        inspect_candidate(self.task, self.config(), self.settings, ROOT, check_forward=True)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_windows_and_chronological_split(self):
        self.assertEqual(split_bounds(101, self.settings), (60, 80))
        windows = WindowDataset(np.arange(30)[:, None], 12, 3)
        self.assertEqual(len(windows), 16)
        x, y = windows[1]
        self.assertEqual(x[:, 0].tolist(), list(range(1, 13)))
        self.assertEqual(y[:, 0].tolist(), [13, 14, 15])

    def test_test_values_cannot_affect_train_or_validation(self):
        data = np.arange(500, dtype=np.float32)[:, None]
        path = self.directory / "data.npy"
        np.save(path, data)
        settings = load_settings()
        a, b, mean, std = train_val_loaders(path, self.task, settings, 16, 2026)
        data[400:] = np.nan
        np.save(path, data)
        c, d, mean2, std2 = train_val_loaders(path, self.task, settings, 16, 2026)
        np.testing.assert_array_equal(mean, mean2)
        np.testing.assert_array_equal(std, std2)
        self.assertTrue(torch.equal(a.dataset.data, c.dataset.data))
        self.assertTrue(torch.equal(b.dataset.data, d.dataset.data))

    def test_validation_signal_guard_rejects_constant_segment(self):
        data = np.arange(500, dtype=np.float32)[:, None]
        data[300:400] = data[299]
        path = self.directory / "constant_validation.npy"
        np.save(path, data)
        self.assertFalse(validation_diagnostics(path, load_settings())["tuning_allowed"])
        data[300:400] = np.arange(100, dtype=np.float32)[:, None]
        np.save(path, data)
        self.assertTrue(validation_diagnostics(path, load_settings())["tuning_allowed"])

    def test_search_ranges_are_wider_and_tpe_is_not_required(self):
        settings = load_settings()
        self.assertEqual(settings["space"]["width_scales"], [0.5, 0.75, 1.0, 1.25, 1.5])
        self.assertEqual(settings["space"]["batch_size"], [8, 16, 32, 64])
        self.assertFalse(settings["space"]["enforce_param_ratio"])
        self.assertNotIn("param_ratio_min", settings["space"])
        self.assertNotIn("param_ratio_max", settings["space"])
        self.assertNotIn("startup_trials", settings["search"])

    def test_metric_compatibility_with_old_pipeline(self):
        path = ROOT.parent / "1experiment" / "test_metrics_iterative.py"
        if not path.exists():
            self.skipTest("The standalone project does not include the earlier reference pipeline")
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"average_future_steps", "calc_basic_metrics"}]
        namespace = {"np": np}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        values = np.random.default_rng(7).normal(size=(20, 12, 4)).astype(np.float32)
        ours = average_future_steps(values)
        old = namespace["average_future_steps"](values)
        np.testing.assert_array_equal(ours, old)
        old_metrics = namespace["calc_basic_metrics"](ours, old + 1)
        for key, value in metrics(ours, old + 1).items():
            self.assertEqual(value, old_metrics[key])

    def test_training_preflight_consumes_rng_like_historical_r(self):
        task, settings = self.task, load_settings(smoke=True)
        config = default_config(task, settings)
        code = candidate_code(task, config, ROOT)
        batches = (1, 2, config["batch_size"])

        set_seed(2026)
        CandidateModel = candidate_class(code)
        for batch in batches:
            model = CandidateModel(12, task["horizon"], task["channels"])
            x = torch.randn(batch, 12, task["channels"])
            y = torch.randn(batch, task["horizon"], task["channels"])
            (model(x) - y).abs().mean().backward()
        expected = instantiate(code, task).state_dict()

        set_seed(2026)
        inspected = inspect_candidate(task, config, settings, ROOT, check_forward=True,
                                      batch_sizes=batches, preserve_rng=False)
        actual = instantiate(inspected["code"], task).state_dict()
        self.assertEqual(expected.keys(), actual.keys())
        self.assertTrue(all(torch.equal(expected[name], actual[name]) for name in expected))

    def synthetic_record(self, proposal, slot):
        value = 1.0 if proposal["phase"] == "baseline" else 1.0 - slot / 1000
        return {"slot": slot, "executed_config": proposal["config"], "decision": None,
                "phase": proposal["phase"], "search_round": proposal["search_round"],
                "parameter": proposal.get("parameter"), "probe_value": proposal.get("probe_value"),
                "base_slot": proposal.get("base_slot"), "parent_slots": proposal.get("parent_slots", []),
                "search_priority": proposal.get("search_priority"),
                "effective_priority": proposal.get("effective_priority"),
                "parent_pool_slots": proposal.get("parent_pool_slots", []),
                "parent_pool_size": proposal.get("parent_pool_size", 0),
                "parent_pool_expanded": proposal.get("parent_pool_expanded", False),
                "result": {"status": "COMPLETE", "val_rmse": value,
                           "params": self.task["original_params"], "seed": proposal["seed"]}}

    def test_search_uses_fixed_stage_allocation_and_joint_global_candidates(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        proposals = []
        while not coordinator.complete:
            slot = len(coordinator.records) + 1
            proposal = coordinator.propose(slot)
            if proposal is None:
                self.assertTrue(coordinator.complete)
                break
            proposals.append(proposal)
            coordinator.observe(self.synthetic_record(proposal, slot))
        trace = coordinator.trace()
        self.assertEqual(trace["parameter_order"], coordinate_order(self.task, self.settings))
        self.assertEqual(trace["stage_slot_plan"], {"baseline": 1, "global": 16,
                                                    "crossover": 9, "local": 6})
        for stage, budget in trace["stage_slot_plan"].items():
            self.assertLessEqual(trace["completed_by_stage"][stage], budget)
            if trace["completed_by_stage"][stage] < budget:
                self.assertIn(stage, trace["exhausted_stages"])
        self.assertEqual(trace["completed_by_stage"]["global"], 16)
        self.assertEqual(trace["completed_by_stage"]["local"], 6)
        self.assertLessEqual(trace["completed_slots"], 32)
        self.assertTrue(trace["coordinate_decisions"])
        self.assertTrue(any(item["accepted"] for item in trace["coordinate_decisions"]))
        self.assertEqual(trace["confirmation_slots"], [])
        self.assertEqual(trace["confirmation_strategy"], "top4_plus_default_seed2028_then_mean_2026_2027_2028")
        self.assertEqual(trace["model_priority_slots"], {"global": 8, "local": 3})
        self.assertEqual(trace["training_priority_slots"], {"global": 8, "local": 3})
        global_proposals = [proposal for proposal in proposals if proposal["search_round"] == "global"]
        training = set(coordinator.training_parameters)
        model = set(coordinator.parameters) - training
        for index, proposal in enumerate(global_proposals):
            changed = sum(proposal["config"][name] != coordinator.defaults[name]
                          for name in coordinator.parameters)
            self.assertGreaterEqual(changed, 2)
            self.assertLessEqual(changed, min(4, len(coordinator.parameters)))
            model_changes = sum(proposal["config"][name] != coordinator.defaults[name]
                                for name in model)
            training_changes = sum(proposal["config"][name] != coordinator.defaults[name]
                                   for name in training)
            self.assertEqual(proposal["search_priority"], "training" if index % 2 else "model")
            if index % 2 == 0:
                self.assertGreaterEqual(model_changes, training_changes)
                self.assertGreater(model_changes, 0)
            else:
                self.assertGreater(training_changes, model_changes)
                self.assertEqual(model_changes, 0)

    def test_search_accepts_any_strict_rmse_improvement_without_threshold(self):
        settings = copy.deepcopy(self.settings)
        settings["search"]["trials"] = 3
        coordinator = Coordinator(self.task, settings, ROOT)
        values = [1.0, 0.9999, 1.1]
        for slot, value in enumerate(values, 1):
            proposal = coordinator.propose(slot)
            record = self.synthetic_record(proposal, slot)
            record["result"]["val_rmse"] = value
            coordinator.observe(record)
        decision = coordinator.trace()["coordinate_decisions"][0]
        self.assertEqual(decision["selected_slot"], 2)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "lowest_validation_rmse")
        self.assertNotIn("required_improvement", decision)

    def test_current_search_fixes_loss_and_rejects_llm_loss_overrides(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        self.assertEqual(coordinator.training_parameters, ["lr", "weight_decay", "batch_size"])
        self.assertNotIn("loss_profile", coordinator.space)
        self.assertNotIn("loss_profile", coordinator.defaults)
        self.assertNotIn("loss_profile", self.settings["search"]["probe_values"])
        reviewed = []
        with patch.object(coordinator, "_valid", return_value=True):
            for slot in range(1, 18):
                proposal = coordinator.propose(slot)
                self.assertNotIn("loss_profile", proposal["config"])
                self.assertNotIn("loss_profile", proposal["editable_parameters"])
                if proposal.get("review_stage"):
                    reviewed.append(get_search_state(self.task, self.settings, proposal,
                                                     coordinator.records, slot))
                coordinator.observe(self.synthetic_record(proposal, slot))
        state = next(state for state in reviewed if state["review_stage"] == "training_parameters")
        protocol = state["training_protocol"]
        self.assertFalse(protocol["loss_tunable"])
        self.assertNotIn("loss_profiles", protocol)
        self.assertEqual(protocol["fixed_loss"],
                         {"loss_profile": "mae_heavy", "mse_weight": 0.2, "mae_weight": 0.8,
                          "checkpoint_metric": "normalized_validation_0.2_mse_0.8_mae_loss"})
        self.assertEqual(protocol["early_stop_metric"], protocol["fixed_loss"]["checkpoint_metric"])
        for parameter, value in (("loss_profile", "mse"), ("mse_weight", 1.0), ("mae_weight", 0.0)):
            with self.subTest(parameter=parameter):
                self.assertIn(parameter, protocol["fixed_parameters"])
                self.assertIn(parameter, state["protected_parameters"])
                response = {"action": "edit_proposal", "overrides": {parameter: value},
                            "evidence_trials": [1], "hypothesis": "Change the loss function."}
                with self.assertRaisesRegex(ValueError, "protected"):
                    parse_decision(json.dumps(response), state, self.task, self.settings, ROOT, set())

    def test_legacy_loss_space_is_described_from_its_frozen_choices(self):
        settings = copy.deepcopy(self.settings)
        settings["space"]["loss_profile"] = ["mae_heavy", "balanced", "mse"]
        settings["search"]["probe_values"]["loss_profile"] = ["balanced", "mse"]
        coordinator = Coordinator(self.task, settings, ROOT)
        proposal = coordinator.propose(1)
        proposal["config"] = {**proposal["config"], "loss_profile": "balanced"}
        state = get_search_state(self.task, settings, proposal, [], 1)
        protocol = state["training_protocol"]
        self.assertTrue(protocol["loss_tunable"])
        self.assertNotIn("fixed_loss", protocol)
        self.assertEqual(set(protocol["loss_profiles"]), {"mae_heavy", "balanced", "mse"})
        self.assertEqual(protocol["selected_loss"]["mse_weight"], 0.5)
        self.assertEqual(protocol["selected_loss"]["mae_weight"], 0.5)
        self.assertEqual(protocol["early_stop_metric"], "normalized_validation_0.5_mse_0.5_mae_loss")

    def test_input_window_is_a_model_coordinate_and_both_long_windows_are_explored(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        self.assertIn("time_lag", coordinator.model_parameters)
        self.assertNotIn("time_lag", coordinator.training_parameters)
        self.assertEqual(coordinator.defaults["time_lag"], 12)
        self.assertEqual(coordinator.space["time_lag"], [12, 32, 64])
        proposals = []
        # Isolate search coverage from model-specific forward validation.
        with patch.object(coordinator, "_valid", return_value=True):
            for slot in range(1, 18):
                proposal = coordinator.propose(slot)
                proposals.append(proposal)
                coordinator.observe(self.synthetic_record(proposal, slot))
        self.assertEqual({proposal["config"]["time_lag"] for proposal in proposals}, {12, 32, 64})
        for proposal in proposals:
            if proposal.get("review_stage") == "structural_parameters":
                self.assertIn("time_lag", proposal["editable_parameters"])
            elif proposal.get("review_stage") == "training_parameters":
                self.assertNotIn("time_lag", proposal["editable_parameters"])

    def test_local_window_search_continues_from_improved_32_to_64(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        coordinator.stage_slots.update({"global": 0, "crossover": 0})
        coordinator.model_parameters, coordinator.training_parameters = ["time_lag"], []
        coordinator.observe(self.synthetic_record(coordinator.propose(1), 1))
        with patch.object(coordinator, "_valid", return_value=True):
            first = coordinator.propose(2)
            self.assertEqual((first["parameter"], first["config"]["time_lag"], first["base_slot"]),
                             ("time_lag", 32, 1))
            coordinator.observe(self.synthetic_record(first, 2))
            second = coordinator.propose(3)
        self.assertEqual((second["parameter"], second["config"]["time_lag"], second["base_slot"]),
                         ("time_lag", 64, 2))

    def test_llm_window_edit_is_model_only_and_keeps_fixed_forecast_horizon(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        baseline = self.synthetic_record(coordinator.propose(1), 1)
        coordinator.observe(baseline)
        with patch.object(coordinator, "_valid", return_value=True):
            proposal = coordinator.propose(2)
        model_parameter = next(name for name in coordinator.model_parameters if name != "time_lag")
        proposal.update(config={**coordinator.defaults, "time_lag": 32,
                                model_parameter: coordinator.probes[model_parameter][0]},
                        review_stage="structural_parameters", editable_parameters=["time_lag"],
                        editable_values={"time_lag": [12, 32, 64]})
        state = get_search_state(self.task, self.settings, proposal, [baseline], 2)
        self.assertEqual(state["task"]["time_lag"], 32)
        self.assertEqual(state["task"]["default_time_lag"], 12)
        self.assertEqual(state["task"]["time_lag_choices"], [12, 32, 64])
        self.assertTrue(state["task"]["input_window_tunable"])
        self.assertEqual(state["task"]["horizon"], self.task["horizon"])
        self.assertEqual(state["evaluation_protocol"]["forecast_origin_reference_time_lag"], 12)
        response = {"action": "edit_proposal", "overrides": {"time_lag": 64},
                    "evidence_trials": [1], "hypothesis": "Reuse a previously tested longer input window."}
        with patch("hpo.agent.inspect_candidate"):
            decision = parse_decision(json.dumps(response), state, self.task, self.settings, ROOT, set())
        self.assertEqual(decision["executed_config"]["time_lag"], 64)
        self.assertNotIn("horizon", decision["executed_config"])
        training_state = {**state, "editable_parameters": coordinator.training_parameters}
        with self.assertRaisesRegex(ValueError, "protected"):
            parse_decision(json.dumps(response), training_state, self.task, self.settings, ROOT, set())
        response["overrides"] = {"horizon": 12}
        with self.assertRaisesRegex(ValueError, "protected"):
            parse_decision(json.dumps(response), state, self.task, self.settings, ROOT, set())

    def test_legacy_search_without_window_space_retains_default_window(self):
        settings = copy.deepcopy(self.settings)
        settings["space"].pop("time_lag", None)
        settings["search"]["probe_values"].pop("time_lag", None)
        coordinator = Coordinator(self.task, settings, ROOT)
        proposal = coordinator.propose(1)
        self.assertNotIn("time_lag", coordinator.parameters)
        state = get_search_state(self.task, settings, proposal, [], 1)
        self.assertEqual(state["task"]["time_lag"], 12)
        self.assertEqual(state["task"]["time_lag_choices"], [12])
        self.assertFalse(state["task"]["input_window_tunable"])

    def test_llm_parameter_counts_are_record_only_when_bounds_are_disabled(self):
        settings = copy.deepcopy(self.settings)
        settings["space"]["enforce_param_ratio"] = False
        settings["space"].pop("param_ratio_min", None)
        settings["space"].pop("param_ratio_max", None)
        coordinator = Coordinator(self.task, settings, ROOT)
        proposal = coordinator.propose(1)
        record = self.synthetic_record(proposal, 1)
        record["result"].update(params=10 * self.task["original_params"], param_ratio=10.0)
        state = get_search_state(self.task, settings, proposal, [record], 2)
        self.assertEqual(state["parameter_count_policy"],
                         {"enforce_limits": False, "ratio_limits": None, "usage": "recording_only"})
        self.assertEqual(state["history"][0]["param_ratio"], 10.0)
        # Historical settings without the new switch retain their recorded size limits.
        settings["space"].pop("enforce_param_ratio")
        settings["space"].update(param_ratio_min=0.25, param_ratio_max=3.0)
        legacy = get_search_state(self.task, settings, proposal, [record], 2)
        self.assertEqual(legacy["parameter_count_policy"],
                         {"enforce_limits": True, "ratio_limits": [0.25, 3.0], "usage": "validity_constraint"})

    def test_model_priority_prefers_fewer_training_changes_at_equal_model_coverage(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        coordinator.observe(self.synthetic_record(coordinator.propose(1), 1))
        model_only = dict(coordinator.defaults)
        for name in coordinator.model_parameters[:2]:
            model_only[name] = coordinator.probes[name][0]
        mixed = {**model_only, "lr": coordinator.probes["lr"][0]}
        coordinator.global_candidates = {"model": [mixed, model_only], "training": []}
        with patch.object(coordinator, "_valid", return_value=True):
            self.assertEqual(coordinator._global_proposal("model"), model_only)

    def test_model_priority_explores_new_pairs_after_individual_values_are_covered(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        coordinator.observe(self.synthetic_record(coordinator.propose(1), 1))
        left, right = coordinator.model_parameters[:2]
        low_left, high_left = coordinator.probes[left]
        low_right, high_right = coordinator.probes[right]
        observed = [{**coordinator.defaults, left: low_left, right: low_right},
                    {**coordinator.defaults, left: high_left, right: high_right}]
        for index, config in enumerate(observed, 2):
            coordinator.records.append(self.fake_record(index, config))
        repeated_pair = {**observed[0], "lr": coordinator.probes["lr"][0]}
        new_pair = {**coordinator.defaults, left: low_left, right: high_right,
                    "lr": coordinator.probes["lr"][0]}
        coordinator.global_candidates = {"model": [repeated_pair, new_pair], "training": []}
        with patch.object(coordinator, "_valid", return_value=True):
            self.assertEqual(coordinator._global_proposal("model"), new_pair)

    def test_global_priority_falls_back_when_the_requested_category_has_no_valid_candidate(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        coordinator.observe(self.synthetic_record(coordinator.propose(1), 1))
        training_only = {**coordinator.defaults, "lr": coordinator.probes["lr"][0],
                         "weight_decay": coordinator.probes["weight_decay"][0]}
        coordinator.global_candidates = {"model": [], "training": [training_only]}
        with patch.object(coordinator, "_valid", return_value=True):
            self.assertEqual(coordinator._global_proposal("model"), training_only)

    def test_global_search_covers_shmetro_preferred_levels_before_crossover(self):
        for task in self.tasks:
            if task["dataset"] != "0208_shmetro_inflow_6716_288":
                continue
            with self.subTest(horizon=task["horizon"]):
                coordinator = Coordinator(task, self.settings, ROOT)
                for slot in range(1, 18):
                    proposal = coordinator.propose(slot)
                    coordinator.observe(self.synthetic_record(proposal, slot))
                observed = [record["executed_config"] for record in coordinator.records]
                for name, values in coordinator.probes.items():
                    self.assertTrue(set(values) <= {config[name] for config in observed}, name)
                self.assertTrue(all(not values for values in coordinator.trace()["untested_preferred_values"].values()))

    def test_local_search_balances_directions_when_both_are_available(self):
        task = next(task for task in self.tasks if task["dataset"] == "0208_shmetro_inflow_6716_288")
        coordinator = Coordinator(task, self.settings, ROOT)
        coordinator.stage_slots.update({"global": 0, "crossover": 0})
        with patch.object(coordinator, "_valid", return_value=True):
            for slot in range(1, 8):
                proposal = coordinator.propose(slot)
                self.assertIsNotNone(proposal)
                record = self.synthetic_record(proposal, slot)
                if slot == 1:
                    record["executed_config"] = {name: values[len(values) // 2]
                                                 for name, values in coordinator.space.items()}
                else:
                    record["result"]["val_rmse"] = 2.0  # Keep all elite coordinates away from boundaries.
                coordinator.observe(record)
        self.assertEqual(coordinator.trace()["local_direction_counts"], {"-1": 3, "1": 3})
        self.assertTrue(coordinator.complete)

    def test_local_search_uses_available_direction_at_parameter_boundaries(self):
        for index, direction in ((0, "1"), (-1, "-1")):
            with self.subTest(boundary=index):
                coordinator = Coordinator(self.task, self.settings, ROOT)
                coordinator.stage_slots.update({"global": 0, "crossover": 0})
                baseline = self.synthetic_record(coordinator.propose(1), 1)
                baseline["executed_config"] = {name: choices[index] for name, choices in coordinator.space.items()}
                coordinator.observe(baseline)
                # Isolate adjacency and direction selection from model forward validation.
                with patch.object(coordinator, "_valid", return_value=True):
                    for slot in range(2, 8):
                        proposal = coordinator.propose(slot)
                        record = self.synthetic_record(proposal, slot)
                        record["result"]["val_rmse"] = 2.0  # Keep the boundary baseline as the elite.
                        coordinator.observe(record)
                self.assertEqual(coordinator.trace()["local_direction_counts"][direction], 6)
                self.assertTrue(coordinator.complete)

    def test_empty_crossover_advances_to_local_and_replays_without_transferring_budget(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        for slot in range(1, 18):
            proposal = coordinator.propose(slot)
            record = self.synthetic_record(proposal, slot)
            if slot > 1:
                record["result"].update(status="FAIL", val_rmse=None)
            coordinator.observe(record)
        self.assertFalse(coordinator.complete)
        proposal = coordinator.propose(18)
        self.assertEqual(proposal["search_round"], "local")
        self.assertIn("crossover", coordinator.trace()["exhausted_stages"])
        coordinator.observe(self.synthetic_record(proposal, 18))
        replayed = Coordinator(self.task, self.settings, ROOT)
        for record in coordinator.records:
            replayed.observe(record)
        self.assertEqual(coordinator.propose(19), replayed.propose(19))
        for slot in range(19, 24):
            proposal = coordinator.propose(slot)
            coordinator.observe(self.synthetic_record(proposal, slot))
        self.assertTrue(coordinator.complete)
        self.assertEqual(coordinator.trace()["completed_by_stage"],
                         {"baseline": 1, "global": 16, "crossover": 0, "local": 6})
        self.assertEqual(coordinator.trace()["completed_slots"], 23)

    def test_exhausted_stages_respect_truncated_budget_and_terminate(self):
        settings = copy.deepcopy(self.settings)
        settings["search"]["trials"] = 18
        coordinator = Coordinator(self.task, settings, ROOT)
        for slot in range(1, 18):
            proposal = coordinator.propose(slot)
            record = self.synthetic_record(proposal, slot)
            if slot > 1:
                record["result"].update(status="FAIL", val_rmse=None)
            coordinator.observe(record)
        self.assertIsNone(coordinator.propose(18))
        self.assertTrue(coordinator.complete)
        self.assertEqual(coordinator.trace()["completed_by_stage"]["local"], 0)

        coordinator = Coordinator(self.task, self.settings, ROOT)
        coordinator.observe(self.synthetic_record(coordinator.propose(1), 1))
        with patch.object(coordinator, "_valid", return_value=False):
            self.assertIsNone(coordinator.propose(2))
        self.assertTrue(coordinator.complete)
        self.assertEqual(set(coordinator.trace()["exhausted_stages"]), {"global", "crossover", "local"})

    def test_llm_joint_review_can_only_reuse_completed_values(self):
        # The first review follows only two model-directed global probes.
        task = max(self.tasks, key=lambda task: len(active_space(task, self.settings)))
        coordinator = Coordinator(task, self.settings, ROOT)
        for slot in range(1, 6):
            proposal = coordinator.propose(slot)
            coordinator.observe(self.synthetic_record(proposal, slot))
        proposal = coordinator.propose(6)
        self.assertEqual(proposal["review_stage"], "structural_parameters")
        parameter = next(name for name in proposal["editable_parameters"]
                         if any(value not in proposal["editable_values"][name]
                                for value in coordinator.space[name]))
        unseen = next(value for value in coordinator.space[parameter]
                      if value not in proposal["editable_values"][parameter])
        state = {"coordinate_proposal": proposal["config"],
                 "coordinate_parameter": proposal["parameter"],
                 "editable_parameters": proposal["editable_parameters"],
                 "editable_values": proposal["editable_values"],
                 "executed_slots": list(range(1, 6))}
        response = {"action": "edit_proposal",
                    "overrides": {parameter: unseen},
                    "evidence_trials": [1], "hypothesis": "Try an unobserved value."}
        with self.assertRaisesRegex(ValueError, "already tested values"):
            parse_decision(json.dumps(response), state, task,
                           self.settings, ROOT, coordinator.seen)

    def test_llm_cannot_collapse_a_global_candidate_to_one_changed_parameter(self):
        defaults = self.config()
        parameters = coordinate_order(self.task, self.settings)[:2]
        proposal = dict(defaults)
        for name in parameters:
            proposal[name] = next(value for value in active_space(self.task, self.settings)[name]
                                  if value != defaults[name])
        state = {"coordinate_proposal": proposal, "coordinate_parameter": None,
                 "search_round": "global", "editable_parameters": parameters,
                 "editable_values": {name: [defaults[name]] for name in parameters},
                 "executed_slots": [1]}
        response = {"action": "edit_proposal",
                    "overrides": {parameters[0]: defaults[parameters[0]],
                                  parameters[1]: defaults[parameters[1]]},
                    "evidence_trials": [1], "hypothesis": "Collapse the joint proposal."}
        with self.assertRaisesRegex(ValueError, "at least two parameters"):
            parse_decision(json.dumps(response), state, self.task,
                           self.settings, ROOT, {json.dumps(defaults, sort_keys=True)})

    def test_llm_cannot_convert_training_slot_into_model_priority(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        model_parameter = coordinator.model_parameters[0]
        config = {**coordinator.defaults, "lr": coordinator.probes["lr"][0],
                  "weight_decay": coordinator.probes["weight_decay"][0],
                  model_parameter: coordinator.probes[model_parameter][0]}
        state = {"coordinate_proposal": config, "coordinate_parameter": None,
                 "search_round": "global", "effective_priority": "training",
                 "editable_parameters": coordinator.training_parameters,
                 "editable_values": {"lr": [coordinator.defaults["lr"]]},
                 "executed_slots": [1]}
        response = {"action": "edit_proposal", "overrides": {"lr": coordinator.defaults["lr"]},
                    "evidence_trials": [1], "hypothesis": "Return learning rate to its default."}
        with self.assertRaisesRegex(ValueError, "preserve the scheduled"):
            parse_decision(json.dumps(response), state, self.task, self.settings, ROOT, set())

    def test_coordinate_replay_is_deterministic(self):
        first = Coordinator(self.task, self.settings, ROOT)
        history = []
        for slot in range(1, 5):
            proposal = first.propose(slot)
            record = self.synthetic_record(proposal, slot)
            first.observe(record)
            history.append(record)
        replayed = Coordinator(self.task, self.settings, ROOT)
        for record in history:
            replayed.observe(record)
        self.assertEqual(first.propose(5), replayed.propose(5))

    def test_all_tasks_respect_stage_budgets_without_duplicate_configs(self):
        for task in self.tasks:
            coordinator = Coordinator(task, self.settings, ROOT)
            keys = set()
            while not coordinator.complete:
                slot = len(coordinator.records) + 1
                proposal = coordinator.propose(slot)
                if proposal is None:
                    self.assertTrue(coordinator.complete)
                    break
                self.assertIsNotNone(proposal["config"], f"{task['id']} slot {slot}")
                key = json.dumps(proposal["config"], sort_keys=True)
                self.assertNotIn(key, keys, f"{task['id']} slot {slot}")
                keys.add(key)
                record = {"slot": slot, "executed_config": proposal["config"],
                          "phase": proposal["phase"], "search_round": proposal["search_round"],
                          "parameter": proposal.get("parameter"), "probe_value": proposal.get("probe_value"),
                          "base_slot": proposal.get("base_slot"), "parent_slots": proposal.get("parent_slots", []),
                          "result": {"status": "COMPLETE", "val_rmse": 1.0 if slot == 1 else 2.0,
                                     "params": task["original_params"], "seed": proposal["seed"]}}
                coordinator.observe(record)
            self.assertLessEqual(len(coordinator.records), 32)
            trace = coordinator.trace()
            tested = {value for values in trace["tested_target_values"].values() for value in values}
            self.assertTrue(tested)
            for stage, budget in trace["stage_slot_plan"].items():
                self.assertLessEqual(trace["completed_by_stage"][stage], budget)
                if trace["completed_by_stage"][stage] < budget:
                    self.assertIn(stage, trace["exhausted_stages"], task["id"])

    def test_crossover_and_local_elites_follow_new_improvements(self):
        task = max(self.tasks, key=lambda task: len(active_space(task, self.settings)))
        coordinator = Coordinator(task, self.settings, ROOT)
        rounds, proposals, reused_crossovers, reused_locals = [], [], [], []
        while not coordinator.complete:
            slot = len(coordinator.records) + 1
            proposal = coordinator.propose(slot)
            if proposal is None:
                break
            rounds.append(proposal["search_round"])
            proposals.append(proposal)
            by_slot = {record["slot"]: record for record in coordinator.records}
            if proposal["search_round"] == "crossover":
                pool_size = proposal["parent_pool_size"]
                eligible = [record["slot"] for record in coordinator._crossover_elites(limit=None)]
                self.assertEqual(proposal["parent_pool_slots"], eligible[:pool_size])
                self.assertEqual(len(proposal["parent_slots"]), 2)
                self.assertTrue(set(proposal["parent_slots"]) <= set(proposal["parent_pool_slots"]))
                self.assertEqual(proposal["parent_pool_expanded"], pool_size > 4)
                if pool_size > 4:
                    self.assertIn(eligible[pool_size - 1], proposal["parent_slots"])
                reused_crossovers.extend(parent for parent in proposal["parent_slots"]
                                         if by_slot[parent]["search_round"] == "crossover")
                for name in coordinator.parameters:
                    self.assertIn(proposal["config"][name],
                                  {by_slot[parent]["executed_config"][name]
                                   for parent in proposal["parent_slots"]})
            elif proposal["search_round"] == "local":
                top2 = {record["slot"] for record in coordinator._local_elites()}
                self.assertIn(proposal["base_slot"], top2)
                base = by_slot[proposal["base_slot"]]
                if base["search_round"] == "local":
                    reused_locals.append(base["slot"])
                changed = [name for name in coordinator.parameters
                           if proposal["config"][name] != base["executed_config"][name]]
                self.assertEqual(changed, [proposal["parameter"]])
                allowed = list(coordinator.space[proposal["parameter"]])
                before = base["executed_config"][proposal["parameter"]]
                self.assertEqual(abs(allowed.index(before) - allowed.index(proposal["probe_value"])), 1)
            validation_rmse = 1.0 - slot / 1000
            # Diverse early parents expose a newly improved crossover to the next slot.
            if proposal["search_round"] == "global" and slot > 5:
                validation_rmse += 1.0
            record = {"slot": slot, "executed_config": proposal["config"],
                      "phase": proposal["phase"], "search_round": proposal["search_round"],
                      "parameter": proposal.get("parameter"),
                      "probe_value": proposal.get("probe_value"),
                      "base_slot": proposal.get("base_slot"),
                      "parent_slots": proposal.get("parent_slots", []),
                      "result": {"status": "COMPLETE", "val_rmse": validation_rmse,
                                 "params": task["original_params"], "seed": proposal["seed"]}}
            coordinator.observe(record)
        self.assertEqual(rounds[:17], ["baseline", *(["global"] * 16)])
        self.assertEqual(rounds[-6:], ["local"] * 6)
        self.assertLessEqual(rounds.count("crossover"), 9)
        self.assertTrue(reused_crossovers)
        self.assertTrue(reused_locals)
        self.assertEqual([record["slot"] for record in coordinator.records],
                         list(range(1, len(coordinator.records) + 1)))
        self.assertIsNone(coordinator.propose(len(coordinator.records) + 1))
        self.assertEqual([(slot, proposal["review_stage"])
                          for slot, proposal in enumerate(proposals, 1) if proposal.get("review_stage")],
                         [(6, "structural_parameters"), (11, "training_parameters"),
                          (17, "training_parameters")])
        local_parameters = [proposal["parameter"] for proposal in proposals
                            if proposal["search_round"] == "local"]
        self.assertTrue(all(parameter not in TRAINING_PARAMETERS for parameter in local_parameters[::2]))
        self.assertTrue(all(parameter in TRAINING_PARAMETERS for parameter in local_parameters[1::2]))

    def test_crossover_expands_incrementally_when_top_four_only_differ_in_one_field(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        coordinator.stage_slots["global"] = 0
        baseline = coordinator.propose(1)
        coordinator.observe(self.synthetic_record(baseline, 1))
        configs = [{**coordinator.defaults, "lr": lr} for lr in (0.0001, 0.0003, 0.003, 0.005)]
        configs.append({**coordinator.defaults, "weight_decay": 0.0001})
        for slot, config in enumerate(configs, 2):
            record = self.fake_record(slot, config)
            record.update(search_round="global", screening_status="COMPLETE",
                          screening_val_rmse_mean=1.0 + slot / 10)
            coordinator.observe(record)
        # Top 5 differ only in lr, so only the sixth-ranked parent enables a new combination.
        with patch.object(coordinator, "_valid", return_value=True) as valid:
            proposal = coordinator.propose(7)
        self.assertEqual(proposal["search_round"], "crossover")
        self.assertEqual(proposal["parent_pool_slots"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(proposal["parent_pool_size"], 6)
        self.assertTrue(proposal["parent_pool_expanded"])
        self.assertIn(6, proposal["parent_slots"])
        self.assertEqual(proposal["source"], "expanded_elite_crossover")
        self.assertNotIn(coordinator._key(proposal["config"]), coordinator.seen)
        self.assertEqual(valid.call_count, 1)  # Stop as soon as the first new legal candidate appears.
        record = self.synthetic_record(proposal, 7)
        coordinator.observe(record)
        self.assertEqual(coordinator.trace()["crossover_pool_expansions"],
                         [{"slot": 7, "parent_slots": proposal["parent_slots"],
                           "parent_pool_slots": [1, 2, 3, 4, 5, 6], "parent_pool_size": 6}])

    def test_crossover_keeps_top_four_order_and_does_not_expand_while_valid(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        for slot in range(1, 18):
            proposal = coordinator.propose(slot)
            coordinator.observe(self.synthetic_record(proposal, slot))
        first = next((config, parents) for config, parents in coordinator._crossover_candidates()
                     if coordinator._key(config) not in coordinator.seen)
        with patch.object(coordinator, "_valid", return_value=True):
            proposal = coordinator.propose(18)
        self.assertEqual((proposal["config"], proposal["parent_slots"]), first)
        self.assertEqual(proposal["parent_pool_size"], 4)
        self.assertFalse(proposal["parent_pool_expanded"])
        self.assertEqual(proposal["source"], "top4_crossover")

    def test_screening_mean_controls_incumbent_and_incomplete_seed_cannot_win(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        records = []
        # Seed 2026 alone prefers slot 2; the reliable screening mean prefers slot 3.
        for slot, single, mean, status in ((1, 1.0, 1.0, "COMPLETE"),
                                          (2, 0.1, 1.2, "COMPLETE"),
                                          (3, 0.8, 0.9, "COMPLETE"),
                                          (4, 0.01, None, "FAIL")):
            proposal = coordinator.propose(slot)
            record = self.synthetic_record(proposal, slot)
            record["result"]["val_rmse"] = single
            record.update(screening_val_rmse_mean=mean, screening_status=status)
            coordinator.observe(record)
            records.append(record)
        self.assertEqual(coordinator.incumbent["slot"], 3)
        self.assertEqual([record["slot"] for record in coordinator._crossover_elites()], [3, 1, 2])
        self.assertEqual(coordinator.trace()["challenger_slot"], 3)
        self.assertFalse(record_complete(records[3]))
        self.assertIsNone(record_score(records[3]))
        self.assertEqual(records[2]["result"]["val_rmse"], 0.8)
        proposal = coordinator.propose(5)
        state = get_search_state(self.task, self.settings, proposal, records, 5)
        self.assertEqual(state["incumbent_slot"], 3)
        self.assertEqual(state["difference_from_incumbent"],
                         {name: {"current": records[2]["executed_config"][name], "proposal": value}
                          for name, value in proposal["config"].items()
                          if value != records[2]["executed_config"][name]})

    def test_agent_history_exposes_seed_mean_without_relabeling_raw_fit(self):
        coordinator = Coordinator(self.task, self.settings, ROOT)
        record = self.synthetic_record(coordinator.propose(1), 1)
        record.update(screening_status="COMPLETE", screening_val_rmse_mean=1.2,
                      screening_val_rmse_std=0.2,
                      screening_results={"2026": {"result": record["result"]},
                                         "2027": {"result": {"status": "COMPLETE", "val_rmse": 1.4}}})
        item = summary(record, coordinator.defaults)
        self.assertEqual(item["val_rmse"], 1.2)
        self.assertEqual(item["seed_results"]["2026"]["val_rmse"], 1.0)
        self.assertEqual(item["seed_results"]["2027"]["val_rmse"], 1.4)

    def test_no_mock_results_in_formal_runs(self):
        with self.assertRaises(ValueError):
            require_llm(load_settings(), mock=True)

    def test_deepseek_file_settings_ignore_legacy_environment(self):
        with patch.dict(os.environ, {"PAR_LLM_MODEL": "other-model",
                                     "PAR_LLM_BASE_URL": "https://other-provider.invalid"}):
            settings = load_settings()
        self.assertEqual(settings["agent"]["model"], "deepseek-flash")
        self.assertEqual(settings["agent"]["base_url"], "https://api.deepseek.com")

    def test_llm_http_client_is_explicitly_direct(self):
        settings = load_settings(smoke=True)
        agent = Agent(settings, api_key="unit-placeholder-secret")
        http_client, openai_client = MagicMock(), MagicMock()
        response = MagicMock()
        response.model_dump.return_value = {"choices": []}
        openai_client.__enter__.return_value.chat.completions.create.return_value = response
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy.invalid:9999"}), \
             patch("openai.DefaultHttpxClient", return_value=http_client) as client_factory, \
             patch("openai.OpenAI", return_value=openai_client) as openai_factory:
            self.assertEqual(agent.call({"model": "test", "messages": []}, {}), {"choices": []})
        client_factory.assert_called_once_with(trust_env=False, proxy=None)
        self.assertIs(openai_factory.call_args.kwargs["http_client"], http_client)
        self.assertEqual(openai_factory.call_args.kwargs["max_retries"], 3)
        self.assertEqual(openai_factory.call_args.kwargs["timeout"], 180)

    def test_llm_sdk_retries_transient_failures_without_network(self):
        import httpx
        import openai

        settings = load_settings()
        settings["agent"]["base_url"] = "https://unit.invalid"
        errors = {"timeout": openai.APITimeoutError,
                  "connection": openai.APIConnectionError,
                  429: openai.RateLimitError, 500: openai.InternalServerError}
        cases = [(failure, failures, errors[failure] if failures == 4 else None,
                  4 if failures == 4 else 2)
                 for failure in errors for failures in (1, 4)]
        cases.append((401, 4, openai.AuthenticationError, 1))
        success = {"id": "unit-completion", "object": "chat.completion", "created": 0,
                   "model": "test", "choices": [{"index": 0, "finish_reason": "stop",
                   "message": {"role": "assistant", "content": "{}"}}]}
        for failure, failures, expected_error, attempts in cases:
            with self.subTest(failure=failure, failures=failures):
                requests = []

                def respond(request):
                    requests.append(request)
                    if len(requests) > failures:
                        return httpx.Response(200, json=success)
                    if failure == "timeout":
                        raise httpx.ReadTimeout("synthetic timeout", request=request)
                    if failure == "connection":
                        raise httpx.ConnectError("synthetic connection failure", request=request)
                    return httpx.Response(failure, json={"error": {"message": "synthetic failure"}})

                client = openai.DefaultHttpxClient(trust_env=False, proxy=None,
                                                 transport=httpx.MockTransport(respond))
                agent = Agent(settings, api_key="unit-placeholder-secret")
                with patch("openai.DefaultHttpxClient", return_value=client), \
                     patch("openai._base_client.time.sleep") as retry_sleep:
                    request = {"model": "test", "messages": [{"role": "user", "content": "test"}]}
                    if expected_error:
                        with self.assertRaises(expected_error):
                            agent.call(request, {})
                    else:
                        self.assertEqual(agent.call(request, {})["id"], success["id"])
                self.assertEqual(len(requests), attempts)
                self.assertEqual(retry_sleep.call_count, attempts - 1)
                self.assertTrue(client.is_closed)
                for request in requests:
                    self.assertEqual(request.extensions["timeout"],
                                     {name: 180 for name in ("connect", "read", "write", "pool")})

    def test_llm_retry_setting_accepts_only_zero_to_three_integers(self):
        settings = load_settings()
        path = self.directory / "settings.json"
        for value in (0, 3, -1, 4, True, 3.0):
            with self.subTest(max_retries=value):
                settings["agent"]["max_retries"] = value
                path.write_text(json.dumps(settings), encoding="utf-8")
                if type(value) is int and 0 <= value <= 3:
                    self.assertEqual(load_settings(path)["agent"]["max_retries"], value)
                else:
                    with self.assertRaisesRegex(ValueError, "max_retries"):
                        load_settings(path)

    def test_deepseek_key_resolution_does_not_change_settings(self):
        before = copy.deepcopy(self.settings)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "  unit-env-key  "}):
            self.assertEqual(require_llm(self.settings), "unit-env-key")
            self.assertEqual(require_llm(self.settings, api_key=" unit-cli-key "), "unit-cli-key")
            with self.assertRaises(ValueError):
                require_llm(self.settings, api_key="   ")
            self.assertIsNone(require_llm(self.settings, mock=True))
        self.assertEqual(self.settings, before)

    def test_failed_training_has_no_fabricated_score(self):
        with patch("hpo.training.evaluate", side_effect=FloatingPointError("synthetic numerical failure")):
            result = fit(self.task, self.config(), self.settings, ROOT, self.directory,
                         seed=2026, epochs=1, patience=1, purpose="unit_failure")
        self.assertEqual(result["status"], "FAIL")
        self.assertIsNone(result["val_rmse"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_training_on_two_module_branches(self):
        settings = copy.deepcopy(self.settings)
        settings["training"]["device"] = "cuda"
        tasks = [self.task, next(task for task in self.tasks if "LSTM Block" in task["spec"]["B"])]
        for index, task in enumerate(tasks):
            config = self.config(task)
            config["batch_size"] = 32
            if "LSTM Block" in task["spec"]["B"]:
                config["lstm_width_scale"] = 0.75
                config["lstm_layers"] = 2
            if "tcn_kernel" in config:
                config["tcn_kernel"] = 4
            if "patch_profile" in config:
                config["patch_profile"] = "p6_s3"
            result = fit(task, config, settings, ROOT, self.directory / str(index),
                         seed=2026, epochs=1, patience=1, purpose="unit_cuda_smoke")
            self.assertEqual(result["status"], "COMPLETE", result.get("error"))

    def test_checkpoint_and_early_stop_use_normalized_validation_loss(self):
        first = {"val_loss": 1.0, "val_mae": 1.0, "val_rmse": 9.0}
        second = {"val_loss": 2.0, "val_mae": 1.0, "val_rmse": 1.0}
        with patch("hpo.training.evaluate", side_effect=[first, second, first]):
            result = fit(self.task, self.config(), self.settings, ROOT, self.directory,
                         seed=2026, epochs=2, patience=2, purpose="unit_checkpoint_contract")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["val_loss"], 1.0)
        self.assertEqual(result["val_rmse"], 9.0)
        self.assertEqual(result["checkpoint_metric"], "normalized_validation_0.2_mse_0.8_mae_loss")
        with (self.directory / result["history"]).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([float(row["best_val_loss"]) for row in rows], [1.0, 1.0])

    def test_formal_patience_stops_after_fifteen_non_improving_epochs(self):
        best = {"val_loss": 1.0, "val_mae": 1.0, "val_rmse": 2.0}
        worse = {"val_loss": 1.1, "val_mae": 0.9, "val_rmse": 1.0}
        with patch("hpo.training.evaluate", side_effect=[best, *([worse] * 15), best]):
            result = fit(self.task, self.config(), self.settings, ROOT, self.directory,
                         seed=2026, epochs=300, patience=15, purpose="unit_patience_contract")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["epochs_run"], 16)


if __name__ == "__main__":
    unittest.main()
