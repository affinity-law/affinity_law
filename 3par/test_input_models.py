"""Input-length parameterization preserves forecast horizons and frozen L modules."""

import ast
import copy
import tempfile
import unittest
from pathlib import Path

import torch

from hpo.config import ROOT, load_settings, tasks_from_manifest
from hpo.models import (active_space, candidate_code, default_config, extract_spec,
                        inspect_candidate, instantiate, validate_config)


class InputModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = tasks_from_manifest()
        cls.settings = load_settings(smoke=True)
        cls.settings["space"]["time_lag"] = [12, 32, 64]
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix=".input_model_test_", dir=ROOT)
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.assertTrue(self.folder.resolve().is_relative_to(ROOT.resolve()))
        self.temp.cleanup()

    def test_default_sources_are_verbatim_and_legacy_spaces_stay_unchanged(self):
        legacy = copy.deepcopy(self.settings)
        legacy["space"].pop("time_lag")
        for task in self.tasks:
            with self.subTest(task=task["id"]):
                config = default_config(task, self.settings)
                self.assertEqual(config["time_lag"], 12)
                self.assertEqual(active_space(task, self.settings)["time_lag"], [12, 32, 64])
                self.assertEqual(candidate_code(task, config, ROOT),
                                 (ROOT / task["model_path"]).read_text(encoding="utf-8-sig"))
                old_config = default_config(task, legacy)
                self.assertNotIn("time_lag", old_config)
                self.assertEqual(validate_config(old_config, task, legacy), old_config)
                self.assertEqual(candidate_code(task, old_config, ROOT), candidate_code(task, config, ROOT))

    def test_patch_decomposition_and_heads_consume_long_inputs_with_fixed_outputs(self):
        chosen = [next(task for task in self.tasks
                       if task["dataset"] == "0113_electricity_vic_9229_1" and task["horizon"] == horizon)
                  for horizon in (3, 12)]
        chosen.extend(next(task for task in self.tasks if module in task["spec"]["B"])
                      for module in ("LSTM Block", "State Space Model Block"))
        for task in chosen:
            for lag in (32, 64):
                with self.subTest(task=task["id"], time_lag=lag):
                    config = {key: 0.5 if key.endswith("_width_scale") else value
                              for key, value in default_config(task, self.settings).items()}
                    config["time_lag"] = lag
                    inspected = inspect_candidate(task, config, self.settings, ROOT,
                                                  check_forward=True, batch_sizes=(1, 2))
                    self.assertEqual(extract_spec(inspected["code"]), task["spec"])
                    model = instantiate(inspected["code"], task, config)
                    self.assertEqual(model.time_lag, lag)
                    self.assertEqual(model.pre_len, task["horizon"])
                    self.assertEqual(model(torch.randn(2, lag, task["channels"])).shape,
                                     (2, task["horizon"], task["channels"]))
                    for module in model.modules():
                        if type(module).__name__ == "_A_Patch":
                            self.assertEqual(module.time_lag, lag)
                        if type(module).__name__ == "_A_SeriesDecomposition":
                            self.assertEqual(module.trend_proj.in_features, lag)
                            self.assertEqual(module.trend_proj.out_features, task["horizon"])

    def test_semantic_arguments_do_not_rewrite_horizon_channels_or_other_twelves(self):
        source = '''import torch
from torch import nn

class _B_MLP(nn.Module):
    def __init__(self, channels, time_lag, hidden=12, kernel=12):
        super().__init__()
        self.time_lag, self.hidden, self.kernel = time_lag, hidden, kernel
        self.scale = nn.Parameter(torch.ones(channels))
    def forward(self, x):
        assert x.shape[1] == self.time_lag
        return x * self.scale

class _C_LinearHead(nn.Module):
    def __init__(self, channels, pre_len, time_lag, marker=12):
        super().__init__()
        self.channels, self.marker = channels, marker
        self.proj = nn.Linear(time_lag, pre_len)
    def forward(self, x):
        return self.proj(x.transpose(1, 2)).transpose(1, 2)

class _E_Composition(nn.Module):
    def __init__(self, count, name):
        super().__init__()

def _build_b(name):
    return _B_MLP(12, 12, hidden=12, kernel=12)

def _build_c(name):
    return _C_LinearHead(12, 12, time_lag=12, marker=12)

class CandidateModel(nn.Module):
    def __init__(self, time_lag, pre_len, station_num):
        super().__init__()
        self.b = _build_b("MLP Block")
        self.head = _build_c("Linear Head")
        self.e = _E_Composition(1, "Single Block")
    def forward(self, x):
        return self.head(self.b(x))
'''
        path = self.folder / "semantic_model.py"
        path.write_text(source, encoding="utf-8")
        task = {"model_path": str(path), "spec": extract_spec(source), "horizon": 12, "channels": 12}
        config = {**default_config(task, self.settings), "time_lag": 32}
        code = candidate_code(task, config, ROOT)
        tree = ast.parse(code)
        calls = {node.func.id: node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id in {"_B_MLP", "_C_LinearHead"}}
        self.assertEqual([ast.literal_eval(arg) for arg in calls["_B_MLP"].args], [12, 32])
        self.assertEqual([ast.literal_eval(arg) for arg in calls["_C_LinearHead"].args], [12, 12])
        self.assertEqual({kw.arg: ast.literal_eval(kw.value) for kw in calls["_C_LinearHead"].keywords},
                         {"time_lag": 32, "marker": 12})
        model = instantiate(code, task, config)
        self.assertEqual((model.b.hidden, model.b.kernel, model.head.channels, model.head.marker), (12, 12, 12, 12))
        self.assertEqual(model(torch.randn(2, 32, 12)).shape, (2, 12, 12))

    def test_exported_source_and_weights_reload_and_use_history_beyond_twelve(self):
        task = self.tasks[0]
        config = {**default_config(task, self.settings), "time_lag": 64}
        inspected = inspect_candidate(task, config, self.settings, ROOT, check_forward=True)
        original = instantiate(inspected["code"], task, config).eval()
        path = self.folder / "exported_model.py"
        path.write_text(inspected["code"], encoding="utf-8")
        loaded = instantiate(path.read_text(encoding="utf-8"), task, config).eval()
        loaded.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(2, 64, task["channels"])
        with torch.no_grad():
            expected = original(x)
            actual = loaded(x)
            changed = x.clone()
            changed[:, :20] += 3.0
            with_long_history_changed = loaded(changed)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.equal(actual, with_long_history_changed))
        self.assertEqual(loaded.time_lag, 64)
        self.assertEqual(actual.shape, (2, task["horizon"], task["channels"]))

    def test_legacy_parameter_budget_still_rejects_an_oversized_long_input_model(self):
        task = None
        for candidate in self.tasks:
            config = {key: 0.5 if key.endswith("_width_scale") else value
                      for key, value in default_config(candidate, self.settings).items()}
            ratios = [inspect_candidate(candidate, {**config, "time_lag": lag}, self.settings, ROOT)["param_ratio"]
                      for lag in (32, 64)]
            if ratios[0] <= 3.0 < ratios[1]:
                task = candidate
                break
        self.assertIsNotNone(task, "Need a current L model crossing the legacy parameter limit")
        legacy_settings = copy.deepcopy(self.settings)
        legacy_settings["space"].pop("enforce_param_ratio", None)
        legacy_settings["space"].update(param_ratio_min=0.25, param_ratio_max=3.0)
        config = {key: 0.5 if key.endswith("_width_scale") else value
                  for key, value in default_config(task, self.settings).items()}
        config["time_lag"] = 64
        with self.assertRaisesRegex(ValueError, "parameter ratio is outside"):
            inspect_candidate(task, config, legacy_settings, ROOT, check_forward=True)
        inspected = inspect_candidate(task, {**config, "time_lag": 32}, legacy_settings, ROOT,
                                      check_forward=True)
        self.assertLessEqual(inspected["param_ratio"], 3.0)

    def test_removed_parameter_limit_accepts_formerly_excluded_long_inputs(self):
        unrestricted = copy.deepcopy(self.settings)
        unrestricted["space"]["enforce_param_ratio"] = False
        unrestricted["space"].pop("param_ratio_min", None)
        unrestricted["space"].pop("param_ratio_max", None)
        formerly_excluded = []
        for task in self.tasks:
            minimum = {key: 0.5 if key.endswith("_width_scale") else value
                       for key, value in default_config(task, unrestricted).items()}
            minimum["time_lag"] = 64
            model = instantiate(candidate_code(task, minimum, ROOT), task, minimum)
            ratio = sum(p.numel() for p in model.parameters() if p.requires_grad) / task["original_params"]
            if ratio > 3.0:
                formerly_excluded.append(task)
        self.assertTrue(formerly_excluded)
        for task in formerly_excluded:
            with self.subTest(task=task["id"]):
                config = {**default_config(task, unrestricted), "time_lag": 64}
                inspected = inspect_candidate(task, config, unrestricted, ROOT,
                                              check_forward=True, batch_sizes=(1, 2))
                self.assertGreater(inspected["param_ratio"], 3.0)
                self.assertGreater(inspected["params"], task["original_params"])

    def test_generic_direct_input_models_need_no_literal_input_rewrite(self):
        from test_tune import SimpleTuningTests

        fixture = SimpleTuningTests()
        fixture.setUp()
        try:
            source = fixture.model.read_text(encoding="utf-8")
            task = {"model_path": str(fixture.model), "spec": extract_spec(source),
                    "horizon": 3, "channels": 2}
            for lag in (12, 32, 64):
                config = {**default_config(task, self.settings), "time_lag": lag}
                code = candidate_code(task, config, ROOT)
                model = instantiate(code, task, config)
                self.assertEqual(model(torch.randn(2, lag, 2)).shape, (2, 3, 2))
                self.assertEqual(extract_spec(code), task["spec"])
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
