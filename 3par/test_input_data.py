"""Aligned evaluation dates, historical compatibility, and real long-input training."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hpo.artifacts import read_json, save_json
from hpo.config import ROOT, load_settings
from hpo.data import (TARGET_ALIGNMENT, input_window_details, split_bounds,
                      test_loader, train_val_loaders)
from hpo.evaluation import evaluate_test, read_test_cache, test_request
from hpo.models import candidate_class, default_config, extract_spec
from hpo.training import fit, training_request


MODEL_SOURCE = '''import torch
from torch import nn

class _B_MLP(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.layer = nn.Linear(channels, hidden)
    def forward(self, x):
        return torch.relu(self.layer(x))

class _C_Linear(nn.Module):
    def __init__(self, horizon, channels):
        super().__init__()
        self.layer = nn.Linear(8, channels)
        self.horizon = horizon
    def forward(self, x):
        return self.layer(x.mean(dim=1, keepdim=True)).expand(-1, self.horizon, -1)

def _build_b(name, channels):
    return _B_MLP(channels, hidden=8)

def _build_c(name, horizon, channels):
    return _C_Linear(horizon, channels)

class _E_Composition(nn.Module):
    def __init__(self, count, name):
        super().__init__()

class CandidateModel(nn.Module):
    def __init__(self, time_lag, pre_len, station_num):
        super().__init__()
        self.time_lag = time_lag
        self.b = _build_b("MLP Block", station_num)
        self.c = _build_c("Linear Head", pre_len, station_num)
        self.e = _E_Composition(1, "Single Block")
    def forward(self, x):
        assert x.shape[1] == self.time_lag
        return self.c(self.b(x))
'''


class InputDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".input_data_test_", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "series.npy"
        positions = np.arange(640, dtype=np.float32)
        self.series = np.column_stack((positions, np.sin(positions / 13)))
        np.save(self.path, self.series)
        self.settings = load_settings(smoke=True)
        self.settings["smoke"] = False
        self.settings["training"].update(device="cpu", cpu_threads=1)
        (self.directory / "model.py").write_text(MODEL_SOURCE, encoding="utf-8")
        params = sum(p.numel() for p in candidate_class(MODEL_SOURCE)(12, 3, 2).parameters())
        self.task = {"id": "aligned__h3", "dataset": "aligned", "horizon": 3, "channels": 2,
                     "data_path": "series.npy", "model_path": "model.py",
                     "spec": extract_spec(MODEL_SOURCE), "original_params": params}

    def loaders(self, lag, horizon=3):
        task = dict(self.task, horizon=horizon)
        config = {"time_lag": lag}
        train, val, mean, std = train_val_loaders(self.path, task, self.settings, 32, 2026, config)
        test = test_loader(self.path, task, self.settings, 32, 2026, mean, std, config)
        return train, val, test, mean, std

    @staticmethod
    def targets(batches):
        return torch.cat([target for _, target in batches]).numpy()

    def test_all_input_lengths_preserve_val_and_test_dates_and_targets(self):
        train_end, val_end = split_bounds(len(self.series), self.settings)
        for horizon in (3, 12):
            reference = None
            for lag in (12, 32, 64):
                train, val, test, mean, std = self.loaders(lag, horizon)
                self.assertEqual(train.dataset[0][0].shape, (lag, 2))
                self.assertEqual(len(val.dataset), val_end - train_end - 12 - horizon + 1)
                self.assertEqual(len(test.dataset), len(self.series) - val_end - 12 - horizon + 1)
                val_y, test_y = self.targets(val), self.targets(test)
                np.testing.assert_allclose(val_y[0] * std + mean, self.series[train_end + 12:train_end + 12 + horizon], atol=4e-5)
                np.testing.assert_allclose(test_y[0] * std + mean, self.series[val_end + 12:val_end + 12 + horizon], atol=4e-5)
                np.testing.assert_allclose(val.dataset[0][0].numpy() * std + mean,
                                           self.series[train_end + 12 - lag:train_end + 12], atol=4e-5)
                if reference is not None:
                    np.testing.assert_array_equal(val_y, reference[0])
                    np.testing.assert_array_equal(test_y, reference[1])
                reference = val_y, test_y

    def test_explicit_12_is_identical_to_historical_loading_and_requests_stay_legacy(self):
        old_train, old_val, mean, std = train_val_loaders(self.path, self.task, self.settings, 32, 2026)
        old_test = test_loader(self.path, self.task, self.settings, 32, 2026, mean, std)
        train, val, test, new_mean, new_std = self.loaders(12)
        np.testing.assert_array_equal(mean, new_mean)
        np.testing.assert_array_equal(std, new_std)
        for old, new in ((old_train, train), (old_val, val), (old_test, test)):
            for old_batch, new_batch in zip(old, new):
                for left, right in zip(old_batch, new_batch):
                    self.assertTrue(torch.equal(left, right))
        config = {"lr": 0.001, "batch_size": 16, "weight_decay": 0.0}
        request = training_request(self.task, config, self.settings, 2026, 1, 1, "test")
        self.assertNotIn("input_window", request)
        self.assertEqual(test_request(self.task, config, self.settings, {"attempt": "attempt_001"}, 2026),
                         {"task_id": self.task["id"], "fit_attempt": "attempt_001", "config": config,
                          "seed": 2026, "smoke": False})
        self.assertEqual(input_window_details(config, self.settings), {})

    def test_future_test_changes_cannot_affect_training_or_validation(self):
        _, val_end = split_bounds(len(self.series), self.settings)
        before = {lag: self.loaders(lag) for lag in (12, 32, 64)}
        altered = self.series.copy()
        altered[val_end:] = np.nan
        np.save(self.path, altered)
        for lag in (12, 32, 64):
            train, val, mean, std = train_val_loaders(self.path, self.task, self.settings, 32, 2026, {"time_lag": lag})
            old_train, old_val, _, old_mean, old_std = before[lag]
            self.assertTrue(torch.equal(train.dataset.data, old_train.dataset.data))
            self.assertTrue(torch.equal(val.dataset.data, old_val.dataset.data))
            np.testing.assert_array_equal(mean, old_mean)
            np.testing.assert_array_equal(std, old_std)

    def test_default_12_training_matches_the_historical_input_pipeline(self):
        config = dict(default_config(self.task, self.settings), time_lag=12)
        legacy = {key: value for key, value in config.items() if key != "time_lag"}
        legacy_settings = copy.deepcopy(self.settings)
        legacy_settings["space"].pop("time_lag")
        results = []
        for label, candidate, settings in (("legacy", legacy, legacy_settings), ("explicit12", config, self.settings)):
            folder = self.directory / label
            result = fit(self.task, candidate, settings, self.directory, folder, 2026, 1, 1, "compatibility")
            self.assertEqual(result["status"], "COMPLETE", result.get("error"))
            weights = torch.load(folder / result["checkpoint"], weights_only=True, map_location="cpu")
            results.append((result, weights))
        for metric in ("val_rmse", "val_loss", "val_mae", "train_windows", "val_windows"):
            self.assertEqual(results[0][0][metric], results[1][0][metric])
        for name, tensor in results[0][1].items():
            self.assertTrue(torch.equal(tensor, results[1][1][name]), name)

    def test_real_long_input_training_evaluation_and_cache_identity(self):
        for lag in (32, 64):
            config = dict(default_config(self.task, self.settings), time_lag=lag)
            fit_folder = self.directory / f"fit_{lag}"
            result = fit(self.task, config, self.settings, self.directory, fit_folder, 2026, 1, 1, "input_window_test")
            self.assertEqual(result["status"], "COMPLETE", result.get("error"))
            self.assertEqual(result["time_lag"], lag)
            self.assertEqual(result["target_alignment"], TARGET_ALIGNMENT)
            saved_request = read_json(fit_folder / "training.json")["training"]["request"]
            self.assertEqual(saved_request["input_window"]["time_lag"], lag)
            stats_path = fit_folder / result["stats"]
            with np.load(stats_path, allow_pickle=False) as stats:
                self.assertEqual(stats["time_lag"].item(), lag)
                self.assertEqual(stats["pre_len"].item(), 3)
                stats_values = {key: stats[key] for key in stats.files}
            destination = self.directory / f"test_{lag}"
            evaluated = evaluate_test(self.task, config, self.settings, self.directory, fit_folder, result, destination, 2026)
            self.assertEqual(evaluated["time_lag"], lag)
            self.assertEqual(evaluated["request"]["input_window"]["target_start_offset"], 12)
            self.assertEqual(read_test_cache(self.task, config, self.settings, result, destination, 2026), evaluated)
            with self.assertRaisesRegex(ValueError, "input window"):
                read_test_cache(self.task, dict(config, time_lag=12), self.settings, result, destination, 2026)
            corrupted = copy.deepcopy(evaluated)
            corrupted["request"]["input_window"]["time_lag"] = 12
            save_json(destination / "metrics.json", corrupted)
            with self.assertRaisesRegex(ValueError, "Test cache"):
                read_test_cache(self.task, config, self.settings, result, destination, 2026)
            corrupted = copy.deepcopy(evaluated)
            corrupted["target_start_offset"] = 64
            save_json(destination / "metrics.json", corrupted)
            with self.assertRaisesRegex(ValueError, "Test cache input-window metadata"):
                read_test_cache(self.task, config, self.settings, result, destination, 2026)
            stats_values["time_lag"] = np.array(12)
            np.savez(stats_path, **stats_values)
            with self.assertRaisesRegex(ValueError, "different input time_lag"):
                evaluate_test(self.task, config, self.settings, self.directory, fit_folder, result,
                              self.directory / f"test_wrong_stats_{lag}", 2026)


if __name__ == "__main__":
    unittest.main()
