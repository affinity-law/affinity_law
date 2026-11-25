"""Regression checks for detailed tables and historical evidence preservation."""

import copy
import csv
import tempfile
import unittest
from pathlib import Path

from hpo.artifacts import save_json
from hpo.config import ROOT
from hpo.slot_table import TABLE_NAME, backfill_run_slot_tables, slot_rows, write_slot_tables


class SlotTableTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".slot_table_test_", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.task = {"dataset": "Example dataset", "horizon": 3, "id": "demo__h3"}
        self.settings = {"time_lag": 12, "train_ratio": 0.6, "val_ratio": 0.2,
                         "training": {"short_epochs": 300, "short_patience": 15,
                                      "search_train_seed": 2026, "mse_weight": 0.2, "mae_weight": 0.8}}
        self.config = {"lr": 0.001, "weight_decay": 0.0, "batch_size": 16, "mlp_width_scale": 1.0}

    def record(self, slot=1, config=None, score=2.0):
        return {"slot": slot, "executed_config": config or dict(self.config),
                "proposal_config": dict(self.config), "phase": "baseline" if slot == 1 else "coordinate",
                "search_round": "baseline" if slot == 1 else "local", "base_slot": None if slot == 1 else 1,
                "parent_slots": [] if slot == 1 else [1], "proposal_source": "default",
                "execution_source": "default", "decision": None,
                "result": {"seed": 2026, "status": "COMPLETE", "val_rmse": score, "params": 100,
                           "best_epoch": 4, "epochs_run": 19}}

    def read(self, path):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_two_seed_results_and_llm_change_are_individual_columns(self):
        baseline = self.record()
        changed = dict(self.config, lr=0.003, loss_profile="mse")
        record = self.record(2, changed, 3.0)
        record.update(search_priority="training", effective_priority="training",
                      screening_status="COMPLETE", screening_val_rmse_mean=2.0,
                      screening_results={"2026": {"result": record["result"]},
                                         "2027": {"result": {"seed": 2027, "status": "COMPLETE", "val_rmse": 1.0}}},
                      decision={"action": "edit_proposal", "fallback": False,
                                "evidence_trials": [1], "hypothesis": "Check loss and learning rate."})
        folder = self.run / "tasks" / self.task["dataset"] / "h3"
        write_slot_tables(folder, [baseline, record], self.settings, self.task)
        path = folder / TABLE_NAME
        self.assertTrue(path.read_bytes().startswith(b"\xef\xbb\xbf"))
        rows = self.read(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["lr"], "0.003")
        self.assertEqual(rows[1]["proposal_lr"], "0.001")
        self.assertEqual(rows[1]["loss_profile"], "mse")
        self.assertEqual(rows[1]["mse_weight"], "1.0")
        self.assertEqual(rows[1]["mae_weight"], "0.0")
        self.assertEqual(rows[1]["seed_2026_val_rmse"], "3.0")
        self.assertEqual(rows[1]["seed_2027_val_rmse"], "1.0")
        self.assertEqual(rows[1]["screening_val_rmse_mean"], "2.0")
        self.assertIn("lr: 0.001 -> 0.003", rows[1]["llm_changes"])
        self.assertEqual(rows[1]["base_slot"], "1")
        self.assertEqual(rows[1]["search_priority"], "training")

    def test_failed_second_seed_is_not_reported_as_successful_mean(self):
        record = self.record()
        record.update(screening_status="FAILED", screening_results={
            "2026": {"result": record["result"]},
            "2027": {"result": {"status": "FAILED", "seed": 2027, "val_rmse": None}},
        })
        row = slot_rows([record], self.settings, self.task)[0]
        self.assertEqual(row["status"], "FAILED")
        self.assertIsNone(row["screening_val_rmse_mean"])
        self.assertEqual(row["screening_seed_count"], 2)

    def test_input_window_column_uses_executed_value_and_preserves_horizon(self):
        baseline = self.record(config={**self.config, "time_lag": 12})
        changed = self.record(2, {**self.config, "time_lag": 64})
        changed["proposal_config"] = {**self.config, "time_lag": 32}
        folder = self.run / "tasks" / self.task["dataset"] / "h3"
        write_slot_tables(folder, [baseline, changed], self.settings, self.task)
        rows = self.read(folder / TABLE_NAME)
        self.assertEqual([row["time_lag"] for row in rows], ["12", "64"])
        self.assertEqual(rows[1]["proposal_time_lag"], "32")
        self.assertEqual([row["horizon"] for row in rows], ["3", "3"])
        self.assertEqual(self.read(folder.parent / TABLE_NAME), rows)

    def test_crossover_parent_pool_preserves_expansion_and_missing_history(self):
        legacy = self.record()
        regular = self.record(2)
        regular.update(parent_pool_slots=[2, 1, 4, 3], parent_pool_size=4, parent_pool_expanded=False)
        expanded = self.record(3)
        expanded.update(parent_pool_slots=[3, 2, 1, 4, 5], parent_pool_size=5, parent_pool_expanded=True,
                        proposal_source="expanded_elite_crossover", execution_source="expanded_elite_crossover")
        folder = self.run / "tasks" / self.task["dataset"] / "h3"
        write_slot_tables(folder, [legacy, regular, expanded], self.settings, self.task)
        rows = self.read(folder / TABLE_NAME)
        self.assertEqual([row["parent_pool_expanded"] for row in rows], ["", "False", "True"])
        self.assertEqual(rows[0]["parent_pool_slots"], "")
        self.assertEqual(rows[0]["parent_pool_size"], "")
        self.assertEqual(rows[1]["parent_pool_slots"], "2;1;4;3")
        self.assertEqual(rows[1]["parent_pool_size"], "4")
        self.assertEqual(rows[2]["parent_pool_slots"], "3;2;1;4;5")
        self.assertEqual(rows[2]["parent_pool_size"], "5")
        self.assertEqual(rows[2]["execution_source"], "expanded_elite_crossover")
        self.assertEqual(self.read(folder.parent / TABLE_NAME), rows)

    def test_legacy_weights_are_read_from_saved_protocol(self):
        settings = copy.deepcopy(self.settings)
        settings["training"].update(mse_weight=0.3, mae_weight=0.7)
        record = self.record()
        row = slot_rows([record], settings, self.task)[0]
        self.assertEqual(row["mse_weight"], 0.3)
        self.assertEqual(row["mae_weight"], 0.7)
        self.assertEqual(row["loss_profile"], "legacy_0.3_mse_0.7_mae")
        self.assertEqual(row["loss_profile_source"], "frozen_training_weights")
        self.assertEqual(row["screening_seeds"], "2026")
        self.assertNotIn("seed_2027_val_rmse", row)
        self.assertEqual(row["screening_val_rmse_mean"], 2.0)

    def test_dataset_table_unions_horizons_and_model_parameter_columns(self):
        folder = self.run / "tasks" / self.task["dataset"]
        write_slot_tables(folder / "h3", [self.record()], self.settings, self.task)
        other = dict(self.task, horizon=12, id="demo__h12")
        config = {"lr": 0.003, "batch_size": 8, "weight_decay": 0, "lstm_layers": 2}
        write_slot_tables(folder / "h12", [self.record(config=config)], self.settings, other)
        rows = self.read(folder / TABLE_NAME)
        self.assertEqual([row["horizon"] for row in rows], ["3", "12"])
        self.assertEqual(rows[0]["mlp_width_scale"], "1.0")
        self.assertEqual(rows[0]["lstm_layers"], "")
        self.assertEqual(rows[1]["lstm_layers"], "2")

    def test_backfill_adds_only_new_tables_and_preserves_existing_bytes(self):
        other = dict(self.task, horizon=12, id="demo__h12")
        save_json(self.run / "tuning_protocol.json", {"settings": self.settings, "tasks": [self.task, other]})
        originals = {}
        for task in (self.task, other):
            folder = self.run / "tasks" / task["dataset"] / f"h{task['horizon']}"
            record_path = folder / "trials" / "trial_001" / "trial.json"
            save_json(record_path, {"proposal": {"search_priority": "baseline"}, "record": self.record()})
            (folder / "trials.csv").write_bytes(b"original trial CSV\r\n")
            originals[record_path] = record_path.read_bytes()
            originals[folder / "trials.csv"] = (folder / "trials.csv").read_bytes()
        paths = backfill_run_slot_tables(self.run)
        self.assertEqual(len(paths), 3)
        combined = self.read(self.run / "tasks" / self.task["dataset"] / TABLE_NAME)
        self.assertEqual(len(combined), 2)
        self.assertEqual(combined[0]["search_priority"], "baseline")
        originals.update({path: path.read_bytes() for path in paths})
        self.assertEqual(backfill_run_slot_tables(self.run), [])
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
