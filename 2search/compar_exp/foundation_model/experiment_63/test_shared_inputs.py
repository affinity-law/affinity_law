"""Shared input loading must preserve the frozen inference resume contract."""
import copy
import unittest
from unittest.mock import patch

import common
import portable_paths
import run_portable


class SharedInputTests(unittest.TestCase):
    def test_shared_manifest_matches_tuning_and_current_inputs_exist(self):
        manifest, tasks = portable_paths.load_manifest()
        tuning_manifest = common.read_json(portable_paths.TUNING_ROOT / "tasks.json")
        self.assertEqual(portable_paths.shared_input_signature(manifest),
                         portable_paths.shared_input_signature(tuning_manifest))
        self.assertEqual(len(tasks), 126)
        for task in tasks:
            for field in ("data_path", "model_path", "weights_path"):
                path = (common.MODEL_ROOT / task[field]).resolve()
                self.assertTrue(path.is_file(), path)
                self.assertTrue(path.is_relative_to(portable_paths.PROJECT_ROOT), path)
                self.assertFalse(path.is_relative_to(common.MODEL_ROOT), path)

    def test_current_l_reference_has_complete_three_seed_grid(self):
        path, rows = portable_paths.load_l_reference()
        self.assertEqual(path, portable_paths.RESULT_ROOT / "exp_L.csv")
        _, tasks = portable_paths.load_manifest()
        self.assertEqual(set(rows), {task["id"] for task in tasks})
        for row in rows.values():
            self.assertEqual(tuple(sorted(row["seed_rows"])), portable_paths.L_SEEDS)
            self.assertEqual(row["seed_count"], 3)

    def test_loading_keeps_the_frozen_snapshot_and_metadata_intact(self):
        before = portable_paths.TASKS_PATH.read_bytes()
        original = common.read_json(portable_paths.TASKS_PATH)
        preserved = copy.deepcopy(original)
        mapped = portable_paths.normalize_manifest_paths(original)
        path_fields = {"data_path", "model_path", "weights_path", "source_model"}
        for old, new in zip(original["tasks"], mapped["tasks"]):
            self.assertEqual({k: v for k, v in old.items() if k not in path_fields},
                             {k: v for k, v in new.items() if k not in path_fields})
        self.assertEqual(original, preserved)
        portable_paths.load_manifest()
        self.assertEqual(portable_paths.TASKS_PATH.read_bytes(), before)

    def test_runner_uses_frozen_digest_and_restores_globals_after_failure(self):
        original_loader = common.load_manifest
        original_path = common.TASKS_PATH
        original_description = run_portable.runner.__doc__
        digest = common.sha256_file(portable_paths.TASKS_PATH)

        def probe():
            self.assertEqual(common.sha256_file(common.TASKS_PATH), digest)
            _, tasks = common.load_manifest()
            self.assertTrue((common.MODEL_ROOT / tasks[0]["data_path"]).is_file())
            raise RuntimeError("probe completed")

        with patch.object(run_portable.runner, "main", side_effect=probe):
            with self.assertRaisesRegex(RuntimeError, "probe completed"):
                run_portable.main()
        self.assertIs(common.load_manifest, original_loader)
        self.assertEqual(common.TASKS_PATH, original_path)
        self.assertEqual(run_portable.runner.__doc__, original_description)


if __name__ == "__main__":
    unittest.main()
