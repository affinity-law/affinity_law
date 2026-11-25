"""Current per-seed L results must be aggregated without silently dropping seeds."""
import tempfile
import unittest
from pathlib import Path

from hpo.config import load_settings
from hpo.report import l_original_reference, write_csv


class CurrentReferenceTests(unittest.TestCase):
    def test_current_seed_means_and_sample_deviation(self):
        task = {"dataset": "example", "horizon": 3, "id": "example__h3"}
        rows = [{"dataset": "example", "horizon": 3, "seed": seed,
                 "rmse": value, "mae": value / 2}
                for seed, value in zip((2026, 2027, 2028), (8, 10, 12))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exp_L.csv"
            write_csv(path, rows, list(rows[0]))
            result = l_original_reference(task, load_settings(), path)
            self.assertEqual(result["l_original_test_rmse_mean"], 10)
            self.assertEqual(result["l_original_test_rmse_std"], 2)
            self.assertEqual(result["l_original_status"], "current_L_3seed_reference")
            for invalid in (rows[:2], rows + [rows[0]], [rows[0], rows[0], rows[2]]):
                write_csv(path, invalid, list(rows[0]))
                with self.assertRaisesRegex(ValueError, "three unique seeds"):
                    l_original_reference(task, load_settings(), path)


if __name__ == "__main__":
    unittest.main()
