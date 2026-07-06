import tempfile
import unittest
from pathlib import Path

from analysis.metrics import aggregate_rows, rank_auc, summarize_record
from analysis.result_loader import write_csv
from analysis.statistics import paired_method_tests


class PrestudyAnalysisTests(unittest.TestCase):
    def test_rank_auc_uses_iteration_order(self):
        rows = [
            {"iteration": "2", "adv_rank_max": "5"},
            {"iteration": "1", "adv_rank_max": "1"},
            {"iteration": "3", "adv_rank_max": "9"},
        ]
        self.assertEqual(rank_auc(rows), 10.0)

    def test_summarize_record_keeps_oom_and_partial_queries(self):
        record = {
            "run_id": "r0",
            "env": "HM12",
            "ordering": "rankaware_proxy_pam",
            "seed": 0,
            "status": "oom",
            "oom": True,
            "result": {
                "peak_memory_mb": 12000,
                "diagnostics": {"queried_points_total": 123, "tt_cross_calls": 2},
                "train_time_sec": 3.0,
                "final_metrics": {},
            },
            "rank_rows": [{"iteration": "1", "adv_rank_max": "7", "val_rank_max": "3"}],
            "cross_rows": [],
            "cross_process_rows": [{"rank_max": "8"}],
            "round_rows": [],
        }
        row = summarize_record(record)
        self.assertTrue(row["oom"])
        self.assertEqual(row["status"], "oom")
        self.assertEqual(row["queried_points_total"], 123)
        self.assertEqual(row["last_completed_pi"], 1)
        self.assertEqual(row["cross_process_rank_max"], 8)

    def test_aggregate_includes_oom_rate(self):
        rows = [
            {"env": "HM8", "ordering": "a", "status": "ok", "oom": False, "peak_memory_mb": 1.0},
            {"env": "HM8", "ordering": "a", "status": "oom", "oom": True, "peak_memory_mb": 2.0},
        ]
        agg = aggregate_rows(rows)
        self.assertEqual(agg[0]["runs"], 2)
        self.assertEqual(agg[0]["oom_runs"], 1)
        self.assertEqual(agg[0]["oom_rate"], 0.5)

    def test_statistics_downgrades_when_too_few_paired_seeds(self):
        rows = [
            {"env": "HM8", "ordering": "block_pam", "seed": 0, "metric": 1.0},
            {"env": "HM8", "ordering": "hybrid_pam", "seed": 0, "metric": 2.0},
        ]
        tests = paired_method_tests(
            rows,
            env="HM8",
            methods=["block_pam", "hybrid_pam"],
            metric="metric",
            baseline="block_pam",
        )
        comp = tests["comparisons"]["hybrid_pam_minus_block_pam"]
        self.assertEqual(comp["mode"], "descriptive_only")
        self.assertIsNone(comp["paired_permutation_p"])

    def test_write_csv_handles_empty_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            write_csv(path, [])
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(), "")


if __name__ == "__main__":
    unittest.main()
