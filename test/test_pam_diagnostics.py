import csv
import tempfile
import unittest
from pathlib import Path

from repro.diagnostics import append_csv, standard_run_log, summary_row
from repro.pam_validation import validate_log_fields


def dummy_result():
    return {
        "seed": 0,
        "training_seed": 0,
        "environment_seed": 2026,
        "permutation_seed": 42,
        "state_sampling_seed": 17,
        "env_name": "HM8",
        "action_order_name": "local",
        "canonical_ordering": "local",
        "display_ordering": "Local",
        "baseline_category": "primary_baseline",
        "pam_order_objective": 448.0,
        "pam_order_peak_cut_objective": 46.0,
        "pam_order_rankaware_proxy_objective": 58.2953,
        "train_time_sec": 13.8,
        "ordering_search_time_sec": 0.2,
        "ttpi_training_time_sec": 13.8,
        "total_time_sec": 14.0,
        "status": "ok",
        "oom": False,
        "error_type": "",
        "error_message": "",
        "final_metrics": {"success_rate": 0.0, "cum_reward_mean": -1.0, "mu_success": 0.0},
        "best_tradeoff_metrics": {"success_rate": 0.0, "mu_success": 0.0, "S_times_mu": 0.0},
    }


def dummy_train_data():
    return {
        "a_rank_profile": [[3, 2, 1]],
        "v_rank_profile": [[2, 1]],
        "p_rank_profile": [[4, 3, 2]],
    }


def dummy_diagnostics():
    return {
        "tt_cross_calls": 5,
        "tt_cross_function_evals": 681826,
        "queried_points_total": 681826,
        "cross_process_events": [],
    }


class PamDiagnosticsTests(unittest.TestCase):
    def test_summary_row_contains_required_fields(self):
        row = summary_row(
            result=dummy_result(),
            train_data=dummy_train_data(),
            diagnostics=dummy_diagnostics(),
            peak_memory_mb=648.7,
        )
        self.assertTrue(validate_log_fields(row))
        self.assertEqual(row["ordering_group"], "primary_baseline")
        self.assertEqual(row["training_seed"], 0)
        self.assertEqual(row["environment_seed"], 2026)
        self.assertEqual(row["permutation_seed"], 42)
        self.assertEqual(row["state_sampling_seed"], 17)
        self.assertEqual(row["runtime_sec"], 14.0)
        self.assertAlmostEqual(row["total_time_sec"], row["ordering_search_time_sec"] + row["ttpi_training_time_sec"])

    def test_standard_json_template_shape(self):
        log = standard_run_log(
            run_id="HM8_state20_action20_iter2_orderlocal_seed0",
            result=dummy_result(),
            train_data=dummy_train_data(),
            diagnostics=dummy_diagnostics(),
            peak_memory_mb=648.7,
        )
        self.assertEqual(log["objectives"]["la"], 448.0)
        self.assertEqual(log["tt_cross"]["function_evals_total"], 681826)
        self.assertEqual(log["ranks"]["adv_rank_profile"], [3, 2, 1])
        self.assertEqual(log["training_seed"], 0)
        self.assertEqual(log["resources"]["ordering_search_time_sec"], 0.2)
        self.assertEqual(log["resources"]["total_time_sec"], 14.0)
        self.assertFalse(log["error"]["oom"])

    def test_failed_run_summary_keeps_schema_without_performance_values(self):
        result = dummy_result()
        result.update(
            {
                "status": "oom",
                "oom": True,
                "error_type": "OutOfMemoryError",
                "error_message": "simulated",
                "final_metrics": {},
            }
        )
        row = summary_row(
            result=result,
            train_data=dummy_train_data(),
            diagnostics=dummy_diagnostics(),
            peak_memory_mb=999.0,
        )
        self.assertTrue(validate_log_fields(row))
        self.assertEqual(row["status"], "oom")
        self.assertTrue(row["oom"])
        self.assertEqual(row["success_rate"], "")

    def test_multi_round_csv_header_is_stable_when_fields_expand(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            append_csv(path, {"seed": 0, "env": "HM8"})
            append_csv(path, {"seed": 1, "env": "HM8", "ordering": "local"})
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["ordering"], "")
            self.assertEqual(rows[1]["ordering"], "local")


if __name__ == "__main__":
    unittest.main()
