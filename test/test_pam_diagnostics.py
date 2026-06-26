import csv
import tempfile
import unittest
from pathlib import Path

from repro.diagnostics import append_csv, standard_run_log, summary_row
from repro.pam_validation import validate_log_fields


def dummy_result():
    return {
        "seed": 0,
        "env_name": "HM8",
        "action_order_name": "local",
        "pam_order_objective": 448.0,
        "pam_order_peak_cut_objective": 46.0,
        "pam_order_rankaware_proxy_objective": 58.2953,
        "train_time_sec": 13.8,
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
        self.assertEqual(row["ordering_group"], "baseline")

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
