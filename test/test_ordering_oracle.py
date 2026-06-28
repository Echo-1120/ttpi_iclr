import unittest

from repro.oracle_ordering import (
    expected_oracle_count,
    iter_oracle_orders,
    oracle_rows,
)


class OrderingOracleTests(unittest.TestCase):
    def test_expected_oracle_counts_are_fixed(self):
        self.assertEqual(expected_oracle_count(4, "full_modes"), 40320)
        self.assertEqual(expected_oracle_count(4, "block_with_flips"), 384)
        self.assertEqual(expected_oracle_count(6, "block_with_flips"), 46080)

    def test_hm4_block_with_flips_rows_have_normalized_regret(self):
        rows = oracle_rows(n_actuator=4, mode="block_with_flips", random_seed=0)
        self.assertEqual(len(rows), 384)
        for metric in ("la_objective", "peak_cut_objective", "rankaware_proxy_objective"):
            regrets = [float(row[f"normalized_regret_{metric}"]) for row in rows]
            self.assertAlmostEqual(min(regrets), 0.0)
            self.assertLessEqual(max(regrets), 1.0)
            self.assertGreaterEqual(min(regrets), 0.0)

    def test_full_mode_oracle_is_limited_to_hm4(self):
        with self.assertRaises(ValueError):
            list(iter_oracle_orders(5, "full_modes"))


if __name__ == "__main__":
    unittest.main()
