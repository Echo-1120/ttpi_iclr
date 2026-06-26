import math
import unittest

from repro.pam_ordering import (
    ORDERING_NAMES,
    build_hardmove_orders,
    rankaware_proxy_cost,
    segmented_cut_costs,
    weighted_linear_arrangement,
    weighted_segmented_cut_sum,
)
from repro.pam_validation import validate_block_adjacency, validate_objective_identity, validate_permutation


class PamOrderingTests(unittest.TestCase):
    def test_all_registered_orderings_are_valid_permutations(self):
        coupling, orders = build_hardmove_orders(n_actuator=8, random_seed=0)
        self.assertEqual(set(ORDERING_NAMES), set(orders))
        for name, result in orders.items():
            with self.subTest(ordering=name):
                self.assertEqual(validate_permutation(result.order, len(coupling)), result.order)

    def test_block_preserving_orderings_keep_actuator_pairs_adjacent(self):
        _, orders = build_hardmove_orders(n_actuator=8, random_seed=0)
        for name in ["local", "opposite_pair", "block_pam", "sensitivity_pam", "rankaware_proxy_pam", "hybrid_pam"]:
            with self.subTest(ordering=name):
                validate_block_adjacency(orders[name].order, n_actuator=8)

    def test_linear_arrangement_equals_segmented_cut_sum(self):
        coupling, orders = build_hardmove_orders(n_actuator=6, random_seed=3)
        for name, result in orders.items():
            with self.subTest(ordering=name):
                validate_objective_identity(result.order, coupling)
                self.assertAlmostEqual(
                    weighted_linear_arrangement(result.order, coupling),
                    weighted_segmented_cut_sum(result.order, coupling),
                )
                self.assertGreaterEqual(max(segmented_cut_costs(result.order, coupling), default=0.0), 0.0)

    def test_rankaware_proxy_metadata_matches_definition(self):
        coupling, orders = build_hardmove_orders(n_actuator=8, random_seed=1)
        result = orders["rankaware_proxy_pam"]
        expected = rankaware_proxy_cost(result.order, coupling, lambda_sum=1.0, lambda_peak=2.0)
        self.assertTrue(math.isclose(result.metadata["rankaware_proxy_objective"], expected, rel_tol=0, abs_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
