import math
import unittest

from repro.pam_ordering import (
    ORDERING_NAMES,
    build_hardmove_orders,
    flip_within_block_order,
    peak_cut_cost,
    rankaware_proxy_cost,
    random_order,
    reverse_blocks_order,
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
        for name in [
            "local",
            "reverse_blocks",
            "flip_within_block",
            "opposite_pair",
            "block_pam",
            "sensitivity_pam",
            "sensitivity_lite_fd5",
            "sensitivity_lite_first_order",
            "hybrid_block_only",
            "hybrid_block_plus_physics",
            "hybrid_block_plus_sensitivity",
            "hybrid_current",
            "rankaware_proxy_pam",
            "hybrid_pam",
        ]:
            with self.subTest(ordering=name):
                validate_block_adjacency(orders[name].order, n_actuator=8)

    def test_formal_diagnostic_orders_have_exact_semantics(self):
        self.assertEqual(reverse_blocks_order(4), [6, 7, 4, 5, 2, 3, 0, 1])
        self.assertEqual(flip_within_block_order(4), [1, 0, 3, 2, 5, 4, 7, 6])

    def test_opposite_pair_is_legacy_interleave_not_reverse_or_flip(self):
        _, orders = build_hardmove_orders(n_actuator=8, random_seed=0)
        legacy = orders["opposite_pair"]
        self.assertEqual(legacy.metadata["canonical_name"], "opposite_interleave_legacy")
        self.assertTrue(legacy.metadata["deprecated_alias"])
        self.assertNotEqual(legacy.order, orders["reverse_blocks"].order)
        self.assertNotEqual(legacy.order, orders["flip_within_block"].order)

    def test_random_order_uses_reproducible_permutation_seed(self):
        self.assertEqual(random_order(8, seed=123), random_order(8, seed=123))
        self.assertNotEqual(random_order(8, seed=123), random_order(8, seed=124))
        _, orders = build_hardmove_orders(n_actuator=8, random_seed=123)
        self.assertEqual(orders["random"].metadata["permutation_seed"], 123)

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

    def test_peakcut_pam_metadata_matches_definition(self):
        coupling, orders = build_hardmove_orders(n_actuator=8, random_seed=1)
        result = orders["peakcut_pam"]
        self.assertEqual(result.metadata["baseline_category"], "surrogate_baseline")
        self.assertAlmostEqual(result.metadata["peak_cut_objective"], peak_cut_cost(result.order, coupling))

    def test_prestudy_ordering_metadata_documents_construction(self):
        _, orders = build_hardmove_orders(n_actuator=8, random_seed=1)
        self.assertEqual(orders["sensitivity_pam"].metadata["trajectory_count"], 0)
        self.assertEqual(orders["sensitivity_lite_fd5"].metadata["trajectory_count"], 5)
        self.assertGreater(orders["sensitivity_lite_fd5"].metadata["sensitivity_function_calls"], 0)
        self.assertEqual(
            orders["sensitivity_lite_first_order"].metadata["construction_fallback"],
            "autograd_not_required_for_hardmove_closed_form",
        )
        self.assertEqual(orders["hybrid_current"].metadata["hybrid_physics_weight"], 0.5)
        self.assertEqual(orders["hybrid_current"].order, orders["hybrid_pam"].order)
        self.assertEqual(orders["hybrid_block_plus_physics"].order, orders["block_pam"].order)
        self.assertEqual(orders["hybrid_block_plus_sensitivity"].order, orders["sensitivity_pam"].order)


if __name__ == "__main__":
    unittest.main()
