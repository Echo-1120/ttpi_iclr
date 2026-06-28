import unittest

try:
    import torch

    from repro.hardmove_variants import (
        canonical_env_variant,
        make_env_permutation,
        transform_hardmove_action,
        variant_display_name,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local test env.
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(IMPORT_ERROR is not None, f"torch unavailable: {IMPORT_ERROR}")
class HardMoveVariantTests(unittest.TestCase):
    def setUp(self):
        self.action = torch.tensor(
            [
                [0.1, 1.0, -0.2, 0.0, 0.3, 1.0, -0.4, 0.0],
                [0.5, 0.0, -0.6, 1.0, 0.7, 0.0, -0.8, 1.0],
            ],
            dtype=torch.float64,
        )

    def test_standard_does_not_change_action_values(self):
        out = transform_hardmove_action(self.action, n_actuator=4, variant="standard")
        self.assertTrue(torch.equal(out, self.action))
        self.assertIsNot(out, self.action)

    def test_index_permuted_is_legacy_alias_for_actuator_relabelled(self):
        permutation = [2, 0, 3, 1]
        legacy = transform_hardmove_action(
            self.action,
            n_actuator=4,
            variant="index_permuted",
            permutation=permutation,
        )
        canonical = transform_hardmove_action(
            self.action,
            n_actuator=4,
            variant="actuator_relabelled",
            permutation=permutation,
        )
        self.assertEqual(canonical_env_variant("index_permuted"), "actuator_relabelled")
        self.assertEqual(variant_display_name("index_permuted"), "actuator_relabelled")
        self.assertTrue(torch.equal(legacy, canonical))

    def test_cross_coupled_zero_beta_matches_standard(self):
        standard = transform_hardmove_action(self.action, n_actuator=4, variant="standard")
        coupled = transform_hardmove_action(
            self.action,
            n_actuator=4,
            variant="cross_coupled",
            cross_coupling_strength=0.0,
        )
        self.assertTrue(torch.equal(coupled, standard))

    def test_permutation_seed_is_reproducible(self):
        self.assertEqual(make_env_permutation(8, 2026), make_env_permutation(8, 2026))
        self.assertNotEqual(make_env_permutation(8, 2026), make_env_permutation(8, 2027))
        self.assertEqual(sorted(make_env_permutation(8, 2026)), list(range(8)))


if __name__ == "__main__":
    unittest.main()
