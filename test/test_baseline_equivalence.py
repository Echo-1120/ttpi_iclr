import unittest

from repro.baseline_equivalence import (
    action_inverse_from_order,
    check_index_roundtrip,
    decode_tt_indices_to_physical,
    physical_indices_to_tt,
)


class BaselineEquivalencePureTests(unittest.TestCase):
    def test_local_inverse_is_identity(self):
        order = list(range(16))
        self.assertEqual(action_inverse_from_order(order), order)

    def test_bad_order_roundtrip_restores_physical_indices(self):
        physical = [[0, 1, 2, 0, 3, 1, 4, 0]]
        order = [0, 2, 4, 6, 1, 3, 5, 7]
        inverse = action_inverse_from_order(order)
        tt_rows = physical_indices_to_tt(physical, order)
        self.assertEqual(decode_tt_indices_to_physical(tt_rows, inverse), physical)

    def test_random_index_roundtrip_for_reordered_modes(self):
        order = [6, 7, 4, 5, 2, 3, 0, 1]
        result = check_index_roundtrip(
            action_order=order,
            n_actuator=4,
            n_action=11,
            n_samples=2048,
            seed=123,
        )
        self.assertTrue(result.passed, result.details)


try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent.
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


@unittest.skipIf(TORCH_IMPORT_ERROR is not None, f"torch unavailable: {TORCH_IMPORT_ERROR}")
class BaselineEquivalenceTorchTests(unittest.TestCase):
    def test_hardmove_local_decode_matches_original(self):
        from repro.baseline_equivalence import check_hardmove_local_action_equivalence

        result = check_hardmove_local_action_equivalence(
            n_actuator=4,
            n_state=9,
            n_action=7,
            n_samples=256,
            seed=0,
            device="cpu",
        )
        self.assertTrue(result.passed, result.details)


if __name__ == "__main__":
    unittest.main()
