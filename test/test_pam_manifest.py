import json
import unittest
from pathlib import Path

from repro.scripts.run_pam_ablation import build_run_commands, result_path


ROOT = Path(__file__).resolve().parents[1]


class PamManifestTests(unittest.TestCase):
    def test_manifest_enumerates_expected_number_of_runs(self):
        manifest = json.loads((ROOT / "repro" / "experiments" / "pam_ablation_manifest.json").read_text())
        commands = build_run_commands(manifest)
        expected = len(manifest["environments"]) * len(manifest["seeds"]) * len(manifest["orderings"])
        self.assertEqual(len(commands), expected)
        self.assertEqual(expected, 840)

    def test_output_path_matches_run_naming_convention(self):
        defaults = {"n_state": 40, "n_action": 50, "n_iter": 30}
        path = result_path("HM8_index_permuted", defaults, "block_pam", 7)
        self.assertTrue(str(path).endswith("repro/results/HM8_index_permuted_state40_action50_iter30_orderblock_pam_seed7.json"))

    def test_smoke_manifest_is_small_and_deterministic(self):
        manifest = json.loads((ROOT / "repro" / "experiments" / "pam_ablation_manifest.json").read_text())
        commands = build_run_commands(manifest, smoke=True)
        self.assertEqual(len(commands), 4)
        self.assertTrue(all(item["seed"] == 0 for item in commands))
        self.assertTrue(all(item["training_seed"] == 0 for item in commands))
        self.assertTrue(all(item["permutation_seed"] == 42 for item in commands))
        self.assertTrue(all("--n-iter 2" in " ".join(item["cmd"]) for item in commands))
        self.assertTrue(all("--training-seed 0" in " ".join(item["cmd"]) for item in commands))
        self.assertTrue(all("--permutation-seed 42" in " ".join(item["cmd"]) for item in commands))

    def test_manifest_can_expand_permutation_seed_dimension(self):
        manifest = {
            "training_seeds": [0],
            "permutation_seeds": [3, 4],
            "orderings": ["random"],
            "environments": [{"name": "HM8", "n_actuator": 8, "env_variant": "standard"}],
            "defaults": {"n_state": 20, "n_action": 20, "n_iter": 2},
        }
        commands = build_run_commands(manifest)
        self.assertEqual(len(commands), 2)
        self.assertEqual({item["permutation_seed"] for item in commands}, {3, 4})
        self.assertTrue(all("permseed" in item["result"] for item in commands))

    def test_stage_two_smoke_manifest_only_repeats_random_permutation_seeds(self):
        manifest = json.loads((ROOT / "repro" / "experiments" / "pam_baseline_smoke_manifest.json").read_text())
        commands = build_run_commands(manifest, smoke=True)
        orderings = [item["ordering"] for item in commands]
        random_items = [item for item in commands if item["ordering"] == "random"]
        non_random_items = [item for item in commands if item["ordering"] != "random"]

        self.assertEqual(len(commands), 11)
        self.assertEqual(len(random_items), 3)
        self.assertEqual(len(non_random_items), 8)
        self.assertEqual({item["permutation_seed"] for item in random_items}, {0, 1, 2})
        self.assertTrue(all(item["permutation_seed"] == 0 for item in non_random_items))
        self.assertEqual(len(set(orderings)), 9)
        self.assertTrue(all(item["env"] == "HM8" for item in commands))
        self.assertTrue(all(item["training_seed"] == 0 for item in commands))
        self.assertTrue(all("stage2smoke" in item["result"] for item in commands))
        self.assertTrue(all("permseed" in item["result"] for item in random_items))
        self.assertTrue(all("permseed" not in item["result"] for item in non_random_items))


if __name__ == "__main__":
    unittest.main()
