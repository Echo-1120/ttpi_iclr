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
        self.assertTrue(all("--n-iter 2" in " ".join(item["cmd"]) for item in commands))


if __name__ == "__main__":
    unittest.main()
