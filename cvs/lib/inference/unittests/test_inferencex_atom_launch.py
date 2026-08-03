'''Unit tests for inferencex_atom launch provenance helpers.'''

import unittest
from pathlib import Path

from cvs.lib.inference.inferencex_atom.inferencex_atom_config_loader import load_variant
from cvs.lib.inference.inferencex_atom.inferencex_atom_launch import (
    build_launch_provenance,
    launch_summary,
)


def _cluster_dict():
    return {
        "username": "tester",
        "head_node_dict": {"mgmt_ip": "10.0.0.1"},
        "node_dict": {"10.0.0.1": {"vpc_ip": "10.0.0.1"}},
    }


class TestInferenceXAtomLaunch(unittest.TestCase):
    def _smoke_config(self) -> Path:
        root = Path(__file__).resolve().parents[3]
        return root / (
            "input/config_file/inference/inferencex_atom_single/"
            "mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json"
        )

    def test_launch_summary_includes_driver_tp_and_max_model_len(self):
        variant = load_variant(self._smoke_config(), _cluster_dict())
        summary = launch_summary(variant)
        self.assertIn("atom", summary)
        self.assertIn("TP=8", summary)
        self.assertIn("max_model_len=", summary)

    def test_build_launch_provenance_includes_server_and_bench_commands(self):
        variant = load_variant(self._smoke_config(), _cluster_dict())
        prov = build_launch_provenance(variant)
        self.assertTrue(prov["launch_summary"])
        self.assertIn("atom.entrypoints.openai_server", prov["launch_server_cmd"])
        self.assertIn("benchmark_serving", prov["launch_bench_cmd"])
        self.assertTrue(prov["launch_example_cell"].startswith("ISL="))

    def test_build_launch_provenance_omits_commands_when_sweep_empty(self):
        variant = load_variant(self._smoke_config(), _cluster_dict())
        variant.sweep.runs = []
        prov = build_launch_provenance(variant)
        self.assertTrue(prov["launch_summary"])
        self.assertNotIn("launch_server_cmd", prov)
        self.assertNotIn("launch_bench_cmd", prov)
        self.assertNotIn("launch_example_cell", prov)


if __name__ == "__main__":
    unittest.main()
