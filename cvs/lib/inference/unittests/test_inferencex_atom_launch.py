'''Unit tests for inferencex_atom launch provenance helpers.'''

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


def test_launch_summary_includes_driver_tp_and_max_model_len():
    root = Path("cvs")
    variant = load_variant(
        root / "input/config_file/inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json",
        _cluster_dict(),
    )
    summary = launch_summary(variant)
    assert "atom" in summary
    assert "TP=8" in summary
    assert "max_model_len=" in summary


def test_build_launch_provenance_includes_server_and_bench_commands():
    root = Path("cvs")
    variant = load_variant(
        root / "input/config_file/inference/inferencex_atom/mi300x_inferencex-atom-single_deepseek-r1_fp8_smoke_config.json",
        _cluster_dict(),
    )
    prov = build_launch_provenance(variant)
    assert prov["launch_summary"]
    assert "atom.entrypoints.openai_server" in prov["launch_server_cmd"]
    assert "benchmark_serving" in prov["launch_bench_cmd"]
    assert prov["launch_example_cell"].startswith("ISL=")
