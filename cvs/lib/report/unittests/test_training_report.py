'''Training report and normalize unit tests.'''

from pathlib import Path

from cvs.lib.report.presets.megatron_8b_single import MEGATRON_LLAMA3_8B_SINGLE_REPORT_CONFIG
from cvs.lib.report.training import write_training_report
from cvs.lib.report.training_normalize import megatron_to_training_res_dict


def test_megatron_to_training_res_dict_single_node():
    raw = {
        "throughput_per_gpu": ["100.0", "110.5"],
        "tokens_per_gpu": ["5000"],
        "elapsed_time_per_iteration": ["120.0"],
        "mem_usage": ["42.0"],
    }
    nodes = megatron_to_training_res_dict(raw, ["10.0.0.1"])
    assert "10.0.0.1" in nodes
    assert nodes["10.0.0.1"]["throughput_per_gpu"] == "110.5"
    assert nodes["10.0.0.1"]["mem_usages"] == "42.0"


def test_write_training_report(tmp_path):
    training_res = {
        "node-a": {
            "throughput_per_gpu": "99.5",
            "tokens_per_gpu": "4000",
            "elapsed_time_per_iteration": "130.0",
            "nan_iterations": 0,
            "mem_usages": "40.0",
        }
    }
    artifacts = write_training_report(
        tmp_path / "megatron_report.html",
        config=MEGATRON_LLAMA3_8B_SINGLE_REPORT_CONFIG,
        training_res_dict=training_res,
    )
    html = artifacts["html"].read_text(encoding="utf-8")
    assert "Megatron training report" in html
    assert "node-a" in html
    assert '"report_kind": "training"' in artifacts["json"].read_text(encoding="utf-8")
