'''Inference parity unit tests.'''

import json
from pathlib import Path

from cvs.lib.report.parity.inference import (
    build_inference_parity_config,
    build_inference_parity_payload,
    build_session_parity_config,
    render_inference_parity_html,
    write_inference_parity_report,
)
from cvs.lib.report.presets.inference_parity import w1_triple_parity_config
from cvs.lib.report.types import (
    InferenceParityConfig,
    InferenceParityMetric,
    InferenceParitySource,
    InferenceReportConfig,
)


def _sample_report(framework: str, tput: float, ttft: float = 100.0) -> dict:
    return {
        "schema_version": 1,
        "suite_id": framework,
        "cells": [
            {
                "isl": "1024",
                "osl": "1024",
                "concurrency": 128,
                "policy": "w1",
                "actuals": {
                    "client.output_throughput": tput,
                    "client.mean_ttft_ms": ttft,
                },
            }
        ],
    }


def _write_json(path: Path, data: dict) -> str:
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_build_inference_parity_payload_ratios(tmp_path):
    atom = _write_json(tmp_path / "atom.json", _sample_report("atom", 4000.0, 80.0))
    vllm = _write_json(tmp_path / "vllm.json", _sample_report("vllm", 3600.0, 88.0))
    config = InferenceParityConfig(
        sources=(
            InferenceParitySource("atom", "ATOM", atom),
            InferenceParitySource("vllm", "vLLM", vllm),
        ),
        metrics=(
            InferenceParityMetric(
                "client.output_throughput",
                "vllm",
                "compare.vllm.output_throughput_ratio",
            ),
            InferenceParityMetric(
                "client.mean_ttft_ms",
                "vllm",
                "compare.vllm.mean_ttft_ms_ratio",
            ),
        ),
    )
    payload = build_inference_parity_payload(config)
    row = payload["rows"][0]
    assert row["compare"]["compare.vllm.output_throughput_ratio"] == 0.9
    assert row["compare"]["compare.vllm.mean_ttft_ms_ratio"] == 1.1
    assert "compare.vllm" in render_inference_parity_html(payload)


def test_w1_triple_preset_write(tmp_path):
    atom = _write_json(tmp_path / "atom.json", _sample_report("atom", 5000.0))
    vllm = _write_json(tmp_path / "vllm.json", _sample_report("vllm", 4500.0))
    config = w1_triple_parity_config(atom_json=atom, vllm_json=vllm)
    out = tmp_path / "parity.html"
    artifacts = write_inference_parity_report(out, config=config)
    assert artifacts["html"].is_file()
    assert artifacts["json"].is_file()
    assert '"report_kind": "inference_parity"' in artifacts["json"].read_text(encoding="utf-8")


def test_build_inference_parity_config_defaults(tmp_path):
    ref = _write_json(tmp_path / "ref.json", _sample_report("ref", 2000.0))
    other = _write_json(tmp_path / "other.json", _sample_report("other", 1800.0))
    config = build_inference_parity_config(
        reference=InferenceParitySource("ref", "Ref", ref),
        comparators=(InferenceParitySource("other", "Other", other),),
    )
    payload = build_inference_parity_payload(config)
    row = payload["rows"][0]
    assert row["compare"]["compare.other.output_throughput_ratio"] == 0.9


def test_build_session_parity_config(tmp_path):
    ref = _write_json(tmp_path / "suite.json", _sample_report("suite", 3000.0))
    cmp_path = _write_json(tmp_path / "cmp.json", _sample_report("cmp", 2700.0))
    report_config = InferenceReportConfig(
        suite_id="demo_suite",
        report_basename="demo_suite_report",
        title="Demo",
        subtitle="",
        footer="",
        link_name="Demo report",
        embed_summary="",
        results_columns=(),
        metric_tier_order=(),
        tier_metric_specs=lambda *_: {},
        metric_units={},
        parity_compare_jsons=(("cmp", cmp_path),),
    )
    parity = build_session_parity_config(report_config, ref)
    assert parity is not None
    assert parity.reference_framework_id == "demo_suite"
    payload = build_inference_parity_payload(parity)
    assert payload["rows"][0]["compare"]["compare.cmp.output_throughput_ratio"] == 0.9
