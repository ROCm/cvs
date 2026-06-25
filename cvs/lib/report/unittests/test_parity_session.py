'''Phase E — inference parity session publishing.'''

from __future__ import annotations

import json
from pathlib import Path

from cvs.lib.report.parity.session import (
    publish_session_inference_parity,
    resolve_parity_compare_jsons,
)
from cvs.lib.report.types import InferenceReportConfig


def _sample_cell_report(path: Path, suite_id: str, tput: float) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": suite_id,
                "cells": [
                    {
                        "isl": "1024",
                        "osl": "1024",
                        "concurrency": 128,
                        "policy": "w1",
                        "actuals": {"client.output_throughput": tput},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_resolve_parity_compare_jsons_merges_env(monkeypatch):
    cfg = InferenceReportConfig(
        suite_id="s",
        report_basename="s_report",
        title="t",
        subtitle="",
        footer="",
        link_name="l",
        embed_summary="",
        results_columns=(),
        metric_tier_order=(),
        tier_metric_specs=lambda *_: {},
        metric_units={},
        parity_compare_jsons=(("vllm", "/preset/vllm.json"),),
    )
    monkeypatch.setenv("CVS_INFERENCE_PARITY_COMPARE", "sglang=/env/sglang.json")
    resolved = dict(resolve_parity_compare_jsons(cfg))
    assert resolved["vllm"] == "/preset/vllm.json"
    assert resolved["sglang"] == "/env/sglang.json"


def test_publish_session_inference_parity_writes_artifacts(tmp_path, monkeypatch):
    ref = tmp_path / "inferencex_atom_report.json"
    vllm = tmp_path / "vllm.json"
    _sample_cell_report(ref, "atom", 5000.0)
    _sample_cell_report(vllm, "vllm", 4500.0)

    cfg = InferenceReportConfig(
        suite_id="inferencex_atom",
        report_basename="inferencex_atom_report",
        title="IX",
        subtitle="",
        footer="",
        link_name="IX",
        embed_summary="",
        results_columns=(),
        metric_tier_order=("throughput",),
        tier_metric_specs=lambda *_: {},
        metric_units={},
        parity_reference_framework_id="atom",
        parity_compare_jsons=(("vllm", str(vllm)),),
    )

    class Opt:
        htmlpath = str(tmp_path / "pytest.html")

    class Cfg:
        option = Opt()

    class Mgr:
        is_enabled = True
        added = []

        def add_html_to_report(self, html_file, link_name=None, request=None):
            self.added.append((Path(html_file).name, link_name))

    mgr = Mgr()
    artifacts = publish_session_inference_parity(
        cfg,
        report_manager=mgr,
        pytest_config=Cfg(),
        reference_json_path=ref,
    )
    assert artifacts is not None
    assert artifacts["html"].is_file()
    assert artifacts["json"].is_file()
    assert any("parity" in name for name, _ in mgr.added)
