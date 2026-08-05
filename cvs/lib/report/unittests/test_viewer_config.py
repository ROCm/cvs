'''Tests for profile-driven interactive viewer configuration.'''

import json
from pathlib import Path

from cvs.lib.report.rundeck.config_adapter import resolve_report_config
from cvs.lib.report.rundeck.viewer_config import build_viewer_config
from cvs.lib.report.unittests._fixtures import generic_inference_report_config

_PROFILES = Path(__file__).resolve().parents[1] / "profiles"


def test_build_viewer_config_from_inference_profile():
    profile = json.loads((_PROFILES / "inferencex_atom_single.json").read_text(encoding="utf-8"))
    config = resolve_report_config(profile)
    vc = build_viewer_config(profile, config)

    assert vc["group_by"] == ["isl", "osl"]
    assert {f["field"] for f in vc["filters"]} >= {"isl", "osl", "policy", "host"}
    assert "client.output_throughput" in vc["metrics"]
    assert "client.output_throughput" in vc["heatmap_metrics"]
    assert any(col.get("field") == "isl" for col in vc["table_columns"])
    assert vc["interactivity"]["enabled"] is True
    assert vc["interactivity"]["tpot_metric"] == "client.mean_tpot_ms"


def test_build_viewer_config_legacy_config_only():
    config = generic_inference_report_config()
    vc = build_viewer_config({}, config)

    assert vc["group_by"] == ["isl", "osl"]
    assert vc["default_heatmap_metric"] == config.headline_metric
    assert len(vc["table_columns"]) >= 5


def test_inference_payload_includes_viewer_config():
    from cvs.lib.report.inference_payload import build_inference_report_payload
    from cvs.lib.report.unittests._fixtures import generic_variant, two_cell_inf_res

    config = generic_inference_report_config()
    payload = build_inference_report_payload(
        config=config,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
    )
    assert "viewer_config" in payload
    assert payload["viewer_config"]["group_by"] == ["isl", "osl"]
