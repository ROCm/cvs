'''Parity and integration tests for the unified Run Deck engine.'''

from cvs.lib.report.inference import build_inference_report_payload, render_report_html
from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.unittests._fixtures import generic_sweep_profile, generic_variant, two_cell_inf_res


def test_json_profile_resolves_expected_config():
    profile = generic_sweep_profile()
    cfg = build_inference_config_from_profile(profile)
    assert cfg.suite_id == "test_inference_suite"
    assert cfg.report_basename == "test_inference_suite_run_deck"
    assert cfg.metric_tier_order == ("throughput", "record")
    assert len(cfg.chart_series) == 3


def test_rundeck_payload_matches_inference_builder():
    profile = generic_sweep_profile()
    store = {
        "inf_res_dict": two_cell_inf_res(),
        "variant_config": generic_variant(),
        "lifecycle_report": {},
    }
    config = build_inference_config_from_profile(profile)
    rundeck_payload = build_rundeck_payload(profile=profile, store=store, cvs_version="1.0.0")
    inference_payload = build_inference_report_payload(
        config=config,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
        cvs_version="1.0.0",
    )
    assert len(rundeck_payload["cells"]) == len(inference_payload["cells"])
    assert rundeck_payload["results_table"]["headers"] == inference_payload["results_table"]["headers"]
    assert len(rundeck_payload["gate_matrix"]) == len(inference_payload["gate_matrix"])
    assert rundeck_payload["chart_series"].keys() == inference_payload["chart_series"].keys()
    assert "viewer_config" in rundeck_payload
    assert rundeck_payload["viewer_config"]["group_by"] == ["isl", "osl"]


def test_rundeck_render_contains_core_panels():
    profile = generic_sweep_profile()
    payload = build_rundeck_payload(
        profile=profile,
        store={
            "inf_res_dict": two_cell_inf_res(),
            "variant_config": generic_variant(),
            "lifecycle_report": {},
        },
        cvs_version="1.0.0",
    )
    doc = render_rundeck_html(payload)
    assert "Test Sweep Run Deck" in doc
    assert "report-nav" in doc
    assert "Gate matrix" in doc
    assert "Full results" in doc
    assert "Sweep analytics" in doc


def test_inference_render_path_uses_unified_runtime():
    config = build_inference_config_from_profile(generic_sweep_profile())
    payload = build_inference_report_payload(
        config=config,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
        cvs_version="1.0.0",
    )
    doc = render_report_html(payload)
    assert "Gate matrix" in doc
    assert "Full results" in doc


def test_series_builder_from_nested_graph_shape():
    graph = {
        "all_reduce": {
            "8": {"bus_bw": 12.5, "alg_bw": 11.0, "time": 100},
            "64": {"bus_bw": 45.0, "alg_bw": 40.0, "time": 200},
        }
    }
    profile = {
        "dataset_builder": "series",
        "series": {"x_field": "size", "y_fields": ["bus_bw", "alg_bw"]},
    }
    datasets = build_datasets("series", {"results": graph}, profile)
    assert "bus_bw" in datasets["charts"]
    assert datasets["results_table"]["rows"]
