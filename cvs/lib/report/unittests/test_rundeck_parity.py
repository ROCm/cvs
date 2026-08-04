'''Parity and integration tests for the unified Run Deck engine.'''

from cvs.lib.report.inference import build_inference_report_payload, render_report_html
from cvs.lib.report.presets.inferencex_atom import INFERENCEX_ATOM_REPORT_CONFIG
from cvs.lib.report.profile import load_json_profile
from cvs.lib.report.rundeck.config_adapter import build_inference_config_from_profile
from cvs.lib.report.rundeck.dataset_builders.registry import build_datasets
from cvs.lib.report.rundeck.payload import build_rundeck_payload
from cvs.lib.report.rundeck.render import render_rundeck_html
from cvs.lib.report.unittests._fixtures import generic_variant, two_cell_inf_res


def test_json_profile_matches_legacy_inference_config():
    profile = load_json_profile("inferencex_atom_single")
    assert profile is not None
    from_profile = build_inference_config_from_profile(profile)
    legacy = INFERENCEX_ATOM_REPORT_CONFIG
    assert from_profile.suite_id == legacy.suite_id
    assert from_profile.report_basename == legacy.report_basename
    assert from_profile.results_columns == legacy.results_columns
    assert from_profile.metric_tier_order == legacy.metric_tier_order
    assert len(from_profile.chart_series) == len(legacy.chart_series)


def test_rundeck_payload_parity_with_legacy_builder():
    profile = load_json_profile("inferencex_atom_single")
    store = {
        "inf_res_dict": two_cell_inf_res(),
        "variant_config": generic_variant(),
        "lifecycle_report": {},
    }
    rundeck_payload = build_rundeck_payload(profile=profile, store=store, cvs_version="1.0.0")
    legacy_payload = build_inference_report_payload(
        config=INFERENCEX_ATOM_REPORT_CONFIG,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
        cvs_version="1.0.0",
    )
    assert len(rundeck_payload["cells"]) == len(legacy_payload["cells"])
    assert rundeck_payload["results_table"]["headers"] == legacy_payload["results_table"]["headers"]
    assert len(rundeck_payload["gate_matrix"]) == len(legacy_payload["gate_matrix"])
    assert rundeck_payload["chart_series"].keys() == legacy_payload["chart_series"].keys()


def test_rundeck_render_contains_core_panels():
    profile = load_json_profile("inferencex_atom_single")
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
    assert "IX Run Deck" in doc
    assert "report-nav" in doc
    assert "Gate matrix" in doc
    assert "Full results" in doc
    assert "Sweep analytics" in doc


def test_legacy_render_path_uses_unified_runtime():
    payload = build_inference_report_payload(
        config=INFERENCEX_ATOM_REPORT_CONFIG,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
        cvs_version="1.0.0",
    )
    doc = render_report_html(payload)
    assert "Gate matrix" in doc
    assert "Full results" in doc


def test_series_builder_from_rccl_shape():
    graph = {
        "all_reduce": {
            "8": {"bus_bw": 12.5, "alg_bw": 11.0, "time": 100},
            "64": {"bus_bw": 45.0, "alg_bw": 40.0, "time": 200},
        }
    }
    profile = load_json_profile("rccl_perf")
    datasets = build_datasets("series", {"results": graph}, profile)
    assert "bus_bw" in datasets["charts"]
    assert datasets["results_table"]["rows"]


def test_demo_generator_writes_artifacts(tmp_path):
    from cvs.lib.report.demo.generate_inferencex_atom_rundeck import generate

    paths = generate(tmp_path)
    assert paths["html"].is_file()
    assert paths["json"].is_file()
    html = paths["html"].read_text(encoding="utf-8")
    assert "IX Run Deck" in html
    assert "Full results" in html


def test_vllm_json_profile_loads():
    profile = load_json_profile("vllm")
    assert profile is not None
    cfg = build_inference_config_from_profile(profile)
    assert cfg.suite_id == "vllm"
    assert cfg.session_lifecycle_labels[1] == "topology_discovery"
