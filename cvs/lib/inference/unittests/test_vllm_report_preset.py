'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Tests for the vLLM inference report preset and its parsing-helper bindings.
Render-only: these pin the report wiring, not pass/fail enforcement.
'''

from types import SimpleNamespace

from cvs.lib.inference.utils.vllm_parsing import (
    CLIENT_METRICS,
    GATED_METRICS,
    METRIC_TIER_ORDER,
    METRIC_TIERS,
    RECORD_METRICS,
    VLLM_RESULTS_COLUMNS,
    tier_metric_specs,
)
from cvs.lib.report.auto_register import try_auto_register_inference_suite_report
from cvs.lib.report.inference_payload import build_inference_report_payload
from cvs.lib.report.presets.vllm import VLLM_REPORT_CONFIG
from cvs.lib.report.registry import get_suite_report_config
from cvs.lib.report.types import InferenceReportConfig


# --- auto-registration -----------------------------------------------------


def test_auto_register_loads_vllm_preset_for_stem():
    config = SimpleNamespace(_suite_name="vllm")
    assert try_auto_register_inference_suite_report(config) is True
    registered = get_suite_report_config(config)
    assert isinstance(registered, InferenceReportConfig)
    assert registered.suite_id == "vllm"
    assert registered is VLLM_REPORT_CONFIG


def test_no_register_for_unknown_stem():
    config = SimpleNamespace(_suite_name="definitely_not_a_suite")
    assert try_auto_register_inference_suite_report(config) is False
    assert get_suite_report_config(config) is None


# --- results-table column contract -----------------------------------------


def test_results_columns_first_seven_are_fixed_positional():
    fixed = VLLM_RESULTS_COLUMNS[:7]
    assert [label for label, _key in fixed] == [
        "Model",
        "GPU",
        "ISL",
        "OSL",
        "Policy",
        "Conc",
        "Host",
    ]
    # build_results_table emits these 7 from the cell key/host directly and only
    # looks up metric_keys[7:] in each host's metric dict -> their keys must be None.
    assert all(key is None for _label, key in fixed)


def test_metric_columns_are_client_namespaced():
    for _label, key in VLLM_RESULTS_COLUMNS[7:]:
        assert key is not None
        assert key.startswith("client.")


# --- tier partition fidelity ----------------------------------------------


def test_tier_order_includes_every_tier_plus_record():
    assert set(METRIC_TIERS).issubset(set(METRIC_TIER_ORDER))
    assert METRIC_TIER_ORDER[-1] == "record"
    assert "record" not in METRIC_TIERS


def test_gated_metrics_partition_exactly():
    tiered = [m for names in METRIC_TIERS.values() for m in names]
    # No metric appears in two tiers.
    assert len(tiered) == len(set(tiered))
    # The tiered set is exactly GATED_METRICS -- nothing gated is unranked, and
    # nothing record-only is gated.
    assert set(tiered) == set(GATED_METRICS)


def test_record_metrics_are_disjoint_from_gated():
    assert set(RECORD_METRICS).isdisjoint(set(GATED_METRICS))
    # Every record metric is a real emitted metric.
    emitted = {short for short, _unit in CLIENT_METRICS}
    assert set(RECORD_METRICS).issubset(emitted)


def test_tier_metric_specs_returns_specs_present_in_cell():
    cell = {
        "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
        "client.mean_ttft_ms": {"kind": "max_ms", "value": 200.0},
    }
    tput = tier_metric_specs(cell, "throughput")
    assert tput == {"client.output_throughput": {"kind": "min_tok_s", "value": 1000.0}}
    ttft = tier_metric_specs(cell, "ttft")
    assert ttft == {"client.mean_ttft_ms": {"kind": "max_ms", "value": 200.0}}
    # A tier with no specs present in the cell returns empty.
    assert tier_metric_specs(cell, "health") == {}


# --- preset lifecycle labels match what the suite records ------------------


def test_session_labels_match_suite_recorded_labels():
    # The unified vLLM suite (vllm.py) records exactly these session labels.
    recorded = {
        "container_launch",
        "topology_discovery",
        "model_fetch",
        "server_ready",
        "teardown",
    }
    assert set(VLLM_REPORT_CONFIG.session_lifecycle_labels).issubset(recorded)
    assert "sshd_setup" not in VLLM_REPORT_CONFIG.session_lifecycle_labels
    assert VLLM_REPORT_CONFIG.cell_lifecycle_labels == ("server_ready",)


# --- end-to-end payload over a synthetic sweep -----------------------------


def _vllm_variant():
    return SimpleNamespace(
        model=SimpleNamespace(id="meta-llama/Llama-3.1-70B"),
        gpu_arch="mi300x",
        enforce_thresholds=True,
        thresholds={
            "ISL=1024,OSL=1024,TP=8,PP=2,CONC=64": {
                "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
            },
            "ISL=1024,OSL=1024,TP=8,PP=2,CONC=128": {
                "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
            },
        },
        cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,PP=2,CONC={conc}",
        params=SimpleNamespace(nnodes=2, pipeline_parallel_size=2, tensor_parallelism="8"),
    )


def _two_cell_multihost_inf_res():
    inf_res = {}
    for conc, tput in ((64, 4000.0), (128, 6000.0)):
        key = ("meta-llama/Llama-3.1-70B", "mi300x", "1024", "1024", "default", conc)
        # Two hosts -> exercises multi-host cell rendering.
        inf_res[key] = {
            "10.245.135.11": {
                "client.output_throughput": tput,
                "client.mean_ttft_ms": 100.0,
                "client.mean_tpot_ms": 28.0,
            },
            "10.245.135.13": {
                "client.output_throughput": tput * 0.98,
                "client.mean_ttft_ms": 105.0,
                "client.mean_tpot_ms": 29.0,
            },
        }
    return inf_res


def test_build_payload_over_synthetic_sweep():
    payload = build_inference_report_payload(
        config=VLLM_REPORT_CONFIG,
        variant_config=_vllm_variant(),
        inf_res_dict=_two_cell_multihost_inf_res(),
        lifecycle_report={},
        cvs_version="test",
    )
    assert payload["schema_version"] == 1
    assert payload["suite_id"] == "vllm"
    # 2 cells x 2 hosts = 4 cell records.
    assert len(payload["cells"]) == 4
    # Results table header has the full column width.
    assert len(payload["results_table"]["headers"]) == len(VLLM_RESULTS_COLUMNS)
    for row in payload["results_table"]["rows"]:
        assert len(row) == len(VLLM_RESULTS_COLUMNS)
    # enforce_thresholds=True and an output_throughput gate is met in every cell
    # -> overall pass.
    assert payload["overall_status"] == "pass"
    # The PP= segment shows up in cell ids (distributed key shape).
    assert all("PP=2" in c["cell_id"] for c in payload["cells"])
