'''Unit tests for experimental IX Run Deck HTML export.'''

from types import SimpleNamespace

from cvs.lib.inference.inferencex_atom_run_deck import (
    build_run_deck_payload,
    render_run_deck_html,
)


def _variant():
    return SimpleNamespace(
        model=SimpleNamespace(id="deepseek-ai/DeepSeek-R1-0528"),
        gpu_arch="mi300x",
        ix_recipe_id="dsr1-fp8-mi300x-atom",
        enforce_thresholds=True,
        run_card=SimpleNamespace(
            atom_image_pin="rocm/atom-dev:latest",
            upstream_run_url="",
            notes="smoke",
        ),
        params=SimpleNamespace(driver="atom", tensor_parallelism="8"),
        thresholds={
            "ISL=1024,OSL=1024,TP=8,CONC=128": {
                "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
                "client.mean_ttft_ms": {"kind": "max_ms", "value": 2000.0},
            }
        },
        cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,CONC={conc}",
    )


def test_build_run_deck_payload_cells_and_chart():
    key = ("deepseek-ai/DeepSeek-R1-0528", "mi300x", "1024", "1024", "w1_smoke", 128)
    inf_res = {
        key: {
            "10.0.0.1": {
                "client.output_throughput": 4274.0,
                "client.mean_ttft_ms": 400.0,
                "client.mean_tpot_ms": 28.8,
            }
        }
    }
    lifecycle = {
        "n::test_launch_container": [("container_launch", 12.0, "s")],
        "n::test_inference": [("server_ready", 90.0, "s"), ("client_complete", 300.0, "s")],
    }
    payload = build_run_deck_payload(
        variant_config=_variant(),
        inf_res_dict=inf_res,
        lifecycle_report=lifecycle,
        cvs_version="1.0.0",
    )
    assert payload["run_card"]["model"] == "deepseek-ai/DeepSeek-R1-0528"
    assert len(payload["cells"]) == 1
    assert payload["cells"][0]["tiers"]["throughput"] in ("pass", "fail", "record", "na")
    assert payload["lifecycle"]["container_launch"] == 12.0
    assert payload["chart_points"] == []  # single cell — no concurrency chart


def test_render_run_deck_html_contains_run_deck_title():
    payload = build_run_deck_payload(
        variant_config=_variant(),
        inf_res_dict={},
        lifecycle_report={},
    )
    html = render_run_deck_html(payload)
    assert "IX Run Deck" in html
    assert "deepseek-ai/DeepSeek-R1-0528" in html
    assert "<!DOCTYPE html>" in html


def test_render_includes_concurrency_chart_for_two_cells():
    variant = _variant()
    inf_res = {}
    for conc, tput in ((128, 4000.0), (256, 6000.0)):
        key = ("m", "mi300x", "1024", "1024", "w1", conc)
        inf_res[key] = {"h": {"client.output_throughput": tput, "client.mean_ttft_ms": 100.0}}
    payload = build_run_deck_payload(
        variant_config=variant,
        inf_res_dict=inf_res,
        lifecycle_report={},
    )
    html = render_run_deck_html(payload)
    assert "Throughput vs concurrency" in html
    assert "C=128" in html
    assert "C=256" in html
