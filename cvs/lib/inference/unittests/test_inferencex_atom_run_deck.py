'''Unit tests for IX Run Deck HTML export.'''

from types import SimpleNamespace

from cvs.lib.inference.inferencex_atom_run_deck import (
    build_run_deck_payload,
    render_run_deck_embed_html,
    render_run_deck_html,
    write_run_deck,
)


def _variant():
    return SimpleNamespace(
        model=SimpleNamespace(id="deepseek-ai/DeepSeek-R1-0528"),
        gpu_arch="mi300x",
        ix_recipe_id="dsr1-fp8-mi300x-atom",
        enforce_thresholds=True,
        run_card=SimpleNamespace(
            atom_image_pin="rocm/atom-dev:latest",
            upstream_run_url="https://github.com/ROCm/ATOM/actions/runs/1",
            notes="smoke",
        ),
        params=SimpleNamespace(driver="atom", tensor_parallelism="8"),
        thresholds={
            "ISL=1024,OSL=1024,TP=8,CONC=128": {
                "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
                "client.mean_ttft_ms": {"kind": "max_ms", "value": 2000.0},
            },
            "ISL=1024,OSL=1024,TP=8,CONC=256": {
                "client.output_throughput": {"kind": "min_tok_s", "value": 1000.0},
                "client.mean_ttft_ms": {"kind": "max_ms", "value": 2500.0},
            },
        },
        cell_key=lambda isl, osl, conc: f"ISL={isl},OSL={osl},TP=8,CONC={conc}",
    )


def _two_cell_inf_res():
    inf_res = {}
    for conc, tput, ttft, tpot in (
        (128, 4000.0, 100.0, 28.0),
        (256, 6000.0, 150.0, 30.0),
    ):
        key = ("deepseek-ai/DeepSeek-R1-0528", "mi300x", "1024", "1024", "w1", conc)
        inf_res[key] = {
            "10.0.0.1": {
                "client.output_throughput": tput,
                "client.total_token_throughput": tput * 1.1,
                "client.mean_ttft_ms": ttft,
                "client.mean_tpot_ms": tpot,
                "client.p99_itl_ms": 5.0,
            }
        }
    return inf_res


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
        "pkg::test_inferencex_atom_inference[w1_smoke-128]": [
            ("server_ready", 90.0, "s"),
            ("client_complete", 300.0, "s"),
        ],
    }
    payload = build_run_deck_payload(
        variant_config=_variant(),
        inf_res_dict=inf_res,
        lifecycle_report=lifecycle,
        cvs_version="1.0.0",
        pytest_html_path="/tmp/report.html",
    )
    assert payload["run_card"]["model"] == "deepseek-ai/DeepSeek-R1-0528"
    assert payload["overall_status"] == "pass"
    assert len(payload["cells"]) == 1
    assert payload["cells"][0]["tiers"]["throughput"] in ("pass", "fail", "record", "na")
    assert payload["cells"][0]["cell_lifecycle"]["server_ready"] == 90.0
    assert payload["lifecycle"]["container_launch"] == 12.0
    assert payload["chart_points"] == [(128, 4274.0)]
    assert len(payload["results_table"]["rows"]) == 1


def test_build_run_deck_payload_sweep_summary_and_matrix():
    payload = build_run_deck_payload(
        variant_config=_variant(),
        inf_res_dict=_two_cell_inf_res(),
        lifecycle_report={},
        cvs_version="1.0.0",
    )
    assert len(payload["sweep_summaries"]) == 1
    summary = payload["sweep_summaries"][0]
    assert summary["conc_at_max_tput"] == 256
    assert summary["max_output_throughput"] == 6000.0
    assert len(payload["gate_matrix"]) == 2
    assert "mean_ttft_ms" in payload["chart_series"]
    assert len(payload["chart_series"]["output_throughput"]) == 2


def test_render_run_deck_html_contains_run_deck_title():
    payload = build_run_deck_payload(
        variant_config=_variant(),
        inf_res_dict={},
        lifecycle_report={},
    )
    doc = render_run_deck_html(payload)
    assert "IX Run Deck" in doc
    assert "deepseek-ai/DeepSeek-R1-0528" in doc
    assert "<!DOCTYPE html>" in doc
    assert "status-badge" in doc
    assert "Gate matrix" in doc
    assert "ATOM upstream run" in doc


def test_render_includes_concurrency_charts_for_two_cells():
    payload = build_run_deck_payload(
        variant_config=_variant(),
        inf_res_dict=_two_cell_inf_res(),
        lifecycle_report={},
    )
    doc = render_run_deck_html(payload)
    assert "Sweep analytics" in doc
    assert "Output tok/s" in doc
    assert "Mean TTFT" in doc
    assert "C=128" in doc
    assert "C=256" in doc
    assert "Full results" in doc


def test_render_run_deck_embed_html_iframe():
    payload = build_run_deck_payload(
        variant_config=_variant(),
        inf_res_dict=_two_cell_inf_res(),
        lifecycle_report={},
    )
    embed = render_run_deck_embed_html(payload)
    assert "<iframe" in embed
    assert "IX Run Deck" in embed


def test_write_run_deck_writes_html_and_json(tmp_path):
    artifacts = write_run_deck(
        tmp_path / "inferencex_atom_run_deck.html",
        variant_config=_variant(),
        inf_res_dict=_two_cell_inf_res(),
        lifecycle_report={},
        cvs_version="1.0.0",
    )
    assert artifacts["html"].is_file()
    assert artifacts["json"].is_file()
    assert "IX Run Deck" in artifacts["html"].read_text(encoding="utf-8")
    assert '"overall_status"' in artifacts["json"].read_text(encoding="utf-8")
