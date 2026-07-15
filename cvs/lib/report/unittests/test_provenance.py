'''Unit tests for report provenance helpers.'''

from types import SimpleNamespace

from cvs.lib.report.inference import build_inference_report_payload
from cvs.lib.report.provenance import (
    build_inference_report_provenance,
    extend_run_card_display,
    provenance_run_card_rows,
)
from cvs.lib.report.unittests._fixtures import (
    generic_inference_report_config,
    generic_variant,
    two_cell_inf_res,
)


def test_build_inference_report_provenance_collects_paths():
    config = SimpleNamespace(
        option=SimpleNamespace(
            cluster_file="/cluster.json",
            config_file="/variant.json",
        )
    )
    prov = build_inference_report_provenance(
        config,
        cvs_version="9.9.9",
        pytest_html_path="/out/report.html",
        log_file_path="/out/run.log",
    )
    assert prov["cvs_version"] == "9.9.9"
    assert prov["cluster_file"] == "/cluster.json"
    assert prov["config_file"] == "/variant.json"
    assert prov["pytest_html_path"] == "/out/report.html"
    assert prov["log_file_path"] == "/out/run.log"


def test_provenance_run_card_rows_includes_standard_fields():
    rows = provenance_run_card_rows(
        {
            "cvs_version": "1.0.0",
            "git_ref": "abc1234 @ feature/x",
            "image_display": "rocm/atom-dev:latest @ sha256:deadbeef…",
            "launch_summary": "atom · TP=8 · max_model_len=10240",
            "cluster_file": "/cluster.json",
            "config_file": "/config.json",
        }
    )
    labels = [r[0] for r in rows]
    assert labels == [
        "CVS version",
        "Git ref",
        "Image",
        "Launch",
        "Cluster file",
        "Config file",
    ]


def test_extend_run_card_display_skips_duplicate_labels():
    base = [("Model", "m1", False)]
    extended = extend_run_card_display(
        base,
        {"cvs_version": "1.0.0", "cluster_file": "/cluster.json"},
    )
    labels = [r[0] for r in extended]
    assert "Model" in labels
    assert "CVS version" in labels
    assert labels.count("Model") == 1


def test_payload_includes_provenance_and_run_card_fields():
    payload = build_inference_report_payload(
        config=generic_inference_report_config(),
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
        cvs_version="1.0.0",
        provenance={
            "cluster_file": "/cluster.json",
            "config_file": "/config.json",
            "git_ref": "deadbeef @ main",
            "launch_summary": "atom · TP=8",
            "launch_server_cmd": "python -m atom.entrypoints.openai_server --model demo",
            "launch_bench_cmd": "python -m atom.benchmarks.benchmark_serving --model demo",
        },
    )
    assert payload["provenance"]["cluster_file"] == "/cluster.json"
    labels = [r[0] for r in payload["run_card_display"]]
    assert "Cluster file" in labels
    assert "Config file" in labels
    assert "Git ref" in labels
    assert "Launch" in labels
    assert payload["panels"]["launch"]["server_cmd"].startswith("python -m atom")
