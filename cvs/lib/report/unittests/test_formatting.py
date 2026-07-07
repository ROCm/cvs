'''Unit tests for report HTML formatting helpers.'''

from cvs.lib.report.formatting import link_or_text_html
from cvs.lib.report.inference import build_inference_report_payload, render_report_html
from cvs.lib.report.unittests._fixtures import (
    generic_inference_report_config,
    generic_variant,
    two_cell_inf_res,
)


def test_link_or_text_html_http_url():
    out = link_or_text_html("https://example.com/run", "Upstream")
    assert 'href="https://example.com/run"' in out
    assert 'target="_blank"' in out
    assert ">Upstream</a>" in out


def test_link_or_text_html_local_path_uses_basename():
    out = link_or_text_html("/home/user/cvs_results/run.html", "Pytest report")
    assert 'href="run.html"' in out
    assert ">Pytest report</a>" in out
    assert "target=" not in out


def test_link_or_text_html_empty():
    assert link_or_text_html("", "Pytest report") == "\u2014"


def test_render_report_html_run_card_links_sibling_artifacts():
    config = generic_inference_report_config()
    payload = build_inference_report_payload(
        config=config,
        variant_config=generic_variant(),
        inf_res_dict=two_cell_inf_res(),
        lifecycle_report={},
        pytest_html_path="/out/ix_atom_run.html",
        log_file_path="/out/ix_atom_run.log",
    )
    payload["run_card_display"] = [
        ("Pytest report", "/out/ix_atom_run.html", True),
        ("Run log", "/out/ix_atom_run.log", True),
    ]
    doc = render_report_html(payload)
    assert '<a href="ix_atom_run.html">Pytest report</a>' in doc
    assert '<a href="ix_atom_run.log">Run log</a>' in doc
