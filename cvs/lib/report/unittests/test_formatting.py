'''Unit tests for report HTML formatting helpers.'''

from cvs.lib.report.formatting import link_or_text_html


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


def test_link_or_text_html_preserves_bundle_relative_path():
    out = link_or_text_html("../inferencex_atom.html", "Pytest report")
    assert 'href="../inferencex_atom.html"' in out


def test_link_or_text_html_empty():
    assert link_or_text_html("", "Pytest report") == "\u2014"
