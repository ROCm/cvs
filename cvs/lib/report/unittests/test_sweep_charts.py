'''Unit tests for cross-shape sweep chart renderers.'''

from cvs.lib.report.render.sweep_charts import (
    render_grouped_comparison_chart,
    render_line_comparison_chart,
)

_SAMPLE = {
    "title": "Output tok/s",
    "unit": "tok/s",
    "concurrencies": [16, 32],
    "series": [
        {"label": "ISL=1024 · OSL=1024", "values": [741.0, 1305.0]},
        {"label": "ISL=8192 · OSL=1024", "values": [546.0, 787.0]},
    ],
}


def test_render_grouped_comparison_chart():
    html = render_grouped_comparison_chart(_SAMPLE)
    assert "Output tok/s" in html
    assert "chart-clusters" in html
    assert "chart-has-tip" in html
    assert "C=16" in html
    assert "ISL=1024" in html
    assert "chart-y" not in html


def test_render_grouped_comparison_chart_shared_legend():
    html = render_grouped_comparison_chart(_SAMPLE, show_legend=False)
    assert "chart-legend" not in html


def test_render_line_comparison_chart():
    html = render_line_comparison_chart(_SAMPLE)
    assert "chart-line-svg" in html
    assert "<polyline" in html
    assert "chart-line-point" in html
    assert "C=32" in html
