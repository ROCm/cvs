'''Panel unit tests.'''

from cvs.lib.report.panels.scaling import build_scaling_panel, render_scaling_panel_html
from cvs.lib.report.panels.training_parity import build_training_parity_panel


def test_scaling_panel_multi_host():
    cells = [
        {
            "host": "10.0.0.1",
            "policy": "w1",
            "concurrency": 128,
            "isl": 1024,
            "osl": 1024,
            "actuals": {"client.output_throughput": 4000.0},
        },
        {
            "host": "10.0.0.2",
            "policy": "w1",
            "concurrency": 128,
            "isl": 1024,
            "osl": 1024,
            "actuals": {"client.output_throughput": 3800.0},
        },
    ]
    panel = build_scaling_panel(cells=cells, nnodes=1)
    assert panel is not None
    assert panel["nnodes"] == 2
    assert panel["cluster_throughput"] == 7800.0
    assert "7,800" in render_scaling_panel_html(panel)


def test_scaling_panel_single_host_absent():
    cells = [
        {
            "host": "10.0.0.1",
            "policy": "w1",
            "concurrency": 128,
            "actuals": {"client.output_throughput": 4000.0},
        }
    ]
    assert build_scaling_panel(cells=cells, nnodes=1) is None


def test_training_parity_panel_ratios(tmp_path):
    ref = tmp_path / "ref.json"
    cand = tmp_path / "cand.json"
    ref.write_text(
        '{"nodes": {"node1": {"throughput_per_gpu": 100.0}, "node2": {"throughput_per_gpu": 90.0}}}',
        encoding="utf-8",
    )
    cand.write_text(
        '{"nodes": {"node1": {"throughput_per_gpu": 110.0}, "node2": {"throughput_per_gpu": 85.0}}}',
        encoding="utf-8",
    )
    panel = build_training_parity_panel(
        reference_json_path=str(ref),
        candidate_json_path=str(cand),
    )
    assert len(panel["rows"]) == 2
    assert panel["rows"][0]["compare.prev_run.throughput_per_gpu_ratio"] == 1.1
