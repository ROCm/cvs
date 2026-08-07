'''Unit tests for prev-run comparison panel.'''

import json

from cvs.lib.report.panels.prev_run import (
    build_prev_run_panel,
    resolve_prev_run_json_path,
)


def test_prev_run_panel_flags_regression(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_id": "ISL=1024,OSL=1024,TP=8,CONC=128",
                        "host": "10.0.0.1",
                        "concurrency": 128,
                        "actuals": {"client.output_throughput": 4000.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    current_cells = [
        {
            "cell_id": "ISL=1024,OSL=1024,TP=8,CONC=128",
            "host": "10.0.0.1",
            "concurrency": 128,
            "actuals": {"client.output_throughput": 3600.0},
        }
    ]
    panel = build_prev_run_panel(current_cells, baseline, threshold_pct=5.0)
    assert panel is not None
    row = panel["rows"][0]
    assert row["regression"] is True
    assert row["compare.prev_run.throughput_delta_pct"] == -10.0


def test_resolve_prev_run_json_path_sibling(tmp_path):
    sibling = tmp_path / "inferencex_atom_report_prev.json"
    sibling.write_text("{}", encoding="utf-8")
    resolved = resolve_prev_run_json_path(
        "",
        report_basename="inferencex_atom_report",
        report_dir=tmp_path,
    )
    assert resolved == str(sibling)
