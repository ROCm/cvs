'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

IX Run Deck — a single-page HTML summary for inferencex_atom_single runs.
Render-only; does not affect pytest pass/fail or threshold enforcement.
'''

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from cvs.lib.inference.utils.inferencex_atom_parsing import (
    CLIENT_METRIC_UNITS,
    METRIC_TIER_ORDER,
    tier_metric_specs,
)
from cvs.lib.utils.verdict import _check_one

# Keep in sync with INFERENCEX_ATOM_RESULTS_COLUMNS in inference_suite_results_table.py
_RESULTS_TABLE_COLUMNS = (
    ("Model", None),
    ("GPU", None),
    ("ISL", None),
    ("OSL", None),
    ("Policy", None),
    ("Conc", None),
    ("Host", None),
    ("Output tok/s", "client.output_throughput"),
    ("Total tok/s", "client.total_token_throughput"),
    ("Mean TTFT (ms)", "client.mean_ttft_ms"),
    ("Mean TPOT (ms)", "client.mean_tpot_ms"),
    ("P99 ITL (ms)", "client.p99_itl_ms"),
)

# Lifecycle stages shown left-to-right on the session timeline (max seconds per label).
_TIMELINE_LABELS = (
    "container_launch",
    "sshd_setup",
    "model_fetch",
    "server_ready",
    "client_complete",
    "teardown",
)

_CELL_LIFECYCLE_LABELS = ("server_ready", "client_complete")

_CELL_HIGHLIGHTS = (
    ("output_throughput", "Output tok/s"),
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("mean_tpot_ms", "Mean TPOT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
    ("p95_tpot_ms", "P95 TPOT (ms)"),
)

_CHART_SERIES = (
    ("output_throughput", "Output tok/s", "tok/s", False),
    ("mean_ttft_ms", "Mean TTFT", "ms", True),
    ("mean_tpot_ms", "Mean TPOT", "ms", True),
)

_GATED_TIERS = tuple(t for t in METRIC_TIER_ORDER if t != "record")


def _fmt_num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if abs(f) >= 1000:
        return f"{f:,.{digits}f}"
    return f"{f:.{digits}f}"


def _metric_pass(metric: str, actual: Any, spec: Optional[dict]) -> str:
    """Return pass | fail | na for deck display (does not raise)."""
    if spec is None or actual is None:
        return "na"
    err = _check_one(metric, actual, spec)
    return "fail" if err else "pass"


def _tier_status(actuals: Mapping[str, Any], thresholds_cell: dict, tier: str, enforce: bool) -> str:
    if not enforce or tier == "record":
        return "record"
    specs = tier_metric_specs(thresholds_cell, tier)
    if not specs:
        return "na"
    for metric, spec in specs.items():
        if _metric_pass(metric, actuals.get(metric), spec) == "fail":
            return "fail"
    return "pass"


def _aggregate_lifecycle(lifecycle_report: Mapping[str, list]) -> Dict[str, float]:
    """Max value per stage label across all pytest nodeids."""
    out: Dict[str, float] = {}
    for rows in lifecycle_report.values():
        for label, value, unit in rows:
            if unit != "s":
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            out[label] = max(out.get(label, 0.0), v)
    return out


def _lifecycle_for_cell(lifecycle_report: Mapping[str, list], concurrency: Any) -> Dict[str, float]:
    """Extract server_ready / client_complete timings for one sweep cell."""
    conc = str(concurrency)
    suffix = f"-{conc}]"
    out: Dict[str, float] = {}
    for nodeid, rows in lifecycle_report.items():
        if "test_inferencex_atom_inference" not in nodeid or suffix not in nodeid:
            continue
        for label, value, unit in rows:
            if unit != "s" or label not in _CELL_LIFECYCLE_LABELS:
                continue
            try:
                out[label] = float(value)
            except (TypeError, ValueError):
                continue
    return out


def _bar_pct(actual: float, spec: dict) -> float:
    """Fill width 0–100 for threshold comparison bars."""
    kind = spec.get("kind", "")
    target = float(spec["value"])
    if target <= 0:
        return 0.0
    if kind in ("min", "min_tok_s", "min_ratio"):
        return min(100.0, 100.0 * float(actual) / target)
    if kind in ("max", "max_ms"):
        if float(actual) <= 0:
            return 100.0
        return min(100.0, 100.0 * target / float(actual))
    if kind == "within":
        return 100.0
    return 50.0


def _margin_text(actual: Any, spec: Optional[dict]) -> Optional[str]:
    if spec is None or actual is None:
        return None
    try:
        act = float(actual)
        target = float(spec["value"])
    except (TypeError, ValueError):
        return None
    kind = spec.get("kind", "")
    if kind in ("min", "min_tok_s", "min_ratio"):
        diff = act - target
        if target:
            return f"+{_fmt_num(diff)} ({100.0 * diff / target:.0f}% above gate)"
        return f"+{_fmt_num(diff)} above gate"
    if kind in ("max", "max_ms"):
        diff = target - act
        return f"{_fmt_num(diff)} under gate"
    return None


def _overall_status(cells: List[dict], enforce: bool) -> str:
    if not cells:
        return "na"
    if not enforce:
        return "record"
    for cell in cells:
        for tier in _GATED_TIERS:
            if cell["tiers"].get(tier) == "fail":
                return "fail"
    return "pass"


def _build_chart_series(cells: List[dict]) -> Dict[str, List[Tuple[int, float]]]:
    series: Dict[str, List[Tuple[int, float]]] = {}
    for short, _label, _unit, _invert in _CHART_SERIES:
        points: List[Tuple[int, float]] = []
        for cell in cells:
            val = cell["actuals"].get(f"client.{short}")
            if val is None:
                continue
            try:
                points.append((int(cell["concurrency"]), float(val)))
            except (TypeError, ValueError):
                continue
        if points:
            series[short] = sorted(points, key=lambda p: p[0])
    return series


def _build_sweep_summaries(cells: List[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for cell in cells:
        groups.setdefault((str(cell["isl"]), str(cell["osl"])), []).append(cell)

    summaries: List[dict] = []
    for (isl, osl), group in sorted(groups.items()):
        points = []
        for cell in group:
            tput = cell["actuals"].get("client.output_throughput")
            if tput is None:
                continue
            try:
                points.append((int(cell["concurrency"]), float(tput), cell))
            except (TypeError, ValueError):
                continue
        if not points:
            continue

        best_conc, best_tput, best_cell = max(points, key=lambda p: p[1])
        ttft_at_max = best_cell["actuals"].get("client.mean_ttft_ms")
        sorted_points = sorted(points, key=lambda p: p[0])
        saturated = False
        if len(sorted_points) >= 2:
            last_conc, last_tput, _ = sorted_points[-1]
            prev_tput = sorted_points[-2][1]
            saturated = last_conc == best_conc and last_tput <= prev_tput * 1.01

        summaries.append(
            {
                "isl": isl,
                "osl": osl,
                "max_output_throughput": best_tput,
                "conc_at_max_tput": best_conc,
                "ttft_at_max_tput": ttft_at_max,
                "saturated": saturated,
                "cell_count": len(group),
            }
        )
    return summaries


def _build_gate_matrix(cells: List[dict]) -> List[dict]:
    rows = []
    for cell in cells:
        rows.append(
            {
                "label": f"{cell['policy']} · C={cell['concurrency']}",
                "cell_id": cell["cell_id"],
                "concurrency": cell["concurrency"],
                "tiers": cell["tiers"],
            }
        )
    return rows


def _build_results_table(inf_res_dict: Mapping[tuple, Any]) -> dict:
    headers = [label for label, _key in _RESULTS_TABLE_COLUMNS]
    metric_keys = [key for _label, key in _RESULTS_TABLE_COLUMNS]
    rows: List[List[Any]] = []
    for key, host_dict in sorted(inf_res_dict.items(), key=lambda kv: (kv[0][4], kv[0][5])):
        model, gpu, isl, osl, policy, conc = key
        if not isinstance(host_dict, dict):
            continue
        fixed = [model, gpu, isl, osl, policy, conc]
        for host, metrics in host_dict.items():
            row = list(fixed)
            row.append(host)
            for mk in metric_keys[7:]:
                if mk is None:
                    row.append("—")
                else:
                    v = metrics.get(mk)
                    row.append(v if v is not None else "—")
            rows.append(row)
    return {"headers": headers, "rows": rows}


def build_run_deck_payload(
    *,
    variant_config,
    inf_res_dict: Mapping[tuple, Any],
    lifecycle_report: Mapping[str, list],
    cvs_version: str = "unknown",
    pytest_html_path: str = "",
    log_file_path: str = "",
) -> dict:
    """Structured payload for HTML render (also useful for tests and JSON export)."""
    cells: List[dict] = []
    enforce = bool(variant_config.enforce_thresholds)

    for key, host_dict in sorted(inf_res_dict.items(), key=lambda kv: (kv[0][4], kv[0][5])):
        model, gpu, isl, osl, policy, conc = key
        if not isinstance(host_dict, dict) or not host_dict:
            continue
        _host, actuals = next(iter(host_dict.items()))
        cell_id = variant_config.cell_key(isl, osl, conc)
        thresholds_cell = variant_config.thresholds.get(cell_id) or {}

        metrics = []
        for short, label in _CELL_HIGHLIGHTS:
            full = f"client.{short}"
            spec = thresholds_cell.get(full)
            actual = actuals.get(full)
            metrics.append(
                {
                    "label": label,
                    "metric": full,
                    "actual": actual,
                    "unit": CLIENT_METRIC_UNITS.get(short, ""),
                    "spec": spec,
                    "status": _metric_pass(full, actual, spec) if enforce and spec else "record",
                    "bar_pct": _bar_pct(float(actual), spec)
                    if spec is not None and actual is not None and enforce
                    else None,
                    "margin": _margin_text(actual, spec) if enforce and spec else None,
                }
            )

        tiers = {tier: _tier_status(actuals, thresholds_cell, tier, enforce) for tier in METRIC_TIER_ORDER}

        cells.append(
            {
                "model": model,
                "gpu": gpu,
                "isl": isl,
                "osl": osl,
                "policy": policy,
                "concurrency": conc,
                "host": _host,
                "cell_id": cell_id,
                "metrics": metrics,
                "tiers": tiers,
                "actuals": dict(actuals),
                "cell_lifecycle": _lifecycle_for_cell(lifecycle_report, conc),
            }
        )

    rc = variant_config.run_card
    chart_series = _build_chart_series(cells)
    legacy_chart = chart_series.get("output_throughput", [])

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "cvs_version": cvs_version,
        "overall_status": _overall_status(cells, enforce),
        "run_card": {
            "model": variant_config.model.id,
            "gpu_arch": variant_config.gpu_arch,
            "driver": variant_config.params.driver,
            "ix_recipe_id": variant_config.ix_recipe_id or "",
            "image_pin": rc.atom_image_pin or "",
            "upstream_run_url": rc.upstream_run_url or "",
            "notes": rc.notes or "",
            "enforce_thresholds": enforce,
            "tp": variant_config.params.tensor_parallelism,
        },
        "provenance": {
            "pytest_html_path": pytest_html_path,
            "log_file_path": log_file_path,
        },
        "lifecycle": _aggregate_lifecycle(lifecycle_report),
        "cells": cells,
        "chart_points": legacy_chart,
        "chart_series": chart_series,
        "sweep_summaries": _build_sweep_summaries(cells),
        "gate_matrix": _build_gate_matrix(cells),
        "results_table": _build_results_table(inf_res_dict),
    }


def _link_or_text(url: str, label: str) -> str:
    if not url:
        return "—"
    safe_url = html.escape(url)
    safe_label = html.escape(label)
    if re.match(r"^https?://", url):
        return f'<a href="{safe_url}" target="_blank" rel="noopener">{safe_label}</a>'
    return f"<span>{safe_label}</span>"


def _status_badge(status: str) -> str:
    labels = {
        "pass": "PASS",
        "fail": "FAIL",
        "record": "RECORD",
        "na": "N/A",
    }
    return (
        f'<span class="status-badge status-{html.escape(status)}">'
        f"{html.escape(labels.get(status, status.upper()))}</span>"
    )


def _render_bar_chart(
    title: str,
    points: List[Tuple[int, float]],
    unit: str,
    *,
    invert: bool = False,
    accent: str = "accent",
) -> str:
    if len(points) < 2:
        return ""
    values = [p[1] for p in points]
    max_val = max(values) or 1.0
    min_val = min(values) or 0.0
    bars = []
    for conc, val in points:
        if invert:
            span = max_val - min_val or max_val or 1.0
            norm = (max_val - val) / span if span else 1.0
            h = max(12.0, 20.0 + 80.0 * norm)
        else:
            h = max(12.0, 100.0 * val / max_val)
        bars.append(
            f"<div class='chart-col'><div class='chart-bar chart-bar-{accent}' "
            f"style='height:{h:.0f}%'></div>"
            f"<div class='chart-x'>C={conc}</div>"
            f"<div class='chart-y'>{_fmt_num(val)}</div></div>"
        )
    unit_html = html.escape(unit)
    return (
        f"<div class='chart-panel'><h3>{html.escape(title)}</h3>"
        f"<div class='chart'>{''.join(bars)}</div>"
        f"<div class='chart-unit'>{unit_html}</div></div>"
    )


def _deck_css() -> str:
    return """
:root {
  --bg: #0f1117;
  --panel: #1a1d27;
  --border: #2a2f3d;
  --text: #e8eaef;
  --muted: #9aa3b5;
  --accent: #ff6b35;
  --accent2: #6b9fff;
  --accent3: #c77dff;
  --pass: #3dd68c;
  --fail: #ff5c6a;
  --record: #6b9fff;
  --na: #5c6370;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  background: linear-gradient(160deg, #0a0c12 0%, #12151f 40%, #0f1117 100%);
  color: var(--text);
  line-height: 1.45;
  padding: 1.5rem;
}
.wrap { max-width: 1140px; margin: 0 auto; }
.hero-head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.5rem;
}
h1 {
  font-size: 1.75rem;
  font-weight: 600;
  margin: 0 0 0.25rem;
  letter-spacing: -0.02em;
}
.subtitle { color: var(--muted); margin: 0; font-size: 0.95rem; }
.status-badge {
  display: inline-block;
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  padding: 0.45rem 0.9rem;
  border-radius: 999px;
  border: 1px solid var(--border);
}
.status-pass { background: rgba(61,214,140,0.15); color: var(--pass); }
.status-fail { background: rgba(255,92,106,0.15); color: var(--fail); }
.status-record { background: rgba(107,159,255,0.12); color: var(--record); }
.status-na { background: rgba(92,99,112,0.2); color: var(--na); }
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.panel h2 {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin: 0 0 1rem;
  font-weight: 600;
}
.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 0.75rem 1rem;
}
.meta-item { display: flex; flex-direction: column; gap: 0.15rem; }
.meta-k { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.meta-v { font-size: 0.9rem; font-weight: 500; word-break: break-word; }
.meta-v a { color: var(--accent2); text-decoration: none; }
.meta-v a:hover { text-decoration: underline; }
.tl-row { display: flex; gap: 3px; min-height: 52px; border-radius: 8px; overflow: hidden; }
.tl-seg {
  background: linear-gradient(180deg, #2d3548 0%, #232836 100%);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 0.35rem;
  min-width: 48px;
  border-right: 1px solid var(--border);
}
.tl-lbl { font-size: 0.65rem; color: var(--muted); text-align: center; }
.tl-val { font-size: 0.8rem; font-weight: 600; color: var(--accent); }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
}
.summary-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem;
}
.summary-card h3 { margin: 0 0 0.5rem; font-size: 0.95rem; }
.summary-stat { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
.summary-meta { font-size: 0.8rem; color: var(--muted); margin-top: 0.35rem; }
.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1rem;
}
.chart-panel {
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem;
}
.chart-panel h3 {
  margin: 0 0 0.75rem;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
}
.chart {
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 1.25rem;
  height: 160px;
  padding-top: 0.5rem;
}
.chart-col { display: flex; flex-direction: column; align-items: center; width: 64px; height: 100%; }
.chart-bar {
  width: 40px;
  border-radius: 6px 6px 2px 2px;
  margin-top: auto;
}
.chart-bar-accent { background: linear-gradient(180deg, var(--accent) 0%, #c44d28 100%); }
.chart-bar-accent2 { background: linear-gradient(180deg, var(--accent2) 0%, #3d5a99 100%); }
.chart-bar-accent3 { background: linear-gradient(180deg, var(--accent3) 0%, #7a3db8 100%); }
.chart-x { font-size: 0.75rem; margin-top: 0.5rem; color: var(--muted); }
.chart-y { font-size: 0.7rem; font-weight: 600; margin-top: 0.2rem; }
.chart-unit { text-align: center; font-size: 0.7rem; color: var(--muted); margin-top: 0.35rem; }
.matrix-wrap { overflow-x: auto; }
.matrix {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}
.matrix th, .matrix td {
  border: 1px solid var(--border);
  padding: 0.5rem 0.65rem;
  text-align: center;
}
.matrix th {
  background: rgba(255,255,255,0.04);
  color: var(--muted);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.matrix td:first-child { text-align: left; font-weight: 500; }
.matrix-pass { background: rgba(61,214,140,0.12); color: var(--pass); }
.matrix-fail { background: rgba(255,92,106,0.12); color: var(--fail); }
.matrix-record { background: rgba(107,159,255,0.08); color: var(--record); }
.matrix-na { color: var(--na); }
.results-wrap { overflow-x: auto; }
.results-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.8rem;
}
.results-table th, .results-table td {
  border: 1px solid var(--border);
  padding: 0.45rem 0.55rem;
  text-align: right;
  white-space: nowrap;
}
.results-table th {
  background: rgba(255,255,255,0.04);
  color: var(--muted);
  text-align: center;
}
.results-table td:nth-child(-n+7) { text-align: left; }
.cells { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
.cell-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}
.cell-title { font-weight: 600; font-size: 1.05rem; }
.cell-sub { font-size: 0.8rem; color: var(--muted); }
.headline { font-size: 2.25rem; font-weight: 700; color: var(--accent); line-height: 1; }
.headline-unit { font-size: 0.9rem; font-weight: 500; color: var(--muted); margin-left: 0.35rem; }
.cell-mini-tl {
  display: flex;
  gap: 4px;
  min-height: 36px;
  border-radius: 6px;
  overflow: hidden;
  font-size: 0.65rem;
}
.cell-mini-seg {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 0.25rem;
  background: rgba(255,255,255,0.05);
  min-width: 40px;
}
.cell-mini-lbl { color: var(--muted); }
.cell-mini-val { font-weight: 600; color: var(--accent2); }
.tiers { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.chip {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 0.2rem 0.5rem;
  border-radius: 999px;
}
.chip-pass { background: rgba(61,214,140,0.15); color: var(--pass); }
.chip-fail { background: rgba(255,92,106,0.15); color: var(--fail); }
.chip-record { background: rgba(107,159,255,0.12); color: var(--record); }
.chip-na { background: rgba(92,99,112,0.2); color: var(--na); }
.metric-row { margin-bottom: 0.65rem; }
.metric-label { font-size: 0.75rem; color: var(--muted); }
.metric-val { font-size: 1rem; font-weight: 600; }
.bar-track {
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  margin-top: 0.35rem;
  overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 2px; }
.bar-pass { background: var(--pass); }
.bar-fail { background: var(--fail); }
.bar-record { background: var(--record); }
.target { font-size: 0.7rem; color: var(--muted); margin-left: 0.25rem; }
.margin { font-size: 0.7rem; color: var(--pass); display: block; margin-top: 0.15rem; }
.margin-fail { color: var(--fail); }
.cell-foot { font-size: 0.65rem; color: var(--muted); margin-top: auto; padding-top: 0.5rem; border-top: 1px solid var(--border); }
.muted { color: var(--muted); }
.notes { font-size: 0.85rem; color: var(--muted); margin-top: 0.75rem; }
footer.page-foot { text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 2rem; }
@media (max-width: 640px) {
  body { padding: 1rem; }
  .hero-head { flex-direction: column; }
  .headline { font-size: 1.75rem; }
  .chart { gap: 0.75rem; height: 140px; }
  .chart-col { width: 52px; }
}
@media print {
  body { background: #fff; color: #111; padding: 0.5in; }
  .panel, .cell-card, .summary-card, .chart-panel {
    box-shadow: none;
    break-inside: avoid;
    background: #fff;
    border-color: #ccc;
  }
  .status-badge, .chip, .matrix-pass, .matrix-fail { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
  a { color: #06c; }
}
"""


def render_run_deck_html(payload: dict) -> str:
    """Return a self-contained HTML document for the run deck."""
    rc = payload["run_card"]
    lifecycle = payload.get("lifecycle") or {}
    cells = payload.get("cells") or []
    chart_series = payload.get("chart_series") or {}
    summaries = payload.get("sweep_summaries") or []
    gate_matrix = payload.get("gate_matrix") or []
    results_table = payload.get("results_table") or {}
    provenance = payload.get("provenance") or {}
    overall = payload.get("overall_status", "na")

    timeline_total = sum(lifecycle.get(lbl, 0.0) for lbl in _TIMELINE_LABELS) or 1.0

    def chip(status: str, label: str) -> str:
        return f'<span class="chip chip-{html.escape(status)}">{html.escape(label)}</span>'

    upstream = rc.get("upstream_run_url") or ""
    upstream_display = _link_or_text(upstream, "ATOM upstream run") if upstream else "—"
    pytest_link = _link_or_text(provenance.get("pytest_html_path", ""), "pytest HTML report")
    log_link = _link_or_text(provenance.get("log_file_path", ""), "run log")

    hero_rows = [
        ("Model", rc["model"]),
        ("GPU", rc["gpu_arch"]),
        ("Driver", rc["driver"]),
        ("IX recipe", rc["ix_recipe_id"] or "—"),
        ("Image pin", rc["image_pin"] or "—"),
        ("TP", str(rc["tp"])),
        ("Thresholds", "enforced" if rc["enforce_thresholds"] else "record-only"),
        ("Upstream", upstream_display),
        ("Pytest report", pytest_link),
        ("Run log", log_link),
        ("CVS", payload.get("cvs_version", "—")),
        ("Generated", payload.get("generated_at", "—")),
    ]
    hero_html = "".join(
        f"<div class='meta-item'><span class='meta-k'>{html.escape(k)}</span>"
        f"<span class='meta-v'>{v if k in ('Upstream', 'Pytest report', 'Run log') else html.escape(str(v))}</span></div>"
        for k, v in hero_rows
    )

    timeline_parts = []
    for lbl in _TIMELINE_LABELS:
        sec = lifecycle.get(lbl, 0.0)
        if sec <= 0:
            continue
        pct = 100.0 * sec / timeline_total
        timeline_parts.append(
            f"<div class='tl-seg' style='flex-grow:{pct:.2f}' title='{html.escape(lbl)}: {sec:.1f}s'>"
            f"<span class='tl-lbl'>{html.escape(lbl.replace('_', ' '))}</span>"
            f"<span class='tl-val'>{sec:.1f}s</span></div>"
        )
    timeline_html = "".join(timeline_parts) or "<p class='muted'>No lifecycle timings recorded.</p>"

    summary_cards = []
    for s in summaries:
        sat_note = " · saturated at max C" if s.get("saturated") else ""
        summary_cards.append(
            f"<article class='summary-card'>"
            f"<h3>ISL={html.escape(str(s['isl']))} · OSL={html.escape(str(s['osl']))}</h3>"
            f"<div class='summary-stat'>{_fmt_num(s['max_output_throughput'])} <span class='headline-unit'>tok/s</span></div>"
            f"<div class='summary-meta'>Peak at C={html.escape(str(s['conc_at_max_tput']))}"
            f" · TTFT {_fmt_num(s.get('ttft_at_max_tput'))} ms"
            f"{html.escape(sat_note)}</div></article>"
        )
    summary_html = "".join(summary_cards) or "<p class='muted'>No sweep summary (no throughput data).</p>"

    chart_accent = ("accent", "accent2", "accent3")
    chart_parts = []
    for idx, (short, title, unit, invert) in enumerate(_CHART_SERIES):
        points = chart_series.get(short, [])
        part = _render_bar_chart(title, points, unit, invert=invert, accent=chart_accent[idx % 3])
        if part:
            chart_parts.append(part)
    charts_html = (
        f"<div class='chart-grid'>{''.join(chart_parts)}</div>"
        if chart_parts
        else "<p class='muted'>Concurrency charts need two or more sweep cells.</p>"
    )

    matrix_header = "<tr><th>Cell</th>" + "".join(
        f"<th>{html.escape(t)}</th>" for t in METRIC_TIER_ORDER
    ) + "</tr>"
    matrix_rows = []
    for row in gate_matrix:
        cells_html = "".join(
            f"<td class='matrix-{html.escape(row['tiers'].get(t, 'na'))}'>"
            f"{html.escape(row['tiers'].get(t, 'na'))}</td>"
            for t in METRIC_TIER_ORDER
        )
        matrix_rows.append(
            f"<tr><td>{html.escape(row['label'])}</td>{cells_html}</tr>"
        )
    matrix_html = (
        f"<table class='matrix'>{matrix_header}{''.join(matrix_rows)}</table>"
        if matrix_rows
        else "<p class='muted'>No gate matrix (no cells recorded).</p>"
    )

    rt_headers = results_table.get("headers") or []
    rt_rows = results_table.get("rows") or []
    results_header = "<tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in rt_headers) + "</tr>"
    results_body = []
    for row in rt_rows:
        results_body.append(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        )
    results_html = (
        f"<table class='results-table'>{results_header}{''.join(results_body)}</table>"
        if rt_rows
        else "<p class='muted'>No results table rows.</p>"
    )

    cell_cards = []
    for c in cells:
        tier_chips = "".join(chip(c["tiers"].get(t, "na"), t) for t in METRIC_TIER_ORDER)
        metric_rows = []
        for m in c["metrics"]:
            if m["actual"] is None:
                continue
            bar = ""
            if m["bar_pct"] is not None:
                bar = (
                    f"<div class='bar-track'><div class='bar-fill bar-{m['status']}' "
                    f"style='width:{m['bar_pct']:.0f}%'></div></div>"
                )
            target = ""
            if m["spec"] is not None and rc["enforce_thresholds"]:
                target = f"<span class='target'>gate {_fmt_num(m['spec'].get('value'))}</span>"
            margin = ""
            if m.get("margin"):
                margin_cls = "margin-fail" if m["status"] == "fail" else "margin"
                margin = f"<span class='{margin_cls}'>{html.escape(m['margin'])}</span>"
            metric_rows.append(
                f"<div class='metric-row'>"
                f"<div class='metric-label'>{html.escape(m['label'])}</div>"
                f"<div class='metric-val'>{_fmt_num(m['actual'])} {html.escape(m['unit'])}</div>"
                f"{bar}{target}{margin}</div>"
            )

        mini_tl = c.get("cell_lifecycle") or {}
        mini_total = sum(mini_tl.get(lbl, 0.0) for lbl in _CELL_LIFECYCLE_LABELS) or 1.0
        mini_parts = []
        for lbl in _CELL_LIFECYCLE_LABELS:
            sec = mini_tl.get(lbl, 0.0)
            if sec <= 0:
                continue
            pct = 100.0 * sec / mini_total
            mini_parts.append(
                f"<div class='cell-mini-seg' style='flex-grow:{pct:.2f}'>"
                f"<span class='cell-mini-lbl'>{html.escape(lbl.replace('_', ' '))}</span>"
                f"<span class='cell-mini-val'>{sec:.1f}s</span></div>"
            )
        mini_html = (
            f"<div class='cell-mini-tl'>{''.join(mini_parts)}</div>"
            if mini_parts
            else ""
        )

        headline = next(
            (m for m in c["metrics"] if m["metric"] == "client.output_throughput"),
            None,
        )
        headline_val = _fmt_num(headline["actual"]) if headline else "—"
        cell_cards.append(
            f"<article class='cell-card'>"
            f"<header><div class='cell-title'>{html.escape(c['policy'])}</div>"
            f"<div class='cell-sub'>ISL={html.escape(str(c['isl']))} OSL={html.escape(str(c['osl']))} "
            f"· C={html.escape(str(c['concurrency']))}</div></header>"
            f"<div class='headline'>{headline_val}<span class='headline-unit'>tok/s</span></div>"
            f"{mini_html}"
            f"<div class='tiers'>{tier_chips}</div>"
            f"<div class='metrics'>{''.join(metric_rows)}</div>"
            f"<footer class='cell-foot'>{html.escape(c['cell_id'])} · {html.escape(str(c['host']))}</footer>"
            f"</article>"
        )

    cells_html = (
        "".join(cell_cards)
        if cell_cards
        else "<p class='muted'>No sweep cells recorded (run may have skipped early).</p>"
    )

    notes = ""
    if rc.get("notes"):
        notes = f"<p class='notes'>{html.escape(rc['notes'])}</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>IX Run Deck — {html.escape(rc['model'])}</title>
<style>{_deck_css()}</style>
</head>
<body>
<div class="wrap">
  <div class="hero-head">
    <div>
      <h1>IX Run Deck</h1>
      <p class="subtitle">InferenceX ATOM · lab performance summary</p>
    </div>
    {_status_badge(overall)}
  </div>
  <section class="panel">
    <h2>Run card</h2>
    <div class="meta-grid">{hero_html}</div>
    {notes}
  </section>
  <section class="panel">
    <h2>Lifecycle timeline</h2>
    <div class="tl-row">{timeline_html}</div>
  </section>
  <section class="panel">
    <h2>Sweep analytics</h2>
    <div class="summary-grid">{summary_html}</div>
    {charts_html}
  </section>
  <section class="panel">
    <h2>Gate matrix</h2>
    <div class="matrix-wrap">{matrix_html}</div>
  </section>
  <section class="panel">
    <h2>Sweep cells</h2>
    <div class="cells">{cells_html}</div>
  </section>
  <section class="panel">
    <h2>Full results</h2>
    <div class="results-wrap">{results_html}</div>
  </section>
  <footer class="page-foot">CVS inferencex_atom_single · render-only · does not affect gates</footer>
</div>
</body>
</html>"""


def render_run_deck_embed_html(payload: dict, *, height: str = "900px") -> str:
    """Collapsible embed snippet for pytest-html (iframe srcdoc isolation)."""
    doc = render_run_deck_html(payload)
    return (
        '<details class="ix-run-deck-embed" open>'
        "<summary><strong>IX Run Deck</strong> (embedded)</summary>"
        f'<iframe title="IX Run Deck" srcdoc="{html.escape(doc, quote=True)}" '
        f'style="width:100%;height:{html.escape(height)};border:1px solid #ccc;'
        f'border-radius:8px;margin-top:0.5rem;"></iframe>'
        "</details>"
    )


def write_run_deck(
    path: Path,
    *,
    variant_config,
    inf_res_dict: Mapping[tuple, Any],
    lifecycle_report: Mapping[str, list],
    cvs_version: str = "unknown",
    pytest_html_path: str = "",
    log_file_path: str = "",
) -> dict:
    """Build payload, render HTML + JSON sidecar. Returns artifact paths and payload."""
    payload = build_run_deck_payload(
        variant_config=variant_config,
        inf_res_dict=inf_res_dict,
        lifecycle_report=lifecycle_report,
        cvs_version=cvs_version,
        pytest_html_path=pytest_html_path,
        log_file_path=log_file_path,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_run_deck_html(payload), encoding="utf-8")

    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return {"html": path, "json": json_path, "payload": payload}
