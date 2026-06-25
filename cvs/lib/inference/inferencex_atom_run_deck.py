'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Experimental IX Run Deck — a single-page HTML summary for inferencex_atom_single runs.
Render-only; does not affect pytest pass/fail or threshold enforcement.
'''

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from cvs.lib.inference.utils.inferencex_atom_parsing import (
    CLIENT_METRIC_UNITS,
    METRIC_TIER_ORDER,
    tier_metric_specs,
)
from cvs.lib.utils.verdict import _check_one

# Lifecycle stages shown left-to-right on the timeline (max seconds per label).
_TIMELINE_LABELS = (
    "container_launch",
    "sshd_setup",
    "model_fetch",
    "server_ready",
    "client_complete",
    "teardown",
)

_CELL_HIGHLIGHTS = (
    ("output_throughput", "Output tok/s"),
    ("mean_ttft_ms", "Mean TTFT (ms)"),
    ("mean_tpot_ms", "Mean TPOT (ms)"),
    ("p99_ttft_ms", "P99 TTFT (ms)"),
    ("p95_tpot_ms", "P95 TPOT (ms)"),
)


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


def build_run_deck_payload(
    *,
    variant_config,
    inf_res_dict: Mapping[tuple, Any],
    lifecycle_report: Mapping[str, list],
    cvs_version: str = "unknown",
) -> dict:
    """Structured payload for HTML render (also useful for tests)."""
    cells: List[dict] = []
    chart_points: List[Tuple[int, float]] = []

    for key, host_dict in sorted(inf_res_dict.items(), key=lambda kv: (kv[0][4], kv[0][5])):
        model, gpu, isl, osl, policy, conc = key
        if not isinstance(host_dict, dict) or not host_dict:
            continue
        _host, actuals = next(iter(host_dict.items()))
        cell_id = variant_config.cell_key(isl, osl, conc)
        thresholds_cell = variant_config.thresholds.get(cell_id) or {}
        enforce = bool(variant_config.enforce_thresholds)

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
                }
            )

        tiers = {tier: _tier_status(actuals, thresholds_cell, tier, enforce) for tier in METRIC_TIER_ORDER}

        out_tput = actuals.get("client.output_throughput")
        if out_tput is not None:
            try:
                chart_points.append((int(conc), float(out_tput)))
            except (TypeError, ValueError):
                pass

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
            }
        )

    rc = variant_config.run_card
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "cvs_version": cvs_version,
        "run_card": {
            "model": variant_config.model.id,
            "gpu_arch": variant_config.gpu_arch,
            "driver": variant_config.params.driver,
            "ix_recipe_id": variant_config.ix_recipe_id or "",
            "image_pin": rc.atom_image_pin or "",
            "upstream_run_url": rc.upstream_run_url or "",
            "notes": rc.notes or "",
            "enforce_thresholds": variant_config.enforce_thresholds,
            "tp": variant_config.params.tensor_parallelism,
        },
        "lifecycle": _aggregate_lifecycle(lifecycle_report),
        "cells": cells,
        "chart_points": sorted(chart_points, key=lambda p: p[0]),
    }


def render_run_deck_html(payload: dict) -> str:
    """Return a self-contained HTML document for the run deck."""
    rc = payload["run_card"]
    lifecycle = payload.get("lifecycle") or {}
    cells = payload.get("cells") or []
    chart = payload.get("chart_points") or []

    max_tput = max((p[1] for p in chart), default=1.0) or 1.0
    timeline_total = sum(lifecycle.get(lbl, 0.0) for lbl in _TIMELINE_LABELS) or 1.0

    def chip(status: str, label: str) -> str:
        return f'<span class="chip chip-{html.escape(status)}">{html.escape(label)}</span>'

    hero_rows = [
        ("Model", rc["model"]),
        ("GPU", rc["gpu_arch"]),
        ("Driver", rc["driver"]),
        ("IX recipe", rc["ix_recipe_id"] or "—"),
        ("Image pin", rc["image_pin"] or "—"),
        ("TP", str(rc["tp"])),
        ("Thresholds", "enforced" if rc["enforce_thresholds"] else "record-only"),
        ("CVS", payload.get("cvs_version", "—")),
        ("Generated", payload.get("generated_at", "—")),
    ]
    hero_html = "".join(
        f"<div class='meta-item'><span class='meta-k'>{html.escape(k)}</span>"
        f"<span class='meta-v'>{html.escape(v)}</span></div>"
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

    chart_html = ""
    if len(chart) >= 2:
        bars = []
        for conc, tput in chart:
            h = max(8.0, 100.0 * tput / max_tput)
            bars.append(
                f"<div class='chart-col'><div class='chart-bar' style='height:{h:.0f}%'></div>"
                f"<div class='chart-x'>C={conc}</div>"
                f"<div class='chart-y'>{_fmt_num(tput)}</div></div>"
            )
        chart_html = (
            "<section class='panel'><h2>Throughput vs concurrency</h2>"
            f"<div class='chart'>{''.join(bars)}</div></section>"
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
            metric_rows.append(
                f"<div class='metric-row'>"
                f"<div class='metric-label'>{html.escape(m['label'])}</div>"
                f"<div class='metric-val'>{_fmt_num(m['actual'])} {html.escape(m['unit'])}</div>"
                f"{bar}{target}</div>"
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
<style>
:root {{
  --bg: #0f1117;
  --panel: #1a1d27;
  --border: #2a2f3d;
  --text: #e8eaef;
  --muted: #9aa3b5;
  --accent: #ff6b35;
  --pass: #3dd68c;
  --fail: #ff5c6a;
  --record: #6b9fff;
  --na: #5c6370;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
  background: linear-gradient(160deg, #0a0c12 0%, #12151f 40%, #0f1117 100%);
  color: var(--text);
  line-height: 1.45;
  padding: 1.5rem;
}}
.wrap {{ max-width: 1100px; margin: 0 auto; }}
h1 {{
  font-size: 1.75rem;
  font-weight: 600;
  margin: 0 0 0.25rem;
  letter-spacing: -0.02em;
}}
.subtitle {{ color: var(--muted); margin: 0 0 1.5rem; font-size: 0.95rem; }}
.panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}}
.panel h2 {{
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin: 0 0 1rem;
  font-weight: 600;
}}
.meta-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 0.75rem 1rem;
}}
.meta-item {{ display: flex; flex-direction: column; gap: 0.15rem; }}
.meta-k {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
.meta-v {{ font-size: 0.9rem; font-weight: 500; word-break: break-word; }}
.tl-row {{ display: flex; gap: 3px; min-height: 52px; border-radius: 8px; overflow: hidden; }}
.tl-seg {{
  background: linear-gradient(180deg, #2d3548 0%, #232836 100%);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 0.35rem;
  min-width: 48px;
  border-right: 1px solid var(--border);
}}
.tl-lbl {{ font-size: 0.65rem; color: var(--muted); text-align: center; }}
.tl-val {{ font-size: 0.8rem; font-weight: 600; color: var(--accent); }}
.cells {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }}
.cell-card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}}
.cell-title {{ font-weight: 600; font-size: 1.05rem; }}
.cell-sub {{ font-size: 0.8rem; color: var(--muted); }}
.headline {{ font-size: 2.25rem; font-weight: 700; color: var(--accent); line-height: 1; }}
.headline-unit {{ font-size: 0.9rem; font-weight: 500; color: var(--muted); margin-left: 0.35rem; }}
.tiers {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}
.chip {{
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 0.2rem 0.5rem;
  border-radius: 999px;
}}
.chip-pass {{ background: rgba(61,214,140,0.15); color: var(--pass); }}
.chip-fail {{ background: rgba(255,92,106,0.15); color: var(--fail); }}
.chip-record {{ background: rgba(107,159,255,0.12); color: var(--record); }}
.chip-na {{ background: rgba(92,99,112,0.2); color: var(--na); }}
.metric-row {{ margin-bottom: 0.65rem; }}
.metric-label {{ font-size: 0.75rem; color: var(--muted); }}
.metric-val {{ font-size: 1rem; font-weight: 600; }}
.bar-track {{
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  margin-top: 0.35rem;
  overflow: hidden;
}}
.bar-fill {{ height: 100%; border-radius: 2px; transition: width 0.3s; }}
.bar-pass {{ background: var(--pass); }}
.bar-fail {{ background: var(--fail); }}
.bar-record {{ background: var(--record); }}
.target {{ font-size: 0.7rem; color: var(--muted); margin-left: 0.25rem; }}
.cell-foot {{ font-size: 0.65rem; color: var(--muted); margin-top: auto; padding-top: 0.5rem; border-top: 1px solid var(--border); }}
.chart {{
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 2rem;
  height: 180px;
  padding-top: 1rem;
}}
.chart-col {{ display: flex; flex-direction: column; align-items: center; width: 72px; height: 100%; }}
.chart-bar {{
  width: 48px;
  background: linear-gradient(180deg, var(--accent) 0%, #c44d28 100%);
  border-radius: 6px 6px 2px 2px;
  margin-top: auto;
}}
.chart-x {{ font-size: 0.8rem; margin-top: 0.5rem; color: var(--muted); }}
.chart-y {{ font-size: 0.75rem; font-weight: 600; margin-top: 0.25rem; }}
.muted {{ color: var(--muted); }}
.notes {{ font-size: 0.85rem; color: var(--muted); margin-top: 0.75rem; }}
footer.page-foot {{ text-align: center; color: var(--muted); font-size: 0.75rem; margin-top: 2rem; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>IX Run Deck</h1>
  <p class="subtitle">InferenceX ATOM · experimental report</p>
  <section class="panel">
    <h2>Run card</h2>
    <div class="meta-grid">{hero_html}</div>
    {notes}
  </section>
  <section class="panel">
    <h2>Lifecycle timeline</h2>
    <div class="tl-row">{timeline_html}</div>
  </section>
  {chart_html}
  <section class="panel">
    <h2>Sweep cells</h2>
    <div class="cells">{cells_html}</div>
  </section>
  <footer class="page-foot">CVS inferencex_atom_single · render-only · does not affect gates</footer>
</div>
</body>
</html>"""


def write_run_deck(
    path: Path,
    *,
    variant_config,
    inf_res_dict: Mapping[tuple, Any],
    lifecycle_report: Mapping[str, list],
    cvs_version: str = "unknown",
) -> Path:
    """Build payload, render HTML, and write to ``path``. Returns the path written."""
    payload = build_run_deck_payload(
        variant_config=variant_config,
        inf_res_dict=inf_res_dict,
        lifecycle_report=lifecycle_report,
        cvs_version=cvs_version,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_run_deck_html(payload), encoding="utf-8")
    return path
