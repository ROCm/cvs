'''One-page CI summary HTML for inference suite reports.'''

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, List, Mapping, Optional

from cvs.lib.report.cell_build import cell_has_any_tier_failure
from cvs.lib.report.formatting import status_badge_css, status_badge_html
from cvs.lib.report.json_io import load_report_json
from cvs.lib.report.types import InferenceReportConfig

_PARITY_JSON = "inference_parity_report.json"


def ci_summary_basename(report_basename: str) -> str:
    return f"{report_basename}_summary.html"


def _cell_worst_score(
    cell: dict,
    gated_tiers: tuple[str, ...],
    headline_metric: str,
) -> tuple:
    failed = sum(1 for t in gated_tiers if cell.get("tiers", {}).get(t) == "fail")
    if failed:
        return (0, -failed, cell.get("cell_id", ""))
    if cell_has_any_tier_failure(cell):
        return (1, 0, cell.get("cell_id", ""))
    headline = (cell.get("actuals") or {}).get(headline_metric)
    try:
        tput = float(headline) if headline is not None else float("inf")
    except (TypeError, ValueError):
        tput = float("inf")
    return (2, tput, cell.get("cell_id", ""))


def worst_cells(payload: Mapping[str, Any], config: InferenceReportConfig, *, limit: int = 3) -> List[dict]:
    cells = list(payload.get("cells") or [])
    gated = config.gated_tiers
    headline_metric = config.headline_metric
    scored = sorted(
        ((_cell_worst_score(c, gated, headline_metric), c) for c in cells),
        key=lambda item: item[0],
    )
    worst: List[dict] = []
    for score, cell in scored:
        worst.append(cell)
        if len(worst) >= limit:
            break
    return worst


def _parity_summary(report_dir: Path) -> Optional[dict]:
    path = report_dir / _PARITY_JSON
    if not path.is_file():
        return None
    data = load_report_json(path)
    if not data:
        return None
    rows = data.get("rows") or []
    failed = 0
    for row in rows:
        for val in (row.get("compare") or {}).values():
            if val is not None and float(val) < 0.95:
                failed += 1
                break
    status = "fail" if failed else ("pass" if rows else "na")
    return {
        "status": status,
        "failed_rows": failed,
        "total_rows": len(rows),
        "basename": _PARITY_JSON.replace(".json", ".html"),
    }


def _prev_run_regressions(payload: Mapping[str, Any]) -> int:
    panel = (payload.get("panels") or {}).get("prev_run") or {}
    return sum(1 for row in panel.get("rows") or [] if row.get("regression"))


def render_ci_summary_html(
    payload: Mapping[str, Any],
    config: InferenceReportConfig,
    *,
    full_report_basename: str,
    parity: Optional[dict] = None,
) -> str:
    report = payload.get("report") or {}
    overall = payload.get("overall_status", "na")
    title = report.get("title", config.title)
    subtitle = report.get("subtitle", config.subtitle)
    generated = payload.get("generated_at", "")
    suite_id = payload.get("suite_id", config.suite_id)

    highlights = worst_cells(payload, config, limit=3)
    highlight_rows = []
    for cell in highlights:
        tiers = cell.get("tiers") or {}
        failed = [t for t in config.gated_tiers if tiers.get(t) == "fail"]
        reason = f"failed: {', '.join(failed)}" if failed else "lowest throughput / watch"
        tput = (cell.get("actuals") or {}).get(config.headline_metric)
        highlight_rows.append(
            f"<li><strong>{html.escape(str(cell.get('cell_id', '')))}</strong>"
            f" &middot; C={cell.get('concurrency', '')}"
            f" &middot; {html.escape(reason)}"
            f" &middot; {html.escape(str(tput) if tput is not None else '—')} tok/s</li>"
        )
    highlights_html = (
        "<ul class='highlights'>" + "".join(highlight_rows) + "</ul>"
        if highlight_rows
        else "<p class='muted'>No cells recorded.</p>"
    )

    regressions = _prev_run_regressions(payload)
    prev_line = (
        f"<p><strong>Baseline regressions:</strong> {regressions}</p>"
        if regressions
        else "<p><strong>Baseline regressions:</strong> none flagged</p>"
    )

    if parity:
        parity_line = (
            f"<p><strong>Framework parity:</strong> "
            f"{status_badge_html(str(parity['status']))} "
            f"({parity.get('failed_rows', 0)} failed / {parity.get('total_rows', 0)} rows) "
            f"&middot; <a href='{html.escape(parity.get('basename', ''))}'>full parity report</a></p>"
        )
    else:
        parity_line = "<p><strong>Framework parity:</strong> not generated this run</p>"

    full_html = f"{full_report_basename}.html"
    viewer = (payload.get("summary") or {}).get("viewer_html")
    viewer_link = f" &middot; <a href='{html.escape(viewer)}'>viewer</a>" if viewer else ""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)} — CI summary</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1.5rem; line-height: 1.5; max-width: 52rem; }}
h1 {{ margin: 0 0 0.25rem; font-size: 1.35rem; }}
.subtitle {{ color: #555; margin: 0 0 1rem; }}
.meta {{ font-size: 0.85rem; color: #666; margin-bottom: 1rem; }}
.panel {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }}
.highlights {{ margin: 0.5rem 0 0; padding-left: 1.25rem; }}
{status_badge_css(light=True)}
a {{ color: #0b5fff; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="subtitle">{html.escape(subtitle)}</p>
<p class="meta">{html.escape(suite_id)} &middot; {html.escape(generated)}</p>
<section class="panel">
  <p><strong>Overall:</strong> {status_badge_html(str(overall))}</p>
  {prev_line}
  {parity_line}
  <p><a href="{html.escape(full_html)}">Open full suite report</a>{viewer_link}</p>
</section>
<section class="panel">
  <h2 style="margin:0 0 0.5rem;font-size:1rem;">Cells to review</h2>
  {highlights_html}
</section>
</body></html>"""


def write_inference_ci_summary(
    payload: Mapping[str, Any],
    config: InferenceReportConfig,
    report_dir: Path,
    *,
    parity_json_path: Optional[Path] = None,
) -> Path:
    report_dir = Path(report_dir)
    out = report_dir / ci_summary_basename(config.report_basename)
    parity = None
    if (parity_json_path and parity_json_path.is_file()) or (report_dir / _PARITY_JSON).is_file():
        parity = _parity_summary(report_dir)
    out.write_text(
        render_ci_summary_html(
            payload,
            config,
            full_report_basename=config.report_basename,
            parity=parity,
        ),
        encoding="utf-8",
    )
    return out
