'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Generic inference framework parity.

Merges aligned sweep cells from multiple inference ``*_report.json`` sidecars.
Framework ids (reference and comparators) are arbitrary strings supplied by the
caller. Independent of scaling panels and training parity.

Example::

    from pathlib import Path
    from cvs.lib.report.parity.inference import (
        InferenceParitySource,
        build_inference_parity_config,
        write_inference_parity_report,
    )

    config = build_inference_parity_config(
        reference=InferenceParitySource("ref", "Reference", "ref_report.json"),
        comparators=(InferenceParitySource("b", "Baseline B", "b_report.json"),),
    )
    write_inference_parity_report(Path("parity.html"), config=config)

Suite-specific presets (e.g. IX W1 triple) live under ``cvs.lib.report.presets``.
'''

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from cvs.lib import globals
from cvs.lib.report.types import (
    InferenceParityConfig,
    InferenceParityMetric,
    InferenceParitySource,
    InferenceReportConfig,
)

log = globals.log

CELL_KEY_FIELDS = ("isl", "osl", "concurrency", "policy")

# Default client metrics for auto-generated ``compare.<id>.<short>_ratio`` keys.
STANDARD_CLIENT_PARITY_METRICS = (
    "client.output_throughput",
    "client.mean_ttft_ms",
    "client.mean_tpot_ms",
)


def _cell_key(cell: dict) -> tuple:
    return tuple(str(cell.get(f, "")) for f in CELL_KEY_FIELDS)


def _index_cells(report_json: dict) -> dict[tuple, dict]:
    out = {}
    for cell in report_json.get("cells") or []:
        out[_cell_key(cell)] = cell
    return out


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_value(cell: Optional[dict], metric: str) -> Optional[float]:
    if not cell:
        return None
    raw = (cell.get("actuals") or {}).get(metric)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _ratio(candidate: Optional[float], reference: Optional[float]) -> Optional[float]:
    if candidate is None or reference is None:
        return None
    if reference == 0:
        return None
    return candidate / reference


def _fmt(value) -> str:
    if value is None:
        return "\u2014"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def _cell_label(key: tuple) -> str:
    isl, osl, conc, policy = (key + ("", "", "", ""))[:4]
    parts = [f"ISL={isl}", f"OSL={osl}", f"C={conc}"]
    if policy:
        parts.append(str(policy))
    return " ".join(parts)


def load_framework_cell_indexes(
    sources: tuple[InferenceParitySource, ...],
) -> dict[str, dict[tuple, dict]]:
    return {src.framework_id: _index_cells(_load_json(src.json_path)) for src in sources}


def _resolve_reference_framework_id(config: InferenceParityConfig) -> str:
    if config.reference_framework_id:
        return config.reference_framework_id
    if config.sources:
        return config.sources[0].framework_id
    return "reference"


def default_parity_metrics(
    reference_framework_id: str,
    compare_framework_ids: tuple[str, ...],
    metrics: tuple[str, ...] = STANDARD_CLIENT_PARITY_METRICS,
) -> tuple[InferenceParityMetric, ...]:
    """Build ``compare.<framework>.<metric_short>_ratio`` specs for each comparator."""
    out: list[InferenceParityMetric] = []
    for cmp_id in compare_framework_ids:
        for metric in metrics:
            short = metric.rsplit(".", 1)[-1]
            ratio_key = f"compare.{cmp_id}.{short}_ratio"
            out.append(
                InferenceParityMetric(
                    metric,
                    cmp_id,
                    ratio_key,
                    reference_framework_id=reference_framework_id,
                    title=short.replace("_", " "),
                )
            )
    return tuple(out)


def build_inference_parity_config(
    *,
    reference: InferenceParitySource,
    comparators: tuple[InferenceParitySource, ...],
    metrics: Optional[tuple[InferenceParityMetric, ...]] = None,
    title: str = "Inference framework parity",
    subtitle: str = "Aligned sweep cells across framework report JSON sidecars",
) -> InferenceParityConfig:
    """Assemble parity config from a reference source and zero or more comparators."""
    ref_id = reference.framework_id
    cmp_ids = tuple(c.framework_id for c in comparators)
    if metrics is None:
        metrics = default_parity_metrics(ref_id, cmp_ids)
    return InferenceParityConfig(
        title=title,
        subtitle=subtitle,
        reference_framework_id=ref_id,
        sources=(reference,) + comparators,
        metrics=metrics,
    )


def build_session_parity_config(
    report_config: InferenceReportConfig,
    reference_json_path: str,
) -> Optional[InferenceParityConfig]:
    """Build parity config from ``InferenceReportConfig`` optional compare JSON paths."""
    if not report_config.parity_compare_jsons:
        return None
    ref_id = report_config.parity_reference_framework_id or report_config.suite_id
    reference = InferenceParitySource(ref_id, ref_id, reference_json_path)
    comparators = tuple(
        InferenceParitySource(fw_id, fw_id, path)
        for fw_id, path in report_config.parity_compare_jsons
    )
    metrics = report_config.parity_metrics or None
    return build_inference_parity_config(
        reference=reference,
        comparators=comparators,
        metrics=metrics if metrics else None,
    )


def build_inference_parity_payload(
    config: InferenceParityConfig,
    *,
    cell_indexes: Optional[Mapping[str, dict[tuple, dict]]] = None,
) -> dict:
    """Align sweep cells and emit ``compare.*`` ratios vs the reference framework."""
    if cell_indexes is None:
        cell_indexes = load_framework_cell_indexes(config.sources)

    ref_id = _resolve_reference_framework_id(config)
    ref_cells = cell_indexes.get(ref_id) or {}
    all_keys = set(ref_cells)
    for cells in cell_indexes.values():
        all_keys |= set(cells)

    framework_labels = {s.framework_id: s.label for s in config.sources}
    rows: List[dict] = []
    for key in sorted(all_keys):
        values: Dict[str, Dict[str, Optional[float]]] = {}
        for fw_id, cells in cell_indexes.items():
            values[fw_id] = {}
            cell = cells.get(key)
            for metric in {m.metric for m in config.metrics}:
                values[fw_id][metric] = _metric_value(cell, metric)

        compare: Dict[str, Optional[float]] = {}
        for spec in config.metrics:
            spec_ref = spec.reference_framework_id or ref_id
            ref_val = values.get(spec_ref, {}).get(spec.metric)
            cmp_val = values.get(spec.compare_framework_id, {}).get(spec.metric)
            compare[spec.ratio_key] = _ratio(cmp_val, ref_val)

        rows.append(
            {
                "cell_key": key,
                "cell_label": _cell_label(key),
                "values": values,
                "compare": compare,
            }
        )

    return {
        "schema_version": 1,
        "report_kind": "inference_parity",
        "title": config.title,
        "subtitle": config.subtitle,
        "reference_framework_id": ref_id,
        "frameworks": [
            {"id": s.framework_id, "label": s.label, "json_path": s.json_path}
            for s in config.sources
        ],
        "metrics": [
            {
                "metric": m.metric,
                "ratio_key": m.ratio_key,
                "reference_framework_id": m.reference_framework_id,
                "compare_framework_id": m.compare_framework_id,
                "title": m.title or m.metric,
            }
            for m in config.metrics
        ],
        "rows": rows,
    }


def _render_metric_table(
    payload: dict,
    metric: str,
    specs: List[dict],
    framework_labels: dict,
    ref_id: str,
) -> str:
    fw_ids = [ref_id] + sorted(
        {s["compare_framework_id"] for s in specs if s["compare_framework_id"] != ref_id}
    )
    header = "<tr><th>Cell</th>" + "".join(
        f"<th>{html.escape(framework_labels.get(fid, fid))}<br><small>{html.escape(metric)}</small></th>"
        for fid in fw_ids
    ) + "".join(
        f"<th>{html.escape(s['ratio_key'])}</th>" for s in specs
    ) + "</tr>"

    body_parts = []
    for row in payload.get("rows") or []:
        values = row.get("values") or {}
        compare = row.get("compare") or {}
        cells_html = "".join(
            f"<td>{_fmt((values.get(fid) or {}).get(metric))}</td>" for fid in fw_ids
        )
        ratio_html = "".join(f"<td>{_fmt(compare.get(s['ratio_key']))}</td>" for s in specs)
        body_parts.append(
            f"<tr><td>{html.escape(row.get('cell_label', ''))}</td>{cells_html}{ratio_html}</tr>"
        )

    title = specs[0].get("title") or metric if specs else metric
    return (
        f"<section class='panel'><h2>{html.escape(title)}</h2>"
        f"<table class='parity-table'><thead>{header}</thead><tbody>{''.join(body_parts)}</tbody></table></section>"
    )


def render_inference_parity_html(payload: dict) -> str:
    framework_labels = {
        f["id"]: f["label"] for f in payload.get("frameworks") or []
    }
    ref_id = payload.get("reference_framework_id") or "reference"
    metric_specs = payload.get("metrics") or []

    by_metric: dict[str, list] = {}
    for spec in metric_specs:
        by_metric.setdefault(spec["metric"], []).append(spec)

    tables = "".join(
        _render_metric_table(payload, metric, specs, framework_labels, ref_id)
        for metric, specs in by_metric.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(payload.get('title', 'Inference parity'))}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1.5rem; line-height: 1.45; }}
h1 {{ margin-bottom: 0.25rem; }}
.subtitle {{ color: #555; margin-top: 0; }}
.panel {{ margin: 1.5rem 0; }}
.parity-table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
.parity-table th, .parity-table td {{ border: 1px solid #ccc; padding: 0.45rem 0.6rem; text-align: right; }}
.parity-table th:first-child, .parity-table td:first-child {{ text-align: left; }}
.parity-table th {{ background: #f4f4f4; }}
footer {{ color: #666; font-size: 0.85rem; margin-top: 2rem; }}
</style></head><body>
<h1>{html.escape(payload.get('title', 'Inference parity'))}</h1>
<p class="subtitle">{html.escape(payload.get('subtitle', ''))}</p>
{tables or '<p>No aligned cells.</p>'}
<footer>Reference framework: {html.escape(ref_id)}</footer>
</body></html>"""


def write_inference_parity_report(
    out_html: Path,
    *,
    config: InferenceParityConfig,
) -> dict:
    """Write standalone inference parity HTML + JSON."""
    if not config.sources:
        raise ValueError("InferenceParityConfig.sources must include at least one JSON path")
    payload = build_inference_parity_payload(config)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render_inference_parity_html(payload), encoding="utf-8")
    json_path = out_html.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("Inference parity report written: %s (json: %s)", out_html, json_path)
    return {"html": out_html, "json": json_path, "payload": payload}


def publish_inference_parity_report(
    config: InferenceParityConfig,
    *,
    report_manager,
    pytest_config,
    out_basename: Optional[str] = None,
) -> Optional[dict]:
    """Write parity report beside pytest HTML and add to the zip bundle."""
    htmlpath = getattr(pytest_config.option, "htmlpath", None)
    if not htmlpath:
        return None
    html_path = Path(htmlpath).resolve()
    basename = out_basename or config.report_basename
    artifacts = write_inference_parity_report(
        html_path.parent / f"{basename}.html",
        config=config,
    )
    if report_manager and report_manager.is_enabled:
        report_manager.add_html_to_report(artifacts["html"], link_name=config.link_name)
        report_manager.add_html_to_report(
            artifacts["json"], link_name=f"{config.link_name} JSON"
        )
    return artifacts


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge inference suite report JSONs into parity HTML")
    parser.add_argument("--reference", required=True, help="Reference framework report JSON path")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="Comparator framework id and JSON path (repeatable), e.g. fw_b=path.json",
    )
    parser.add_argument("--out", required=True, help="Output HTML path")
    parser.add_argument(
        "--reference-id",
        default="reference",
        help="Reference framework id (default: reference)",
    )
    parser.add_argument(
        "--reference-label",
        default="",
        help="Display label for reference (default: reference id)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    ref_label = args.reference_label or args.reference_id
    reference = InferenceParitySource(args.reference_id, ref_label, args.reference)
    comparators: list[InferenceParitySource] = []
    for item in args.compare:
        fw_id, _, path = item.partition("=")
        if not fw_id or not path:
            raise SystemExit(f"Invalid --compare value: {item!r} (expected id=path)")
        comparators.append(InferenceParitySource(fw_id, fw_id, path))

    config = build_inference_parity_config(
        reference=reference,
        comparators=tuple(comparators),
    )
    write_inference_parity_report(Path(args.out), config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
