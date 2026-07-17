#!/usr/bin/env python3
"""Render an RCCL A/B regression report (ab_regression_report.json) as Markdown.

Consumes the JSON written by ``cvs/tests/rccl/rccl_ab_regression.py``
(``{"control_mode": bool, "reports": {group_key: <detect_regressions report>}}``)
and emits a GitHub-flavoured Markdown summary suitable for a PR comment or a
GitHub Actions step summary.

Exit code mirrors the gate verdict so the same invocation can drive a check:
  0 = PASS (no confirmed regressions)
  1 = REGRESSION DETECTED (detect mode) / detector unstable (control mode)
Use ``--no-exit-code`` to always exit 0 (e.g. when only rendering).
"""

import argparse
import json
import sys
from pathlib import Path


def _fmt_size(n):
    """Human-readable byte size (1024-based), e.g. 1024 -> '1K', 4294967296 -> '4G'."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(n)
    for unit in ("", "K", "M", "G", "T"):
        if abs(n) < 1024 or unit == "T":
            return f"{n}{unit}" if unit == "" else f"{n:.0f}{unit}"
        n /= 1024.0
    return str(n)


def _thresholds_line(reports):
    """Pull the per-tier thresholds from the first report (identical across groups)."""
    for rep in reports.values():
        thr = rep.get("config", {}).get("thresholds")
        if thr:
            return (
                f"small {thr.get('small', 0) * 100:.1f}% · "
                f"mid {thr.get('mid', 0) * 100:.1f}% · "
                f"large {thr.get('large', 0) * 100:.1f}%"
            )
    return "n/a"


def _collect(reports):
    """Aggregate counts and flatten confirmed regressions across all groups."""
    totals = {"keys": 0, "regressions": 0, "inconclusive": 0, "candidates": 0}
    regressions = []
    for group_key, rep in reports.items():
        s = rep.get("summary", {})
        totals["keys"] += s.get("keys_compared", 0)
        totals["regressions"] += s.get("regressions", 0)
        totals["inconclusive"] += s.get("inconclusive", 0)
        totals["candidates"] += s.get("candidates", 0)
        for v in rep.get("regressions", []):
            k = v.get("key", {})
            regressions.append(
                {
                    "collective": k.get("name", "?"),
                    "dtype": k.get("type", "?"),
                    "size": k.get("size", 0),
                    "a_med": v.get("a", {}).get("median", 0.0),
                    "b_med": v.get("b", {}).get("median", 0.0),
                    "drop": v.get("rel_drop", 0.0),
                    "thr": v.get("threshold", 0.0),
                }
            )
    regressions.sort(key=lambda r: (r["collective"], str(r["dtype"]), r["size"]))
    return totals, regressions


def render(report_data, title="RCCL Perf-Regression Gate"):
    """Return a Markdown string for the given parsed report JSON."""
    control_mode = bool(report_data.get("control_mode", False))
    reports = report_data.get("reports", {})
    totals, regressions = _collect(reports)
    has_regression = totals["regressions"] > 0

    lines = []
    if control_mode:
        # In a control (A=A) run, any regression is a false positive => gate broken.
        verdict = "❌ DETECTOR UNSTABLE" if has_regression else "✅ STABLE (0 false positives)"
        lines.append(f"## {title}: {verdict}")
        lines.append("")
        lines.append("**Mode:** calibration / control (A=A — same build both sides)")
    else:
        verdict = "❌ REGRESSION DETECTED" if has_regression else "✅ PASS"
        lines.append(f"## {title}: {verdict}")
        lines.append("")
        lines.append("**Mode:** detect (reference vs candidate)")

    lines.append(f"**Thresholds (per-tier, calibrated):** {_thresholds_line(reports)}")
    lines.append(
        f"**Keys compared:** {totals['keys']} · "
        f"**Confirmed regressions:** {totals['regressions']} · "
        f"**Inconclusive:** {totals['inconclusive']}"
    )
    lines.append("")

    if regressions:
        lines.append(f"### Confirmed regressions ({len(regressions)})")
        lines.append("")
        lines.append("| collective | dtype | size | A (ref) GB/s | B (cand) GB/s | drop % | thr % |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in regressions:
            lines.append(
                f"| {r['collective']} | {r['dtype']} | {_fmt_size(r['size'])} "
                f"| {r['a_med']:.2f} | {r['b_med']:.2f} "
                f"| {r['drop'] * 100:.1f} | {r['thr'] * 100:.1f} |"
            )
        lines.append("")

    # Per-group breakdown (collapsed) so reviewers can see coverage / inconclusive spread.
    lines.append("<details><summary>Per-collective breakdown</summary>")
    lines.append("")
    lines.append("| group | keys | regressions | inconclusive |")
    lines.append("|---|---:|---:|---:|")
    for group_key, rep in sorted(reports.items()):
        s = rep.get("summary", {})
        lines.append(
            f"| {group_key} | {s.get('keys_compared', 0)} "
            f"| {s.get('regressions', 0)} | {s.get('inconclusive', 0)} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines), has_regression, control_mode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        nargs="?",
        default="/it-share/rccl-ci/ab_artifacts/ab_regression_report.json",
        help="Path to ab_regression_report.json",
    )
    parser.add_argument("-o", "--output", help="Write Markdown here (default: stdout)")
    parser.add_argument("--title", default="RCCL Perf-Regression Gate", help="Heading title")
    parser.add_argument(
        "--no-exit-code",
        action="store_true",
        help="Always exit 0 (do not map verdict to exit code)",
    )
    args = parser.parse_args(argv)

    path = Path(args.report)
    try:
        report_data = json.loads(path.read_text())
    except FileNotFoundError:
        print(f"error: report not found: {path}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: could not parse {path}: {exc}", file=sys.stderr)
        return 2

    markdown, has_regression, _control = render(report_data, title=args.title)

    if args.output:
        Path(args.output).write_text(markdown)
    else:
        print(markdown)

    if args.no_exit_code:
        return 0
    return 1 if has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
