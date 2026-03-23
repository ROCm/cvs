#!/usr/bin/env python3
"""
Print a human-friendly summary of a JUnit-style XML results file.

Usage:
  - Set env var XML_OUTPUT_FILE and run this script; or
  - Pass --xml /path/to/results.xml

Designed to be used in GitHub Actions after artifacts are uploaded.
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import textwrap
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Counts:
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    time_seconds: float = 0.0

    @property
    def passed(self) -> int:
        return max(0, self.tests - self.failures - self.errors - self.skipped)


@dataclass(frozen=True)
class FailedCase:
    kind: str  # "failure" or "error"
    suite: str
    classname: str
    name: str
    message: str

    @property
    def display_name(self) -> str:
        if self.classname and self.name:
            return f"{self.classname}::{self.name}"
        return self.name or self.classname or "<unknown test>"


@dataclass(frozen=True)
class JUnitMeta:
    root_name: str
    suite_names: tuple[str, ...]
    hostnames: tuple[str, ...]
    testcase_names: tuple[str, ...]


@dataclass(frozen=True)
class FileResult:
    xml_path: str
    label: str
    status: str  # PASS/FAIL/MISSING/PARSE_ERROR/READ_ERROR
    counts: Counts
    meta: Optional[JUnitMeta]
    failed: tuple[FailedCase, ...]
    error: str = ""
    s3_url: str = ""


def _int_attr(elem: ET.Element, key: str) -> int:
    raw = elem.attrib.get(key)
    if raw is None or raw == "":
        return 0
    try:
        return int(float(raw))
    except ValueError:
        return 0


def _float_attr(elem: ET.Element, key: str) -> float:
    raw = elem.attrib.get(key)
    if raw is None or raw == "":
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _iter_suites(root: ET.Element) -> Iterable[ET.Element]:
    # JUnit commonly has either <testsuite> root or <testsuites> root.
    if root.tag == "testsuite":
        yield root
        return
    for suite in root.findall(".//testsuite"):
        yield suite


def parse_junit_xml(xml_path: str) -> tuple[Counts, JUnitMeta, list[FailedCase]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    total = Counts()
    failed: list[FailedCase] = []
    suite_names: list[str] = []
    hostnames: set[str] = set()
    testcase_names: list[str] = []
    root_name = root.attrib.get("name", "")

    for suite in _iter_suites(root):
        suite_name = suite.attrib.get("name", "")
        if suite_name:
            suite_names.append(suite_name)
        hostname = (suite.attrib.get("hostname") or "").strip()
        if hostname:
            hostnames.add(hostname)
        total = Counts(
            tests=total.tests + _int_attr(suite, "tests"),
            failures=total.failures + _int_attr(suite, "failures"),
            errors=total.errors + _int_attr(suite, "errors"),
            skipped=total.skipped + _int_attr(suite, "skipped") + _int_attr(suite, "disabled"),
            time_seconds=total.time_seconds + _float_attr(suite, "time"),
        )

        for tc in suite.findall(".//testcase"):
            classname = tc.attrib.get("classname", "")
            name = tc.attrib.get("name", "")
            if name:
                testcase_names.append(name)

            for kind in ("failure", "error"):
                node = tc.find(kind)
                if node is None:
                    continue
                msg = (node.attrib.get("message") or "").strip()
                if not msg:
                    # Sometimes details are in the node text.
                    msg = (node.text or "").strip()
                msg = msg.replace("\r\n", "\n").replace("\r", "\n")
                if not msg:
                    msg = "<no message>"
                failed.append(
                    FailedCase(
                        kind=kind,
                        suite=suite_name,
                        classname=classname,
                        name=name,
                        message=msg,
                    )
                )

    meta = JUnitMeta(
        root_name=root_name,
        suite_names=tuple(suite_names),
        hostnames=tuple(sorted(hostnames)),
        testcase_names=tuple(testcase_names),
    )
    return total, meta, failed


def _derive_label(xml_path: str) -> str:
    base = os.path.basename(xml_path)
    # Strip common extensions
    for ext in (".junit.xml", ".xml"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return base


def _load_one(xml_path: str, label: str, strict: bool) -> FileResult:
    # strict is included for symmetry/future, but we don't branch behavior here;
    # strict handling happens at the end when computing the exit code.
    if not xml_path:
        return FileResult(
            xml_path=xml_path,
            label=label or "<unset>",
            status="MISSING",
            counts=Counts(),
            meta=None,
            failed=tuple(),
            error="XML path not provided",
        )

    if not os.path.exists(xml_path):
        return FileResult(
            xml_path=xml_path,
            label=label or _derive_label(xml_path),
            status="MISSING",
            counts=Counts(),
            meta=None,
            failed=tuple(),
            error="File not found",
        )

    try:
        counts, meta, failed_list = parse_junit_xml(xml_path)
    except ET.ParseError as e:
        return FileResult(
            xml_path=xml_path,
            label=label or _derive_label(xml_path),
            status="PARSE_ERROR",
            counts=Counts(),
            meta=None,
            failed=tuple(),
            error=str(e),
        )
    except OSError as e:
        return FileResult(
            xml_path=xml_path,
            label=label or _derive_label(xml_path),
            status="READ_ERROR",
            counts=Counts(),
            meta=None,
            failed=tuple(),
            error=str(e),
        )

    status = "PASS" if (counts.failures == 0 and counts.errors == 0) else "FAIL"
    return FileResult(
        xml_path=xml_path,
        label=label or _derive_label(xml_path),
        status=status,
        counts=counts,
        meta=meta,
        failed=tuple(failed_list),
    )


def _read_summary_json(summary_json: str) -> dict:
    if not summary_json.strip():
        return {}
    try:
        data = json.loads(summary_json)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _get_summary_file_path() -> str:
    runner_temp = os.environ.get("RUNNER_TEMP", "")
    if runner_temp:
        return os.path.join(runner_temp, "cvs_test_summary.json")
    return ""


def _load_summary_from_file(file_path: str) -> dict:
    if not file_path or not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_summary_file(file_path: str, summary: dict) -> None:
    if not file_path:
        return
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, separators=(",", ":"), sort_keys=False)
    except OSError:
        pass


def _load_summary(var_name: str) -> dict:
    """Load summary from persistent file (preferred) merged with env var (fallback).

    The file is always up-to-date because every write goes to both the file
    and GITHUB_ENV.  Reading the file avoids stale-env issues when a composite
    action wrote to GITHUB_ENV but the change did not propagate to the next
    workflow step.
    """
    file_path = _get_summary_file_path()
    file_data = _load_summary_from_file(file_path)
    env_data = _read_summary_json(os.environ.get(var_name, "{}"))

    if not file_data:
        return env_data
    if not env_data:
        return file_data

    merged = dict(file_data)
    for label, entry in env_data.items():
        if not isinstance(entry, dict):
            continue
        existing = merged.get(label)
        if not isinstance(existing, dict):
            merged[label] = entry
            continue
        if entry.get("s3_url") and not existing.get("s3_url"):
            existing["s3_url"] = entry["s3_url"]
        if existing.get("s3_url") and not entry.get("s3_url"):
            entry["s3_url"] = existing["s3_url"]
        merged[label] = existing
    return merged


def _save_summary(var_name: str, summary: dict) -> None:
    """Persist summary to both the file and GITHUB_ENV."""
    value = json.dumps(summary, separators=(",", ":"), sort_keys=False)
    _save_summary_file(_get_summary_file_path(), summary)
    _write_github_env_var(var_name, value)


def _load_summary_from_env(var_name: str) -> dict:
    return _load_summary(var_name)


def _summary_upsert(summary: dict, result: FileResult) -> dict:
    existing = summary.get(result.label)
    prev_s3_url = ""
    if isinstance(existing, dict):
        prev_s3_url = existing.get("s3_url", "")

    entry: dict = {
        "xml": result.xml_path,
        "status": result.status,
        "tests": result.counts.tests,
        "passed": result.counts.passed,
        "skipped": result.counts.skipped,
        "failures": result.counts.failures,
        "errors": result.counts.errors,
        "time_seconds": result.counts.time_seconds,
    }
    if result.meta:
        if result.meta.hostnames:
            entry["hosts"] = list(result.meta.hostnames)
        if result.meta.root_name:
            entry["root_name"] = result.meta.root_name
        if result.meta.suite_names:
            entry["suite_names"] = list(sorted(set(result.meta.suite_names)))

    s3_url = result.s3_url or prev_s3_url
    if s3_url:
        entry["s3_url"] = s3_url

    summary[result.label] = entry
    return summary


def _summary_to_results(summary: dict) -> list[FileResult]:
    results: list[FileResult] = []
    for label, entry in summary.items():
        if not isinstance(entry, dict):
            continue

        status = str(entry.get("status", "UNKNOWN"))
        xml_path = str(entry.get("xml", ""))
        counts = Counts(
            tests=int(entry.get("tests", 0) or 0),
            failures=int(entry.get("failures", 0) or 0),
            errors=int(entry.get("errors", 0) or 0),
            skipped=int(entry.get("skipped", 0) or 0),
            time_seconds=float(entry.get("time_seconds", 0.0) or 0.0),
        )

        hosts = entry.get("hosts", [])
        suite_names = entry.get("suite_names", [])
        meta = None
        if isinstance(hosts, list) or isinstance(suite_names, list):
            meta = JUnitMeta(
                root_name=str(entry.get("root_name", "") or ""),
                suite_names=tuple(str(x) for x in (suite_names or []) if x is not None),
                hostnames=tuple(str(x) for x in (hosts or []) if x is not None),
                testcase_names=tuple(),
            )

        results.append(
            FileResult(
                xml_path=xml_path,
                label=str(label),
                status=status,
                counts=counts,
                meta=meta,
                failed=tuple(),
                error=str(entry.get("error", "") or ""),
                s3_url=str(entry.get("s3_url", "") or ""),
            )
        )
    return results


def _write_github_env_var(name: str, value: str) -> None:
    env_path = os.environ.get("GITHUB_ENV")
    if not env_path:
        return
    # Use the multiline-safe syntax to avoid escaping JSON.
    delimiter = "EOF_JUNIT_SUMMARY"
    try:
        with open(env_path, "a", encoding="utf-8") as f:
            f.write(f"{name}<<{delimiter}\n")
            f.write(value)
            if not value.endswith("\n"):
                f.write("\n")
            f.write(f"{delimiter}\n")
    except OSError:
        pass


def _format_seconds(seconds: float) -> str:
    # Keep it readable in a table.
    if seconds >= 3600:
        return f"{seconds/3600:.2f}h"
    if seconds >= 60:
        return f"{seconds/60:.1f}m"
    return f"{seconds:.3f}s"


def _print_table(prefix: str, results: Sequence[FileResult]) -> None:
    # Simple fixed table without external deps.
    headers = ["label", "status", "host(s)", "tests", "passed", "skipped", "failures", "errors", "time", "logs"]
    rows: list[list[str]] = []

    for r in results:
        hosts = ""
        if r.meta and r.meta.hostnames:
            hosts = ",".join(r.meta.hostnames)
        logs_col = r.s3_url if r.s3_url else r.xml_path
        rows.append(
            [
                r.label,
                r.status,
                hosts,
                str(r.counts.tests),
                str(r.counts.passed),
                str(r.counts.skipped),
                str(r.counts.failures),
                str(r.counts.errors),
                _format_seconds(r.counts.time_seconds),
                logs_col,
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cols: Sequence[str]) -> str:
        return " | ".join(cols[i].ljust(widths[i]) for i in range(len(headers)))

    line = "-+-".join("-" * w for w in widths)
    print(prefix + fmt_row(headers))
    print(prefix + line)
    for row in rows:
        print(prefix + fmt_row(row))


def _markdown_table(results: Sequence[FileResult]) -> str:
    def esc(s: str) -> str:
        return s.replace("|", "\\|").replace("\n", " ").strip()

    lines = []
    lines.append("| label | status | host(s) | tests | passed | skipped | failures | errors | time | logs |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        hosts = ""
        if r.meta and r.meta.hostnames:
            hosts = ", ".join(r.meta.hostnames)
        if r.s3_url:
            logs_col = f"[View logs]({esc(r.s3_url)})"
        else:
            logs_col = f"`{esc(r.xml_path)}`"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{esc(r.label)}`",
                    esc(r.status),
                    f"`{esc(hosts)}`" if hosts else "",
                    str(r.counts.tests),
                    str(r.counts.passed),
                    str(r.counts.skipped),
                    str(r.counts.failures),
                    str(r.counts.errors),
                    esc(_format_seconds(r.counts.time_seconds)),
                    logs_col,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _write_step_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(markdown)
            if not markdown.endswith("\n"):
                f.write("\n")
            f.write("\n")
    except OSError:
        # Never fail a job just because step summary couldn't be written.
        pass


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Display a summary of a JUnit XML results file (XML_OUTPUT_FILE)."
    )
    parser.add_argument("--xml", default=None, help="Path to a single JUnit XML file")
    parser.add_argument(
        "--xmls",
        nargs="+",
        default=None,
        help="One or more JUnit XML files (combined tabular summary)",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Label to display for single-file mode (ignored for --xmls unless you want a shared prefix)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if XML is missing/unreadable or if failures/errors are present.",
    )
    parser.add_argument(
        "--update-summary",
        action="store_true",
        help="Update a globally maintained JSON summary (dictionary) and optionally persist it via GITHUB_ENV.",
    )
    parser.add_argument(
        "--summary-env-var",
        default="TEST_SUMMARY_JSON",
        help="Environment variable name that stores the JSON summary dictionary.",
    )
    parser.add_argument(
        "--from-summary-env",
        action="store_true",
        help="Render the combined table from the JSON summary stored in --summary-env-var (no XML inputs required).",
    )
    parser.add_argument(
        "--set-s3-url",
        action="store_true",
        help="Set the s3_url field on an existing summary entry identified by --label. Requires --s3-url.",
    )
    parser.add_argument(
        "--s3-url",
        default="",
        help="S3 console URL to store in the summary entry (used with --set-s3-url).",
    )
    args = parser.parse_args(argv)

    label = (args.label or "").strip()
    prefix = f"[{label}] " if label else ""

    if args.set_s3_url:
        if not label:
            print("ERROR: --set-s3-url requires --label", file=sys.stderr)
            return 2
        s3_url = (args.s3_url or "").strip()
        if not s3_url:
            print("ERROR: --set-s3-url requires --s3-url <url>", file=sys.stderr)
            return 2
        summary = _load_summary(args.summary_env_var)
        if label in summary and isinstance(summary[label], dict):
            summary[label]["s3_url"] = s3_url
            _save_summary(args.summary_env_var, summary)
            print(f"Updated {args.summary_env_var}[{label}].s3_url")
        else:
            print(f"WARNING: label '{label}' not found in {args.summary_env_var}; skipping s3_url update")
        return 0

    if args.from_summary_env:
        summary = _load_summary(args.summary_env_var)
        results = _summary_to_results(summary)
        if not results:
            print(f"{prefix}No entries found in {args.summary_env_var}; nothing to summarize.")
            return 0

        print(f"::group::{prefix}JUnit results summary (from {args.summary_env_var})")
        _print_table(prefix, results)
        print("::endgroup::")

        md = []
        title = f"## {label} — JUnit results summary\n\n" if label else "## JUnit results summary\n\n"
        md.append(title)
        md.append(_markdown_table(results))
        _write_step_summary("".join(md))
        if args.strict:
            for r in results:
                if r.status in ("MISSING", "PARSE_ERROR", "READ_ERROR"):
                    return 2
                if r.counts.failures > 0 or r.counts.errors > 0 or r.status == "FAIL":
                    return 1
        return 0

    xml_paths: list[str] = []
    if args.xmls:
        xml_paths.extend(args.xmls)
    else:
        # Backwards-compatible default: if neither --xml nor --xmls is set,
        # fall back to XML_OUTPUT_FILE.
        xml_single = args.xml if args.xml is not None else os.environ.get("XML_OUTPUT_FILE")
        if xml_single:
            xml_paths.append(xml_single)

    if not xml_paths:
        print(f"{prefix}XML_OUTPUT_FILE is not set and neither --xml nor --xmls was provided.")
        return 2 if args.strict else 0

    # Combined output
    print(f"::group::{prefix}JUnit results summary")
    results: list[FileResult] = []
    for p in xml_paths:
        # For --xmls we derive per-file labels from filenames; in single mode we honor --label.
        per_file_label = "" if args.xmls else label
        results.append(_load_one(p, per_file_label, args.strict))

    if args.update_summary:
        summary = _load_summary(args.summary_env_var)
        for r in results:
            summary = _summary_upsert(summary, r)

        print(f"{prefix}Updated {args.summary_env_var} with {len(results)} result(s).")
        _save_summary(args.summary_env_var, summary)
        print("::endgroup::")
        # In update mode, don't spam details; honor --strict semantics.
        if args.strict:
            for r in results:
                if r.status in ("MISSING", "PARSE_ERROR", "READ_ERROR"):
                    return 2
                if r.counts.failures > 0 or r.counts.errors > 0:
                    return 1
        return 0

    _print_table(prefix, results)

    # Optionally print per-file failure details (kept short, high-signal).
    for r in results:
        if r.status in ("MISSING", "PARSE_ERROR", "READ_ERROR"):
            print("")
            print(f"{prefix}{r.label}: {r.status} ({r.error})")
            continue
        if r.failed:
            max_to_print = 25
            print("")
            print(f"{prefix}{r.label}: failures/errors (showing up to {max_to_print})")
            for fc in r.failed[:max_to_print]:
                first_line = fc.message.split("\n", 1)[0].strip()
                first_line = textwrap.shorten(first_line, width=220, placeholder="…") if first_line else ""
                print(f"- [{fc.kind}] {fc.display_name}" + (f" — {first_line}" if first_line else ""))
            if len(r.failed) > max_to_print:
                print(f"{prefix}... and {len(r.failed) - max_to_print} more in {r.label}")

    print("::endgroup::")

    # GitHub step summary (markdown table)
    md = []
    if label:
        md.append(f"## {label} — JUnit results summary\n\n")
    else:
        md.append("## JUnit results summary\n\n")
    md.append(_markdown_table(results))
    _write_step_summary("".join(md))

    if args.strict:
        for r in results:
            if r.status in ("MISSING", "PARSE_ERROR", "READ_ERROR"):
                return 2
            if r.counts.failures > 0 or r.counts.errors > 0:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


