'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Build one consolidated report from the Run Decks under one or more directories.
'''

import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from cvs.lib import globals
from cvs.lib.report.consolidate import areas, discovery, normalize, rollup, writer
from cvs.lib.report.consolidate.warning_log import INVALID_DECK, MISSING_FIELD, UNKNOWN_SUITE_ID, WarningLog
from cvs.lib.report.rundeck.publish_helpers import cvs_version

log = globals.log

SCHEMA_VERSION = 1
SUITE_PAGES_DIR = "suites"
_UNSAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")
_SUITE_PAGE_FIELDS = (
    "id",
    "name",
    "category",
    "status",
    "generated",
    "cluster",
    "counts",
    "total",
    "unit",
    "partial",
    "notes",
)
_SUITE_PAGE_LINKS = (
    ("deck", "reportUrl"),
    ("json", "jsonUrl"),
    ("viewer", "viewerUrl"),
    ("pytest", "pytestUrl"),
    ("log", "logUrl"),
)


class ConsolidationError(Exception):
    '''No report could be written: a directory is missing, it holds no decks, or the output cannot be written.'''


class ConsolidationResult:
    '''What one consolidation found and wrote.'''

    def __init__(self, zip_path, summary, found, duplicates):
        self.zip_path = zip_path
        self.summary = summary
        self.found = found
        self.duplicates = duplicates


def resolve_log_dirs(log_dirs):
    '''Directories to search, from a comma-separated string or a list. Raises ConsolidationError for a missing one.'''
    if isinstance(log_dirs, (str, Path)):
        log_dirs = str(log_dirs).split(",")
    roots = [Path(str(item).strip()).expanduser().resolve() for item in log_dirs if str(item).strip()]
    if not roots:
        raise ConsolidationError("No directory given to search for Run Decks.")
    missing = [str(root) for root in roots if not root.is_dir()]
    if missing:
        raise ConsolidationError("Directory not found: %s" % ", ".join(missing))
    return list(dict.fromkeys(roots))


def variant_id(deck):
    '''Suite variant key, which is the stem, made safe for file names.'''
    return _UNSAFE_ID.sub("_", deck.stem or "").strip("._") or "unknown"


def consolidate(log_dirs, output_dir=None):
    '''Write one consolidated report zip over every Run Deck under ``log_dirs`` and return a ConsolidationResult.

    The zip goes to ``output_dir``; by default the searched directory, or the current
    directory when several are searched.
    '''
    roots = resolve_log_dirs(log_dirs)
    if output_dir:
        output = Path(output_dir).expanduser().resolve()
    else:
        output = roots[0] if len(roots) == 1 else Path.cwd().resolve()
    log.info("Searching for Run Decks under %s", ", ".join(str(root) for root in roots))
    warnings = WarningLog()
    found, ignored = discovery.discover(roots, warnings)
    if not found:
        raise ConsolidationError("No Run Decks found under %s" % ", ".join(str(root) for root in roots))
    decks, duplicates = discovery.drop_duplicates(found)
    in_bundles = sum(1 for deck in found if deck.in_bundle)
    log.info(
        "Found %d Run Decks (%d in run folders, %d in zip bundles); %d duplicates dropped; %d other files ignored",
        len(found),
        len(found) - in_bundles,
        in_bundles,
        len(duplicates),
        ignored,
    )

    name = discovery.OUTPUT_PREFIX + datetime.now().strftime("%Y-%m-%dT%H%M%S")
    try:
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=discovery.WORK_PREFIX + ".tmp-", dir=output))
    except OSError as exc:
        raise ConsolidationError("Cannot write to %s: %s" % (output, exc)) from exc
    try:
        report_root = work / name
        report_root.mkdir()
        summary = build_report(decks, report_root, warnings)
        zip_path = writer.write_zip(report_root, output)
    except OSError as exc:
        raise ConsolidationError("Could not write the consolidated report: %s" % exc) from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if work.exists():
            log.warning("Could not remove the temporary folder %s; delete it by hand.", work)

    log.info("Collection verdict: %s", summary["verdict"])
    log.info("Consolidated report written: %s", zip_path)
    return ConsolidationResult(zip_path, summary, found, duplicates)


def build_report(decks, report_root, warnings):
    '''Write ``index.html``, ``summary.json``, the details files, and every run's files under ``report_root``.

    Returns the summary.
    '''
    groups = {}
    for deck in decks:
        groups.setdefault(variant_id(deck), []).append(deck)
    suites, details = [], {}
    for variant in sorted(groups):
        suite, detail = _build_suite(variant, groups[variant], report_root, warnings)
        suites.append(suite)
        details[variant] = detail

    summary = {
        "schema_version": SCHEMA_VERSION,
        "sample": False,
        "generated": rollup.generated_range(suite["_time"] for suite in suites),
        "verdict": rollup.collection_verdict(suites),
        "recommendations": rollup.recommendations(suites),
        "manifest": rollup.build_manifest(suites),
        "warnings": warnings.entries,
        "suites": [_public(suite) for suite in suites],
    }
    for suite in summary["suites"]:
        detail = dict(details[suite["id"]], schema_version=SCHEMA_VERSION, summary=suite)
        writer.write_json(report_root / suite["detailUrl"], detail)
    writer.write_json(report_root / "summary.json", summary)
    area_order = areas.ordered_areas(suite["category"] for suite in suites)
    version = cvs_version()
    _write_suite_pages(suites, details, report_root, area_order, version)
    page_data = {
        "summary": summary,
        "details": details,
        "areas": area_order,
        "built": datetime.now(timezone.utc).strftime(rollup.LEGACY_TIME),
        "cvs_version": version,
    }
    (report_root / "index.html").write_text(writer.render_index(page_data), encoding="utf-8")

    by_area = Counter(suite["category"] for suite in suites)
    runs = sum(1 + len(suite["history"]) for suite in suites)
    unavailable = sum(1 for deck in decks if not discovery.is_valid(deck.data))
    log.info(
        "Suites: %d (%s); runs: %d; valid decks: %d; unavailable decks: %d",
        len(suites),
        ", ".join("%s %d" % (area, count) for area, count in sorted(by_area.items())),
        runs,
        len(decks) - unavailable,
        unavailable,
    )
    return summary


def _public(suite):
    return {key: value for key, value in suite.items() if not key.startswith("_")}


def _write_suite_pages(suites, details, report_root, area_order, version):
    '''Write ``suites/<id>.html`` for every suite whose Run Deck could be read.'''
    paged = sorted(
        (suite for suite in suites if suite["suiteUrl"]),
        key=lambda suite: (area_order.index(suite["category"]), suite["name"].lower()),
    )
    for suite in paged:
        links = {"overview": "../index.html"}
        links.update({key: "../" + suite[field] if suite[field] else None for key, field in _SUITE_PAGE_LINKS})
        page_data = {
            "deck": suite["_deck"],
            "suite": {key: suite[key] for key in _SUITE_PAGE_FIELDS},
            "metadata": details[suite["id"]]["metadata"],
            "links": links,
            "suites": [
                {
                    "name": other["name"],
                    "category": other["category"],
                    "url": PurePosixPath(other["suiteUrl"]).name,
                    "current": other is suite,
                }
                for other in paged
            ],
            "cvs_version": version,
        }
        path = report_root / suite["suiteUrl"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(writer.render_suite_page(page_data), encoding="utf-8")


def _build_suite(variant, decks, report_root, warnings):
    for deck in decks:
        if not discovery.is_valid(deck.data):
            warnings.add(
                INVALID_DECK,
                "%s is unreadable, incomplete, or uses an unsupported schema; listed as unavailable."
                % deck.display_path,
                deck.display_path,
            )
        elif rollup.deck_time(deck.data) is None:
            warnings.add(
                MISSING_FIELD,
                "%s has no usable generated_at; it sorts as the oldest %s run." % (deck.display_path, variant),
                deck.display_path,
            )
    placed = []
    for deck in rollup.order_runs(variant, decks, warnings):
        links = writer.place_run(deck, variant, report_root, warnings)
        for key in ("reportUrl", "viewerUrl"):
            if links[key]:
                writer.add_back_link(report_root / links[key], report_root)
        placed.append((deck, links))

    selected, links = placed[0]
    log.debug("%s: selected %s", variant, selected.display_path)
    for deck, _ in placed[1:]:
        log.debug("%s: superseded %s", variant, deck.display_path)

    data = selected.data if isinstance(selected.data, dict) else {}
    available = discovery.is_valid(selected.data)
    if available:
        result, detail = normalize.normalize(data)
    else:
        result, detail = normalize.unavailable("The Run Deck could not be read or uses an unsupported schema.")
    suite_id = data.get("suite_id") if isinstance(data.get("suite_id"), str) else None
    area = areas.resolve_area(selected.data, selected.stem)
    if area is None:
        warnings.add(
            UNKNOWN_SUITE_ID,
            "No area is known for %s; it is listed under %s." % (suite_id or variant, areas.UNCATEGORIZED),
            selected.display_path,
        )
        area = areas.UNCATEGORIZED
    provenance = data.get("provenance") if isinstance(data.get("provenance"), dict) else {}
    if data.get("cvs_version") and "cvs_version" not in provenance:
        provenance = dict(provenance, cvs_version=data["cvs_version"])

    suite = {
        "id": variant,
        "suite_id": suite_id,
        "name": rollup.display_name(selected.data, selected.stem),
        "category": area,
        "status": result["status"],
        "generated": _generated(selected),
        "cluster": result["cluster"],
        "counts": result["counts"],
        "total": result["total"],
        "unit": result["unit"],
        "partial": result["partial"],
        "notes": result["notes"],
        "failedNodes": result["failed_nodes"],
        "nodes": result["nodes"],
        "scope": result["scope"],
        "detailUrl": "data/details/%s.json" % variant,
        "suiteUrl": "%s/%s.html" % (SUITE_PAGES_DIR, variant) if available else None,
        "jsonUrl": links["jsonUrl"],
        "reportUrl": links["reportUrl"],
        "viewerUrl": links["viewerUrl"],
        "pytestUrl": links["pytestUrl"],
        "logUrl": links["logUrl"],
        "available": available,
        "history": [
            {"generated": _generated(deck), "status": _status(deck), "reportUrl": deck_links["reportUrl"]}
            for deck, deck_links in placed[1:]
        ],
        "_deck": data if available else None,
        "_host_rows": result["host_rows"],
        "_provenance": provenance,
        "_source": selected.display_path,
        "_time": rollup.deck_time(selected.data) if available else None,
    }
    return suite, detail


def _generated(deck):
    value = deck.data.get("generated_at") if isinstance(deck.data, dict) else None
    return value if isinstance(value, str) and value.strip() else "Unavailable"


def _status(deck):
    return normalize.status_of(deck.data.get("overall_status")) if discovery.is_valid(deck.data) else "unknown"
