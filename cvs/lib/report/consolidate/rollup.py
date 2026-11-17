'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Roll suites up into the consolidated report: rerun order, display names, the
collection verdict, recommendations, and the cluster manifest.

Suite dictionaries here are the suite summaries of ``summary.json``. Keys that
start with an underscore are used while building the report and are not written.
'''

import re
from datetime import datetime, timezone

from cvs.lib.report.consolidate.normalize import text
from cvs.lib.report.consolidate.warning_log import TIED_TIMESTAMP

LEGACY_TIME = "%Y-%m-%d %H:%M UTC"
NEEDS_ATTENTION = "needs_attention"
INCOMPLETE_DATA = "incomplete_data"
EVALUATION_INCOMPLETE = "evaluation_incomplete"
ALL_PASSED = "all_passed"
DISPLAY_NAMES = {"ib_perf_bw_test": "IB perf"}
MAX_HOSTS_IN_MESSAGE = 4
_VPC_LABEL = re.compile(r"^(?:\d{1,3}(?:\.\d{1,3}){3}|[0-9A-Fa-f]*:[0-9A-Fa-f:]+)_(?P<host>.+)$")
_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


def deck_time(data):
    '''When a deck was generated, in UTC: ``generated_at_utc``, else ``generated_at`` to the minute, else None.'''
    if not isinstance(data, dict):
        return None
    raw = data.get("generated_at_utc")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    raw = data.get("generated_at")
    if isinstance(raw, str):
        try:
            return datetime.strptime(raw.strip(), LEGACY_TIME).replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def order_runs(variant, decks, warnings):
    '''Return one suite variant's decks, latest first.

    Decks are ordered by generation time, and ties by path relative to the searched
    directory. A deck without a usable timestamp sorts oldest.
    '''
    ordered = sorted(decks, key=lambda deck: (deck_time(deck.data) or _EPOCH, deck.relative_path), reverse=True)
    if len(ordered) > 1:
        latest, previous = deck_time(ordered[0].data), deck_time(ordered[1].data)
        if latest is not None and latest == previous:
            warnings.add(
                TIED_TIMESTAMP,
                "Two %s runs have the same timestamp; the one at the later path, %s, sets the verdict."
                % (variant, ordered[0].relative_path),
                ordered[0].display_path,
            )
    return ordered


def display_name(data, stem):
    '''Suite name from the deck title without "Run Deck", plus the variant when the stem adds one.'''
    if not isinstance(data, dict):
        return stem or "Unknown suite"
    suite_id = data.get("suite_id") if isinstance(data.get("suite_id"), str) else ""
    report = data.get("report") if isinstance(data.get("report"), dict) else {}
    title = report.get("title") if isinstance(report.get("title"), str) else ""
    name = DISPLAY_NAMES.get(suite_id) or re.sub(r"\s+", " ", re.sub(r"\s*\bRun Deck\b", "", title)).strip()
    name = name or suite_id or stem or "Unknown suite"
    if stem and suite_id and stem.startswith(suite_id + "_"):
        name += " \u00b7 " + stem[len(suite_id) + 1 :].replace("_", " ")
    return name


def host_key(label):
    '''Match key for a row label: ``<vpc_ip>_<host>`` matches the plain ``<host>``.'''
    match = _VPC_LABEL.match(label)
    return match.group("host") if match else label


def generated_range(times):
    times = sorted(time for time in times if time is not None)
    if not times:
        return "Unavailable"
    first, last = times[0].strftime(LEGACY_TIME), times[-1].strftime(LEGACY_TIME)
    return first if first == last else "%s \u2013 %s" % (first[: -len(" UTC")], last)


def collection_verdict(suites):
    '''The four-state verdict of the AI Lab CVS dashboard mockup over the included suites.'''
    statuses = [suite["status"] for suite in suites]
    if "fail" in statuses:
        return NEEDS_ATTENTION
    if not suites or any(not s["available"] or s["partial"] or s["status"] == "unknown" for s in suites):
        return INCOMPLETE_DATA
    if any(status in ("record", "na") for status in statuses):
        return EVALUATION_INCOMPLETE
    return ALL_PASSED


def _hosts(suites):
    hosts = {}
    for suite in suites:
        if not suite["_host_rows"]:
            continue
        for node in suite["nodes"]:
            entry = hosts.setdefault(host_key(node), {"suites": [], "failed_in": []})
            if suite["name"] not in entry["suites"]:
                entry["suites"].append(suite["name"])
            if node in suite["failedNodes"] and suite["name"] not in entry["failed_in"]:
                entry["failed_in"].append(suite["name"])
    return hosts


def _recommendation(kind, message, suite=None):
    return {
        "kind": kind,
        "suite": suite["id"] if suite else None,
        "message": message,
        "link": suite["reportUrl"] if suite else None,
    }


def _failure_message(suite):
    if not suite["counts"]["fail"]:
        return "%s reported a failure; open its Run Deck for details." % suite["name"]
    where = ""
    if suite["_host_rows"] and suite["failedNodes"]:
        shown = suite["failedNodes"][:MAX_HOSTS_IN_MESSAGE]
        more = len(suite["failedNodes"]) - len(shown)
        where = " on " + ", ".join(shown) + (" and %d more" % more if more else "")
    return "%s: %d of %d %s failed%s." % (suite["name"], suite["counts"]["fail"], suite["total"], suite["unit"], where)


def recommendations(suites):
    '''Facts from the decks, with no thresholds of their own, in the order a reader should act on them.'''
    recs = []
    repeated = [(host, entry["failed_in"]) for host, entry in _hosts(suites).items() if len(entry["failed_in"]) >= 2]
    for host, names in sorted(repeated, key=lambda item: (-len(item[1]), item[0])):
        recs.append(
            _recommendation("repeat_host", "%s failed in %d suites: %s." % (host, len(names), ", ".join(names)))
        )
    by_name = sorted(suites, key=lambda suite: suite["name"].lower())
    for suite in by_name:
        if suite["status"] == "fail":
            recs.append(_recommendation("failed_suite", _failure_message(suite), suite))
    for suite in by_name:
        if not suite["available"]:
            message = "%s: %s could not be read; rerun the suite or regenerate its Run Deck." % (
                suite["name"],
                suite["_source"],
            )
            recs.append(_recommendation("unavailable", message, suite))
        elif suite["status"] == "record":
            message = "%s has no pass/fail gate; review its measurements." % suite["name"]
            recs.append(_recommendation("record_only", message, suite))
        elif suite["status"] == "na":
            recs.append(_recommendation("not_evaluated", "%s: nothing was evaluated." % suite["name"], suite))
    if not recs:
        recs.append(_recommendation("all_passed", "No action needed for the included suites."))
    return recs


def build_manifest(suites):
    '''Cluster names, hosts in results, and per-suite provenance, built only from what the decks report.

    Only suites whose rows are hosts add hosts; training and inference decks add none.
    Values are listed per suite and never reconciled, because a campaign can mix
    CVS versions and cluster files.
    '''
    clusters = {}
    for suite in suites:
        if suite["cluster"]:
            clusters.setdefault(suite["cluster"], []).append(suite["name"])
    per_suite = []
    for suite in suites:
        provenance = suite["_provenance"]
        per_suite.append(
            {
                "id": suite["id"],
                "name": suite["name"],
                "cluster": suite["cluster"],
                "cvs_version": _field(provenance, "cvs_version"),
                "git_ref": _field(provenance, "git_ref"),
                "image": _field(provenance, "image_display") or _field(provenance, "image_tag"),
                "cluster_file": _field(provenance, "cluster_file"),
                "config_file": _field(provenance, "config_file"),
                "generated": suite["generated"],
            }
        )
    hosts = [dict(host=host, **entry) for host, entry in sorted(_hosts(suites).items())]
    return {"clusters": clusters, "hosts": hosts, "suites": per_suite}


def _field(provenance, key):
    value = provenance.get(key)
    return text(value) if isinstance(value, (str, int, float)) and str(value).strip() else None
