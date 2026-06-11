'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Adapter around AMD node-scraper's offline dmesg analyzer.
#
# CVS continues to collect raw dmesg over its existing parallel-SSH layer; this
# module only reuses node-scraper's curated error-pattern table to parse that
# text in memory. No SSH or system connection is required for analysis, so the
# adapter is a drop-in replacement for CVS's hand-maintained regex scanning
# while keeping the existing collection path and the downstream
# {node: [lines]} contract intact.
#
# Best results come from dmesg collected with `dmesg --time-format iso -x`,
# which gives node-scraper ISO timestamps and the decoded facility/level prefix
# its full pattern set expects. Plain `dmesg -T` output still matches message
# bodies, but per-event timestamps will be empty.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)

try:
    from nodescraper.models import SystemInfo
    from nodescraper.plugins.inband.dmesg.analyzer_args import DmesgAnalyzerArgs
    from nodescraper.plugins.inband.dmesg.dmesg_plugin import DmesgPlugin
    from nodescraper.plugins.inband.dmesg.dmesgdata import DmesgData

    NODE_SCRAPER_AVAILABLE = True
    _IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - only hit when dependency is absent
    NODE_SCRAPER_AVAILABLE = False
    _IMPORT_ERROR = exc


DEFAULT_NODE_NAME = "cvs-node"

EVENT_KEYS = (
    "priority",
    "category",
    "description",
    "match_content",
    "count",
    "timestamps",
    "source",
)


def is_available() -> bool:
    """Return True if the amd-node-scraper package is importable."""
    return NODE_SCRAPER_AVAILABLE


def _require_node_scraper() -> None:
    if not NODE_SCRAPER_AVAILABLE:
        raise RuntimeError(
            "amd-node-scraper is not installed; add it from requirements.txt to use "
            f"the node-scraper dmesg adapter. Original import error: {_IMPORT_ERROR}"
        )


def parse_dmesg(
    dmesg_content: str,
    node_name: Optional[str] = None,
    analysis_args: Optional[Union[dict, "DmesgAnalyzerArgs"]] = None,
) -> List[Dict[str, Any]]:
    """Parse raw dmesg text using node-scraper's offline DmesgAnalyzer.

    Args:
        dmesg_content: Raw dmesg log text. Collect with
            `dmesg --time-format iso -x` for full fidelity (ISO timestamps and
            the decoded level prefix); plain `dmesg -T` still matches message
            bodies but without per-event timestamps.
        node_name: Optional system name used to tag the analysis.
        analysis_args: Optional DmesgAnalyzerArgs instance or dict of analyzer
            args, e.g. {"check_unknown_dmesg_errors": False} or
            {"error_regex": [{"regex": "...", "message": "...",
            "event_category": "NETWORK"}]} to extend the built-in pattern set.

    Returns:
        List of normalized event dicts, each with the keys in EVENT_KEYS:
        priority, category, description, match_content, count, timestamps,
        source.
    """
    _require_node_scraper()

    plugin = DmesgPlugin(system_info=SystemInfo(name=node_name or DEFAULT_NODE_NAME))
    result = plugin.analyze(
        data=DmesgData(dmesg_content=dmesg_content or ""),
        analysis_args=analysis_args,
    )
    return [_normalize_event(event) for event in result.events]


def _normalize_event(event: Any) -> Dict[str, Any]:
    """Convert a node-scraper Event into a plain, JSON-friendly dict."""
    data = getattr(event, "data", {}) or {}
    priority = getattr(event, "priority", None)
    return {
        "priority": getattr(priority, "name", str(priority)),
        "category": getattr(event, "category", None),
        "description": getattr(event, "description", None),
        "match_content": data.get("match_content"),
        "count": data.get("count", 1),
        "timestamps": data.get("timestamps", []),
        "source": data.get("source"),
    }


def event_match_lines(events: List[Dict[str, Any]]) -> List[str]:
    """Flatten normalized events into matched-line strings.

    Preserves the legacy `{node: [lines]}` contract used by CVS dmesg scans:
    each detected error becomes a single human-readable line combining the
    error label and the matched text.
    """
    lines: List[str] = []
    for event in events:
        match = event.get("match_content")
        if isinstance(match, (list, tuple)):
            text = " ".join(str(part) for part in match if part)
        elif match is None:
            text = ""
        else:
            text = str(match)
        description = event.get("description") or ""
        line = f"{description}: {text}".strip().rstrip(":").strip()
        if line:
            lines.append(line)
    return lines
