"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Literal, Optional, Pattern as RePattern

import yaml
from pydantic import BaseModel, ConfigDict

from cvs.lib.failure_taxonomy import FailureCategory

DEFAULT_PATTERNS_FILE = Path(__file__).with_name("failure_patterns.yaml")

# Closed source vocabulary -- only streams the spine actually captures. Adding
# a third source is a deliberate edit here + a G6a wiring change, not a
# silent YAML typo (e.g. "dmsg" would otherwise load fine and then never
# match anything when G6a calls scan(text, source="dmesg")).
SourceLiteral = Literal["dmesg", "framework_log"]
# Closed severity vocabulary -- "fatal" flips the verdict in G6a's B4 wiring;
# a typo ("FATAL", "fatel") would silently downgrade a fatal match.
SeverityLiteral = Literal["fatal", "warn"]


class FailurePattern(BaseModel):
    """One row of ``failure_patterns.yaml``.

    Loaded by :class:`FailurePatternScanner` at construction time. ``category``
    must be a value from G1's :class:`FailureCategory` enum -- unknowns raise
    at load (fail closed; see addendum G6b row #3). ``source`` and ``severity``
    are typed Literals so vocabulary typos fail at load, not silently at scan.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    source: SourceLiteral
    pattern: str
    category: FailureCategory
    severity: SeverityLiteral
    hint: str


class PatternHit(BaseModel):
    """One ``(pattern, first matching line)`` hit produced by ``scan``."""

    model_config = ConfigDict(extra="forbid")

    pattern_id: str
    category: FailureCategory
    source: SourceLiteral
    line_no: int
    line: str


class FailurePatternScanner:
    """Loads the seed catalog and matches it against captured log text.

    Pure-function: ``scan`` takes a text blob plus the source label of the
    stream it came from and returns one :class:`PatternHit` per pattern that
    matched (first hit only -- one hit per pattern per stream is enough for the
    verdict G6a wires up). ``source`` mismatches are silently skipped: G6a
    multiplexes the same scanner over multiple streams and that is the normal
    case, not an error (brief: source mismatch is *not* raised).
    """

    def __init__(self, patterns_file: Optional[Path] = None) -> None:
        path = Path(patterns_file) if patterns_file else DEFAULT_PATTERNS_FILE
        raw = yaml.safe_load(path.read_text()) or {}
        entries = raw.get("patterns", []) or []
        patterns: List[FailurePattern] = []
        compiled: List[RePattern[str]] = []
        seen_ids: set[str] = set()
        for entry in entries:
            # Pydantic validates extras + the FailureCategory enum + the
            # source/severity Literals; unknown values raise ValidationError,
            # which is a ValueError subclass -- matches the brief's
            # ValueError contract.
            pat = FailurePattern(**entry)
            if pat.id in seen_ids:
                raise ValueError(f"duplicate pattern id in catalog: {pat.id!r}")
            seen_ids.add(pat.id)
            # Re-raise re.error as ValueError so all three load-time failure
            # modes (dup id, unknown category/source/severity, bad regex)
            # share one exception type per the brief's fail-closed contract.
            try:
                compiled.append(re.compile(pat.pattern, re.IGNORECASE))
            except re.error as exc:
                raise ValueError(f"invalid regex for pattern {pat.id!r}: {exc}") from exc
            patterns.append(pat)
        self.patterns: List[FailurePattern] = patterns
        self._compiled: List[RePattern[str]] = compiled

    def scan(self, text: str, source: Optional[str] = None) -> List[PatternHit]:
        """Return one hit per pattern whose regex first matches a line of ``text``.

        If ``source`` is given, patterns declaring a different ``source`` are
        skipped silently (multiplex case). One hit per pattern per call -- we
        stop at the first matching line.
        """
        hits: List[PatternHit] = []
        if not text:
            return hits
        lines = text.splitlines()
        for pat, regex in zip(self.patterns, self._compiled):
            if source is not None and pat.source != source:
                continue
            for idx, line in enumerate(lines, start=1):
                if regex.search(line):
                    hits.append(
                        PatternHit(
                            pattern_id=pat.id,
                            category=pat.category,
                            source=pat.source,
                            line_no=idx,
                            line=line.strip()[:500],
                        )
                    )
                    break
        return hits
