"""
Shared plumbing for the host readiness preflight checks.

Each check runs one remote probe per node, parses marker lines, and records a
flat per-node result ``{node: {status, errors, warnings, details}}`` so the
preflight report and Run Deck treat it like the other per-node checks.
"""

import re
from abc import abstractmethod

from cvs.lib.preflight.base import PreflightCheck

CVS_CMD_DENIED = 'CVS_CMD_DENIED'
ON_MISMATCH_FAIL = 'fail'
ON_MISMATCH_WARN = 'warn'
ON_MISMATCH_CHOICES = (ON_MISMATCH_FAIL, ON_MISMATCH_WARN)

_UNREACHABLE_TOKENS = ('ABORT: Host Unreachable', 'Host Unreachable Error')
_MARKER_RE = re.compile(r'^([A-Z][A-Z0-9_]*)=(.*)$')


def parse_markers(output):
    """Return ``{KEY: [value, ...]}`` for every ``KEY=value`` line, in output order."""
    markers = {}
    for raw_line in str(output or '').splitlines():
        match = _MARKER_RE.match(raw_line.rstrip('\r'))
        if match:
            markers.setdefault(match.group(1), []).append(match.group(2))
    return markers


def first_marker(markers, key, default=''):
    values = markers.get(key) or []
    return values[0].strip() if values else default


def to_number(value):
    """Parse an int or float from a probe value; None when it is not numeric."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return None


def findings(mismatches=None, errors=None, warnings=None, details=None):
    """Bundle one node's findings.

    ``mismatches`` are configuration gaps that ``on_mismatch: warn`` downgrades.
    ``errors`` mean the check could not read the node and always fail it.
    """
    return {
        'mismatches': list(mismatches or []),
        'errors': list(errors or []),
        'warnings': list(warnings or []),
        'details': dict(details or {}),
    }


def node_result(found, on_mismatch=ON_MISMATCH_FAIL):
    errors = list(found['errors'])
    warnings = list(found['warnings'])
    if on_mismatch == ON_MISMATCH_WARN:
        warnings = list(found['mismatches']) + warnings
    else:
        errors = list(found['mismatches']) + errors
    return {
        'status': 'FAIL' if errors else 'PASS',
        'errors': errors,
        'warnings': warnings,
        'details': found['details'],
    }


def is_unreachable(output):
    text = str(output or '')
    return any(token in text for token in _UNREACHABLE_TOKENS)


class HostReadinessCheck(PreflightCheck):
    """One probe per node, one verdict per node. Subclasses build and evaluate the probe."""

    probe_timeout = 120

    def __init__(self, orch, on_mismatch=ON_MISMATCH_FAIL, config_dict=None):
        super().__init__(orch, config_dict)
        if on_mismatch not in ON_MISMATCH_CHOICES:
            raise ValueError(f"on_mismatch must be one of {', '.join(ON_MISMATCH_CHOICES)}, got {on_mismatch!r}")
        self.on_mismatch = on_mismatch

    @abstractmethod
    def build_command(self):
        """Return the shell probe run once on every node."""

    @abstractmethod
    def evaluate_node(self, node, output):
        """Return :func:`findings` for one node's probe output."""

    def evaluate(self, node, output):
        if is_unreachable(output):
            return node_result(findings(errors=[f"Node {node} is unreachable: {str(output).strip()}"]))
        return node_result(self.evaluate_node(node, output), self.on_mismatch)

    def run(self):
        out_dict = self.orch.all.exec(self.build_command(), timeout=self.probe_timeout)
        self.results = {node: self.evaluate(node, output) for node, output in out_dict.items()}
        return self.results
