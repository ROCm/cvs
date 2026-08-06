"""
/etc/security/limits.conf Checking Module

Ports the ``ansible/readlimits/readlimits.yml`` playbook. Validates that a
configurable set of required lines are present verbatim in
``/etc/security/limits.conf`` on every reachable node. This is a blocking
(FAIL) check when enabled, matching the original playbook's cluster-wide
``fail:`` gate.
"""

from cvs.lib.preflight.base import PreflightCheck


class LimitsConfCheck(PreflightCheck):
    """Validate required lines are present in /etc/security/limits.conf."""

    def __init__(self, phdl, required_lines, config_dict=None):
        """
        Initialize the limits.conf check.

        Args:
            phdl: Parallel SSH handle for cluster nodes.
            required_lines: List of exact lines that must appear (in any order,
                any amount of surrounding whitespace) in ``/etc/security/limits.conf``.
            config_dict: Optional configuration dictionary.
        """
        super().__init__(phdl, config_dict)
        self.required_lines = list(required_lines or [])

    @staticmethod
    def _normalize(line):
        return ' '.join(line.split())

    def run(self):
        """
        Execute the limits.conf check on every reachable node.

        Returns:
            dict: Per-node results ``{node: {status, missing_lines, errors}}``.
        """
        self.results = {}
        out_dict = self.phdl.exec("cat /etc/security/limits.conf 2>/dev/null || echo ''")

        normalized_required = [self._normalize(line) for line in self.required_lines]

        for node, output in out_dict.items():
            present_lines = {self._normalize(raw_line) for raw_line in (output or '').splitlines()}
            missing_lines = [
                original
                for original, normalized in zip(self.required_lines, normalized_required)
                if normalized not in present_lines
            ]

            self.results[node] = {
                'status': 'PASS' if not missing_lines else 'FAIL',
                'missing_lines': missing_lines,
                'errors': (
                    [f"/etc/security/limits.conf missing {len(missing_lines)} required line(s)"]
                    if missing_lines
                    else []
                ),
            }
        return self.results
