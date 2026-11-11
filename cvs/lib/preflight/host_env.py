"""
Persistent environment readiness check.

Required variables must be defined in persistent host configuration (by default
``/etc/environment``, which pam_env applies to login and non-interactive SSH
sessions alike), not only exported by a job. Files are parsed statically and
never sourced. ``/etc/profile.d`` scripts reach login shells only, so a variable
defined only there is invisible to MPI or SSH-launched workers.
"""

import re
import shlex

from cvs.lib.preflight.host_readiness_base import HostReadinessCheck, findings, parse_markers

_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_ASSIGN_RE = re.compile(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$')


def validate_variable_names(names):
    bad = [name for name in names if not _NAME_RE.match(str(name))]
    if bad:
        raise ValueError(f"Invalid environment variable name(s): {', '.join(map(repr, bad))}")


def parse_assignment_value(raw):
    """Return ``(value, dynamic)`` for the right-hand side of a static ``KEY=VALUE`` line.

    ``dynamic`` is True when the value expands a variable or command, which a
    static read cannot resolve.
    """
    text = raw.strip()
    if text[:1] == "'":
        end = text.find("'", 1)
        return (text[1:end] if end > 0 else text[1:]), False
    if text[:1] == '"':
        end = text.find('"', 1)
        value = text[1:end] if end > 0 else text[1:]
        return value, ('$' in value or '`' in value)
    value = re.split(r'\s+#', text, maxsplit=1)[0].strip()
    return value, ('$' in value or '`' in value)


def session_command(variables):
    lines = []
    for name in variables:
        lines.append(
            f'if _cvs_v=$(printenv {name}); then printf "SESSION=%s=%s\\n" {name} "$_cvs_v"; '
            f'else echo "SESSION_UNSET={name}"; fi'
        )
    return '\n'.join(lines)


def parse_session(markers):
    values = {}
    for item in markers.get('SESSION') or []:
        name, _, value = item.partition('=')
        values[name] = value
    for name in markers.get('SESSION_UNSET') or []:
        values.setdefault(name.strip(), None)
    return values


class PersistentEnvCheck(HostReadinessCheck):
    """Verify required variables are set persistently and seen by sessions on every node."""

    def __init__(
        self,
        orch,
        required,
        sources,
        check_session=True,
        check_container=False,
        on_mismatch='fail',
        config_dict=None,
    ):
        super().__init__(orch, on_mismatch, config_dict)
        self.required = {str(name): str(value) for name, value in (required or {}).items()}
        validate_variable_names(self.required)
        self.sources = list(sources or [])
        self.check_session = bool(check_session)
        self.check_container = bool(check_container)
        self._container_sessions = {}

    def build_command(self):
        if not self.required:
            return 'true'
        pattern = shlex.quote('^[[:space:]]*(export[[:space:]]+)?(' + '|'.join(self.required) + ')=')
        sources = ' '.join(shlex.quote(src) for src in self.sources)
        script = (
            f'for _cvs_src in {sources}; do\n'
            '  if [ -d "$_cvs_src" ]; then\n'
            '    for _cvs_f in "$_cvs_src"/*.sh; do\n'
            f'      [ -r "$_cvs_f" ] && grep -Hn -E {pattern} "$_cvs_f" | sed "s/^/ENVLINE=/"\n'
            '    done\n'
            '  elif [ -r "$_cvs_src" ]; then\n'
            f'    grep -Hn -E {pattern} "$_cvs_src" | sed "s/^/ENVLINE=/"\n'
            '  else\n'
            '    echo "ENV_SOURCE_MISSING=$_cvs_src"\n'
            '  fi\n'
            'done\n'
        )
        if self.check_session:
            script += session_command(self.required) + '\n'
        return script

    def parse_definitions(self, markers):
        """Return ``{name: [{'path', 'line', 'value', 'dynamic'}]}`` in source and line order."""
        definitions = {}
        for item in markers.get('ENVLINE') or []:
            parts = item.split(':', 2)
            if len(parts) != 3:
                continue
            path, line, content = parts
            match = _ASSIGN_RE.match(content)
            if not match or match.group(1) not in self.required:
                continue
            value, dynamic = parse_assignment_value(match.group(2))
            definitions.setdefault(match.group(1), []).append(
                {'path': path, 'line': line, 'value': value, 'dynamic': dynamic}
            )
        return definitions

    def _variable_findings(self, name, expected, entries, session, mismatches, warnings):
        where = ', '.join(self.sources)
        if not entries:
            mismatches.append(f"{name} is not set persistently in {where}; add {name}={expected} to /etc/environment")
        else:
            static_values = sorted({entry['value'] for entry in entries if not entry['dynamic']})
            if len(static_values) > 1:
                listed = ', '.join(f"{e['path']}:{e['line']}={e['value']}" for e in entries)
                mismatches.append(f"{name} has conflicting persistent definitions: {listed}")
            last = entries[-1]
            if last['dynamic']:
                warnings.append(
                    f"{name} at {last['path']}:{last['line']} is computed ({last['value']}); its value cannot be checked statically"
                )
            elif last['value'] != expected:
                mismatches.append(
                    f"{name}={last['value']} at {last['path']}:{last['line']}; expected {name}={expected}"
                )
        if not self.check_session or name not in session:
            return
        actual = session[name]
        if actual is None:
            if entries:
                warnings.append(
                    f"{name} is defined persistently but not visible in a remote command session; "
                    "check that pam_env applies to SSH sessions"
                )
        elif actual != expected:
            mismatches.append(f"{name}={actual} in a remote command session overrides the expected {name}={expected}")

    def evaluate_node(self, node, output):
        markers = parse_markers(output)
        definitions = self.parse_definitions(markers)
        session = parse_session(markers)
        mismatches = []
        warnings = [f"Environment source {src} does not exist" for src in markers.get('ENV_SOURCE_MISSING') or []]
        for name, expected in self.required.items():
            self._variable_findings(name, expected, definitions.get(name, []), session, mismatches, warnings)

        container = self._container_sessions.get(node)
        if container is not None:
            for name, expected in self.required.items():
                actual = container.get(name)
                if actual != expected:
                    shown = 'unset' if actual is None else f"{name}={actual}"
                    mismatches.append(f"{name} is {shown} inside the CVS container; expected {name}={expected}")

        details = {
            'definitions': definitions,
            'session': session,
            'container_session': container,
            'sources': self.sources,
        }
        return findings(mismatches, [], warnings, details)

    def run(self):
        if self.check_container and self.required:
            out = self.orch.exec(session_command(self.required), timeout=self.probe_timeout)
            self._container_sessions = {node: parse_session(parse_markers(text)) for node, text in out.items()}
        return super().run()
