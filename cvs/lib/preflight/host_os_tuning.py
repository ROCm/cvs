"""
OS tuning readiness check: CPU idle states (C-states) and optional CPU governor.

Idle states are read from sysfs rather than the ``cpupower`` binary, which is
often missing or built for a different kernel. ``cpupower idle-set -d N`` writes
the same ``disable`` files this check reads.
"""

from cvs.lib.preflight.host_kernel_params import describe_found, parse_cmdline, requirement_met
from cvs.lib.preflight.host_readiness_base import HostReadinessCheck, findings, first_marker, parse_markers

# Counts are aggregated on the node so a many-core host returns a handful of lines.
PROBE_COMMAND = r"""
echo "CMDLINE=$(cat /proc/cmdline 2>/dev/null)"
echo "CPUIDLE_DRIVER=$(cat /sys/devices/system/cpu/cpuidle/current_driver 2>/dev/null)"
grep -H . /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state[0-9]*/disable 2>/dev/null \
  | awk -F: '{n = split($1, p, "/"); s = p[n-1]; sub("state", "", s); total[s]++; if ($2 == "1") off[s]++}
             END {for (s in total) printf "IDLE_STATE=%s %d %d\n", s, total[s], off[s] + 0}'
grep -H . /sys/devices/system/cpu/cpu0/cpuidle/state[0-9]*/name 2>/dev/null \
  | awk -F: '{n = split($1, p, "/"); s = p[n-1]; sub("state", "", s); printf "IDLE_NAME=%s %s\n", s, $2}'
cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor 2>/dev/null \
  | sort | uniq -c | awk '{printf "GOVERNOR=%s %s\n", $2, $1}'
"""


def parse_idle_states(markers):
    """Return ``{index: {'name', 'cpus', 'disabled'}}`` from IDLE_STATE / IDLE_NAME markers."""
    names = {}
    for value in markers.get('IDLE_NAME') or []:
        index, _, name = value.strip().partition(' ')
        if index.isdigit():
            names[int(index)] = name.strip()
    states = {}
    for value in markers.get('IDLE_STATE') or []:
        parts = value.split()
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            continue
        index, cpus, disabled = (int(part) for part in parts)
        states[index] = {'name': names.get(index, f"state{index}"), 'cpus': cpus, 'disabled': disabled}
    return states


def parse_governors(markers):
    governors = {}
    for value in markers.get('GOVERNOR') or []:
        name, _, count = value.strip().partition(' ')
        if name and count.strip().isdigit():
            governors[name] = int(count)
    return governors


class OsTuningCheck(HostReadinessCheck):
    """Verify deep C-states are disabled on every online CPU, plus optional cmdline and governor settings."""

    def __init__(
        self,
        orch,
        disable_idle_states_from=2,
        max_cstate_params=None,
        cpu_governor=None,
        on_mismatch='fail',
        config_dict=None,
    ):
        super().__init__(orch, on_mismatch, config_dict)
        self.disable_idle_states_from = disable_idle_states_from
        self.max_cstate_params = list(max_cstate_params or [])
        self.cpu_governor = cpu_governor

    def build_command(self):
        return PROBE_COMMAND

    def _idle_state_mismatches(self, states, driver, notes):
        if self.disable_idle_states_from is None:
            return []
        if not states:
            notes.append(
                f"cpuidle is not active (driver: {driver or 'none'}); no idle states can be entered, so none need disabling"
            )
            return []
        mismatches = []
        for index in sorted(states):
            state = states[index]
            if index < self.disable_idle_states_from:
                continue
            enabled = state['cpus'] - state['disabled']
            if enabled:
                mismatches.append(
                    f"Idle state {state['name']} (state{index}) is enabled on {enabled}/{state['cpus']} CPUs; "
                    f"disable it, for example with 'cpupower idle-set -d {index}'"
                )
        return mismatches

    def _governor_mismatches(self, governors):
        if not self.cpu_governor:
            return []
        if not governors:
            return [f"CPU frequency governor is not readable; expected {self.cpu_governor}"]
        other = {name: count for name, count in governors.items() if name != self.cpu_governor}
        if not other:
            return []
        found = ', '.join(f"{name} on {count} CPUs" for name, count in sorted(other.items()))
        return [f"CPU frequency governor is not {self.cpu_governor} everywhere: {found}"]

    def evaluate_node(self, node, output):
        markers = parse_markers(output)
        cmdline = first_marker(markers, 'CMDLINE')
        driver = first_marker(markers, 'CPUIDLE_DRIVER')
        states = parse_idle_states(markers)
        governors = parse_governors(markers)
        notes = []

        mismatches = self._idle_state_mismatches(states, driver, notes)
        errors = []
        if self.max_cstate_params:
            if not cmdline:
                errors.append(f"Could not read /proc/cmdline on {node}")
            else:
                pairs = parse_cmdline(cmdline)
                for requirement in self.max_cstate_params:
                    if not requirement_met(pairs, requirement):
                        mismatches.append(
                            f"Kernel parameter {requirement} is not set (found: {describe_found(pairs, requirement)})"
                        )
        mismatches.extend(self._governor_mismatches(governors))

        details = {
            'cpuidle_driver': driver,
            'idle_states': {f"state{index}": state for index, state in sorted(states.items())},
            'governors': governors,
            'notes': notes,
        }
        return findings(mismatches, errors, [], details)
