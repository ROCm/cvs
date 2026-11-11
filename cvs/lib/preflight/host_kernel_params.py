"""
Kernel command-line readiness check.

Requirements are ``key=value`` (or bare ``key``) tokens matched against the
parsed ``/proc/cmdline``. Matching is per parameter, never by substring, so
``amd_iommu=pt`` cannot satisfy ``iommu=pt``.
"""

import shlex

from cvs.lib.preflight.host_readiness_base import HostReadinessCheck, findings, first_marker, parse_markers

INTEL_VENDOR = 'GenuineIntel'
AMD_VENDOR = 'AuthenticAMD'

# The kernel merges every occurrence of these parameters and reads each value as a
# comma-separated option list, so 'pci=bfsort,realloc=off' sets both options.
LIST_PARAMS = ('pci', 'modprobe.blacklist', 'module_blacklist', 'intel_iommu', 'amd_iommu')

_BLACKLIST_PARAMS = ('modprobe.blacklist', 'module_blacklist')

PROBE_COMMAND = r"""
echo "CMDLINE=$(cat /proc/cmdline 2>/dev/null)"
echo "CPU_VENDOR=$(awk -F: '/^vendor_id/ {gsub(/[ \t]/, "", $2); print $2; exit}' /proc/cpuinfo 2>/dev/null)"
echo "NUMA_BALANCING=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null)"
echo "AMDGPU_INITSTATE=$(cat /sys/module/amdgpu/initstate 2>/dev/null)"
"""


def parse_cmdline(text):
    """Split a kernel command line into ordered ``(key, value)`` pairs; value is None for bare flags."""
    try:
        words = shlex.split(str(text or ''))
    except ValueError:
        words = str(text or '').split()
    pairs = []
    for word in words:
        key, sep, value = word.partition('=')
        pairs.append((key, value if sep else None))
    return pairs


def _list_options(pairs, key):
    """Resolve a list parameter into ``{option: value}``; later options win, bare options map to None."""
    options = {}
    for name, value in pairs:
        if name != key or value is None:
            continue
        for item in value.split(','):
            item = item.strip()
            if not item:
                continue
            opt, sep, opt_value = item.partition('=')
            options[opt] = opt_value if sep else None
    return options


def scalar_values(pairs, key):
    return [value for name, value in pairs if name == key]


def requirement_met(pairs, requirement):
    """True when the parsed command line satisfies one ``key[=value]`` requirement."""
    key, sep, expected = requirement.partition('=')
    if key in LIST_PARAMS and sep:
        options = _list_options(pairs, key)
        opt, opt_sep, opt_value = expected.partition('=')
        if opt not in options:
            return False
        return options[opt] == opt_value if opt_sep else True
    values = scalar_values(pairs, key)
    if not values:
        return False
    if not sep:
        return True
    return values[-1] == expected


def describe_found(pairs, requirement):
    key = requirement.partition('=')[0]
    found = [f"{key}={value}" if value is not None else key for name, value in pairs if name == key]
    return ', '.join(found) if found else 'absent'


def blacklisted_modules(pairs):
    modules = set()
    for key in _BLACKLIST_PARAMS:
        modules.update(_list_options(pairs, key))
    return modules


def conflicting_scalars(pairs):
    """Non-list parameters given more than once with different values; the kernel keeps the last."""
    seen = {}
    for key, value in pairs:
        if key in LIST_PARAMS or value is None:
            continue
        seen.setdefault(key, []).append(value)
    return {key: values for key, values in seen.items() if len(set(values)) > 1}


class KernelParamsCheck(HostReadinessCheck):
    """Verify required and forbidden kernel command-line parameters on every node."""

    def __init__(
        self,
        orch,
        required,
        required_intel=None,
        required_amd=None,
        forbidden=None,
        on_mismatch='fail',
        config_dict=None,
    ):
        super().__init__(orch, on_mismatch, config_dict)
        self.required = list(required or [])
        self.required_intel = list(required_intel or [])
        self.required_amd = list(required_amd or [])
        self.forbidden = list(forbidden or [])

    def build_command(self):
        return PROBE_COMMAND

    def _vendor_requirements(self, vendor, warnings):
        if vendor == INTEL_VENDOR:
            return self.required_intel
        if vendor == AMD_VENDOR:
            return self.required_amd
        if self.required_intel or self.required_amd:
            warnings.append(
                f"Unknown CPU vendor {vendor or '(unreadable)'}; vendor-specific kernel parameters not checked"
            )
        return []

    def evaluate_node(self, node, output):
        markers = parse_markers(output)
        if 'CMDLINE' not in markers or not first_marker(markers, 'CMDLINE'):
            return findings(errors=[f"Could not read /proc/cmdline on {node}"])
        cmdline = first_marker(markers, 'CMDLINE')
        vendor = first_marker(markers, 'CPU_VENDOR')
        runtime_numa = first_marker(markers, 'NUMA_BALANCING')
        initstate = first_marker(markers, 'AMDGPU_INITSTATE')
        pairs = parse_cmdline(cmdline)

        mismatches = []
        errors = []
        warnings = []
        required = self.required + self._vendor_requirements(vendor, warnings)
        for requirement in required:
            if not requirement_met(pairs, requirement):
                mismatches.append(
                    f"Kernel parameter {requirement} is not set (found: {describe_found(pairs, requirement)})"
                )
        for token in self.forbidden:
            if requirement_met(pairs, token):
                mismatches.append(f"Forbidden kernel parameter {token} is set")
        for key, values in conflicting_scalars(pairs).items():
            warnings.append(
                f"Kernel parameter {key} is given more than once ({', '.join(values)}); the last one applies"
            )

        if any(r.partition('=')[0] == 'numa_balancing' for r in required) and runtime_numa not in ('', '0'):
            mismatches.append(f"kernel.numa_balancing is {runtime_numa} at runtime; it must be 0")
        if 'amdgpu' in blacklisted_modules(pairs) and initstate != 'live':
            errors.append(
                "amdgpu is blacklisted on the kernel command line but the amdgpu module is not loaded; "
                "load it after boot (for example with a systemd unit that runs 'modprobe amdgpu')"
            )

        details = {
            'cmdline': cmdline,
            'cpu_vendor': vendor,
            'numa_balancing_runtime': runtime_numa,
            'amdgpu_initstate': initstate,
            'checked': required,
        }
        return findings(mismatches, errors, warnings, details)
