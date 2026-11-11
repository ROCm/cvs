"""
Platform readiness extras carried over from ROCm Deployment Health Check (RDHC).

Each sub-check is opt-in by configuration: Large BAR, PCIe AtomicOps on GPUs,
amdgpu cwsr_enable, persistent memlock/nofile limits, RDMA kernel modules, and
a ``ldd`` scan of the ROCm libraries for unresolved dependencies.
"""

import shlex

from cvs.lib.preflight.host_readiness_base import (
    CVS_CMD_DENIED,
    HostReadinessCheck,
    findings,
    first_marker,
    parse_markers,
    to_number,
)

DEFAULT_RDMA_MODULES = ('ib_core', 'ib_uverbs', 'rdma_ucm', 'rdma_cm')
_UNLIMITED = ('unlimited', 'infinity', '-1')
_LDD_REPORT_CAP = 200

LARGE_BAR_PROBE = r"""
for _cvs_d in /sys/class/drm/card[0-9]*; do
  case "${_cvs_d##*/}" in *-*) continue ;; esac
  [ -r "$_cvs_d/device/mem_info_vram_total" ] || continue
  echo "BAR=${_cvs_d##*/} $(cat "$_cvs_d/device/mem_info_vram_total" 2>/dev/null) $(cat "$_cvs_d/device/mem_info_vis_vram_total" 2>/dev/null)"
done
"""

# AMD display (0x03xx) and accelerator (0x12xx) functions only; AMD bridges and USB controllers are skipped.
ATOMIC_PROBE = r"""
for _cvs_d in /sys/bus/pci/devices/*; do
  [ "$(cat "$_cvs_d/vendor" 2>/dev/null)" = 0x1002 ] || continue
  case "$(cat "$_cvs_d/class" 2>/dev/null)" in 0x03*|0x12*) ;; *) continue ;; esac
  _cvs_b=${_cvs_d##*/}
  if _cvs_out=$(sudo -n lspci -vvv -s "$_cvs_b" 2>/dev/null); then
    echo "ATOMIC=$_cvs_b $(printf '%s\n' "$_cvs_out" | grep -m1 'AtomicOpsCtl:' | sed 's/^[[:space:]]*//')"
  else
    echo "CVS_CMD_DENIED lspci $_cvs_b"
  fi
done
"""

CWSR_PROBE = 'echo "CWSR=$(cat /sys/module/amdgpu/parameters/cwsr_enable 2>/dev/null)"\n'

LIMITS_PROBE = r"""
echo "LIMITS_USER=$(id -un)"
echo "LIMITS_GROUPS=$(id -Gn)"
for _cvs_f in /etc/security/limits.conf /etc/security/limits.d/*.conf; do
  [ -r "$_cvs_f" ] && grep -Hn -E '^[[:space:]]*[^#[:space:]]+[[:space:]]+(soft|hard|-)[[:space:]]+(memlock|nofile)[[:space:]]' "$_cvs_f" | sed 's/^/LIMIT=/'
done
"""

RDMA_PROBE_TEMPLATE = r"""
for _cvs_m in {modules}; do [ -d "/sys/module/$_cvs_m" ] && echo "MODULE=$_cvs_m"; done
for _cvs_d in /sys/class/infiniband/*; do
  [ -e "$_cvs_d" ] || continue
  echo "IBDEV=${{_cvs_d##*/}} $(basename "$(readlink -f "$_cvs_d/device/driver" 2>/dev/null)")"
done
"""

LDD_PROBE_TEMPLATE = r"""
_cvs_lib={lib_dir}
if [ ! -d "$_cvs_lib" ]; then echo "ROCM_LIB_MISSING=$_cvs_lib"; else
  _cvs_to=""; command -v timeout >/dev/null 2>&1 && _cvs_to="timeout {timeout}"
  echo "LDD_CHECKED=$(find "$_cvs_lib" -maxdepth {depth} -name '*.so*' -type f 2>/dev/null | wc -l)"
  find "$_cvs_lib" -maxdepth {depth} -name '*.so*' -type f -print0 2>/dev/null \
    | $_cvs_to xargs -0 -r -n 50 ldd 2>/dev/null \
    | awk '/^\// {{lib = $1; sub(/:$/, "", lib)}} /not found/ {{print "LDD_MISSING=" lib " " $1}}' \
    | sort -u | head -n {cap}
fi
"""


def parse_limit_entries(markers):
    """Return limit entries in file order as dicts: path, line, domain, type, item, value."""
    entries = []
    for item in markers.get('LIMIT') or []:
        parts = item.split(':', 2)
        if len(parts) != 3:
            continue
        path, line, content = parts
        words = content.split('#', 1)[0].split()
        if len(words) < 4:
            continue
        entries.append(
            {'path': path, 'line': line, 'domain': words[0], 'type': words[1], 'item': words[2], 'value': words[3]}
        )
    return entries


def _domain_rank(domain, user, groups):
    """pam_limits precedence: a user entry beats a group entry beats the wildcard; root ignores both."""
    if domain == user:
        return 3
    if user == 'root':
        return None
    if domain[:1] in ('@', '%') and domain[1:] in groups:
        return 2
    if domain == '*':
        return 1
    return None


def effective_limits(entries, user, groups):
    """Return ``{(item, 'soft'|'hard'): entry}`` that pam_limits would apply to ``user``."""
    chosen = {}
    for entry in entries:
        rank = _domain_rank(entry['domain'], user, groups)
        if rank is None:
            continue
        kinds = ('soft', 'hard') if entry['type'] == '-' else (entry['type'],)
        for kind in kinds:
            key = (entry['item'], kind)
            if key not in chosen or rank >= chosen[key][0]:
                chosen[key] = (rank, entry)
    return {key: entry for key, (_, entry) in chosen.items()}


def _limit_ok(value, expected):
    if str(expected).lower() in _UNLIMITED:
        return str(value).lower() in _UNLIMITED
    if str(value).lower() in _UNLIMITED:
        return True
    number = to_number(value)
    return number is not None and number >= to_number(expected)


class PlatformExtrasCheck(HostReadinessCheck):
    """Run the enabled RDHC-derived platform checks in a single probe per node."""

    def __init__(
        self,
        orch,
        large_bar=True,
        atomic_ops=True,
        cwsr_enable=None,
        limits=None,
        rdma_modules=None,
        rocm_lib_deps=None,
        rocm_path='/opt/rocm',
        on_mismatch='fail',
        config_dict=None,
    ):
        super().__init__(orch, on_mismatch, config_dict)
        self.large_bar = bool(large_bar)
        self.atomic_ops = bool(atomic_ops)
        self.cwsr_enable = cwsr_enable
        self.limits = dict(limits) if limits else None
        self.rdma_modules = list(rdma_modules) if rdma_modules else []
        self.rocm_lib_deps = dict(rocm_lib_deps) if rocm_lib_deps else None
        self.rocm_path = rocm_path or '/opt/rocm'
        if self.rocm_lib_deps:
            self.probe_timeout = 120 + int(self.rocm_lib_deps.get('timeout_sec', 300))

    def build_command(self):
        parts = []
        if self.large_bar:
            parts.append(LARGE_BAR_PROBE)
        if self.atomic_ops:
            parts.append(ATOMIC_PROBE)
        parts.append(CWSR_PROBE)
        if self.limits:
            parts.append(LIMITS_PROBE)
        if self.rdma_modules:
            parts.append(RDMA_PROBE_TEMPLATE.format(modules=' '.join(shlex.quote(m) for m in self.rdma_modules)))
        if self.rocm_lib_deps:
            parts.append(
                LDD_PROBE_TEMPLATE.format(
                    lib_dir=shlex.quote(f"{self.rocm_path.rstrip('/')}/lib"),
                    depth=int(self.rocm_lib_deps.get('max_depth', 1)),
                    timeout=int(self.rocm_lib_deps.get('timeout_sec', 300)),
                    cap=_LDD_REPORT_CAP,
                )
            )
        return '\n'.join(parts)

    def _large_bar(self, markers, mismatches, warnings, details):
        cards = {}
        for item in markers.get('BAR') or []:
            parts = item.split()
            if len(parts) != 3:
                continue
            card, total, visible = parts[0], to_number(parts[1]), to_number(parts[2])
            cards[card] = {'vram_total': total, 'vis_vram_total': visible}
            if total is None or visible is None:
                warnings.append(f"{card}: VRAM size is unreadable; Large BAR not checked")
            elif total != visible:
                mismatches.append(
                    f"{card}: Large BAR is not enabled (CPU-visible VRAM {visible // (1024 * 1024)} MiB "
                    f"of {total // (1024 * 1024)} MiB); enable Resizable BAR / Above 4G decoding in BIOS"
                )
        if not cards:
            warnings.append('No amdgpu VRAM sizes in sysfs; Large BAR not checked')
        details['large_bar'] = cards

    def _atomic_ops(self, markers, output, mismatches, errors, details):
        devices = {}
        for item in markers.get('ATOMIC') or []:
            bdf, _, ctl = item.partition(' ')
            devices[bdf] = ctl.strip()
            if not ctl.strip():
                mismatches.append(f"GPU {bdf}: AtomicOpsCtl is not reported by lspci")
            elif 'ReqEn+' not in ctl:
                mismatches.append(f"GPU {bdf}: PCIe AtomicOps requests are not enabled ({ctl.strip()})")
        denied = [line.split()[-1] for line in str(output).splitlines() if line.startswith(f'{CVS_CMD_DENIED} lspci')]
        if denied:
            errors.append(
                f"'sudo -n lspci -vvv' was denied for {len(denied)} GPU function(s); grant passwordless lspci "
                "or disable platform_extras.atomic_ops"
            )
        details['atomic_ops'] = devices

    def _limits(self, markers, mismatches, details):
        user = first_marker(markers, 'LIMITS_USER')
        groups = set(first_marker(markers, 'LIMITS_GROUPS').split())
        effective = effective_limits(parse_limit_entries(markers), user, groups)
        wanted = {'memlock': self.limits.get('memlock'), 'nofile': self.limits.get('nofile_min')}
        recorded = {}
        for item, expected in wanted.items():
            if expected in (None, ''):
                continue
            for kind in ('soft', 'hard'):
                entry = effective.get((item, kind))
                recorded[f"{kind}_{item}"] = entry['value'] if entry else None
                if entry is None:
                    mismatches.append(
                        f"No {kind} {item} limit applies to user {user or 'unknown'} in limits.conf or limits.d "
                        f"(expected {expected})"
                    )
                elif not _limit_ok(entry['value'], expected):
                    mismatches.append(
                        f"{kind} {item} is {entry['value']} at {entry['path']}:{entry['line']} (expected {expected})"
                    )
        details['limits'] = {'user': user, 'values': recorded}

    def _rdma(self, markers, mismatches, details):
        loaded = {value.strip() for value in markers.get('MODULE') or []}
        missing = [module for module in self.rdma_modules if module not in loaded]
        if missing:
            mismatches.append(f"RDMA kernel module(s) not loaded: {', '.join(missing)}")
        devices = {}
        for item in markers.get('IBDEV') or []:
            name, _, driver = item.partition(' ')
            devices[name] = driver.strip()
        if not devices:
            mismatches.append('No RDMA devices are registered in /sys/class/infiniband')
        details['rdma'] = {'modules_loaded': sorted(loaded), 'devices': devices}

    def _ldd(self, markers, mismatches, errors, details):
        missing_dir = first_marker(markers, 'ROCM_LIB_MISSING')
        if missing_dir:
            errors.append(f"ROCm library directory {missing_dir} does not exist")
            return
        unresolved = markers.get('LDD_MISSING') or []
        if unresolved:
            shown = '; '.join(unresolved[:5])
            more = f" (+{len(unresolved) - 5} more)" if len(unresolved) > 5 else ''
            mismatches.append(f"{len(unresolved)} unresolved ROCm library dependencies: {shown}{more}")
        details['rocm_lib_deps'] = {'checked': to_number(first_marker(markers, 'LDD_CHECKED')), 'missing': unresolved}

    def evaluate_node(self, node, output):
        markers = parse_markers(output)
        mismatches = []
        errors = []
        warnings = []
        details = {}
        if self.large_bar:
            self._large_bar(markers, mismatches, warnings, details)
        if self.atomic_ops:
            self._atomic_ops(markers, output, mismatches, errors, details)
        cwsr = first_marker(markers, 'CWSR')
        details['cwsr_enable'] = cwsr
        if self.cwsr_enable is not None and cwsr != str(self.cwsr_enable):
            mismatches.append(f"amdgpu cwsr_enable is {cwsr or 'unreadable'} (expected {self.cwsr_enable})")
        if self.limits:
            self._limits(markers, mismatches, details)
        if self.rdma_modules:
            self._rdma(markers, mismatches, details)
        if self.rocm_lib_deps:
            self._ldd(markers, mismatches, errors, details)
        return findings(mismatches, errors, warnings, details)
