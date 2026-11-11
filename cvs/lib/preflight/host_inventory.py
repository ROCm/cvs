"""
Host hardware inventory gate.

Collects CPU, memory, DIMM, NVMe and network inventory on every node, resolves
each node to a SKU profile, and compares the inventory with that profile. A
threshold left as None is recorded but not asserted. With no profiles at all
the check only records and exports the inventory.
"""

import json
import os
import re

from cvs.lib.preflight.host_readiness_base import (
    CVS_CMD_DENIED,
    HostReadinessCheck,
    findings,
    first_marker,
    parse_markers,
    to_number,
)

PROFILE_FIELDS = (
    'cpu_sockets',
    'min_cpu_cores',
    'dimm_count',
    'dimm_size_gb',
    'nvme_count',
    'nvme_min_size_gb',
    'frontend_nic_count',
    'frontend_nic_interfaces',
    'min_ram_gb',
)

PROBE_COMMAND = r"""
echo "PRODUCT_NAME=$(cat /sys/class/dmi/id/product_name 2>/dev/null)"
LC_ALL=C lscpu 2>/dev/null | sed 's/^/LSCPU=/'
awk '/^MemTotal:/ {print "MEMTOTAL_KB=" $2}' /proc/meminfo 2>/dev/null
if _cvs_dmi=$(sudo -n dmidecode -t 17 2>/dev/null); then
  printf '%s\n' "$_cvs_dmi" | awk '/^Memory Device/ {n++} /^[ \t]+(Size|Locator|Type|Speed):/ {sub(/^[ \t]+/, ""); print "DIMM_FIELD=" n "|" $0}'
else
  echo "CVS_CMD_DENIED dmidecode"
  for _cvs_f in /sys/devices/system/edac/mc/mc*/dimm*/size /sys/devices/system/edac/mc/mc*/rank*/size; do
    [ -r "$_cvs_f" ] && echo "EDAC_DIMM=$_cvs_f $(cat "$_cvs_f")"
  done
fi
lsblk -d -b -n -P -o NAME,SIZE,TRAN,TYPE,MODEL 2>/dev/null | sed 's/^/BLK=/'
for _cvs_n in /sys/class/net/*; do
  [ -e "$_cvs_n/device" ] || continue
  _cvs_ib=$(ls "$_cvs_n/device/infiniband" 2>/dev/null | head -n1)
  _cvs_vf=0; [ -e "$_cvs_n/device/physfn" ] && _cvs_vf=1
  echo "NIC=${_cvs_n##*/}|$(cat "$_cvs_n/operstate" 2>/dev/null)|$(cat "$_cvs_n/speed" 2>/dev/null)|$(basename "$(readlink -f "$_cvs_n/device/driver" 2>/dev/null)")|$_cvs_ib|$_cvs_vf"
done
"""

_SIZE_RE = re.compile(r'^\s*([0-9.]+)\s*(KB|MB|GB|TB|kB)\s*$', re.I)
_BLK_PAIR_RE = re.compile(r'(\w+)="([^"]*)"')
_UNIT_GB = {'KB': 1.0 / (1024 * 1024), 'MB': 1.0 / 1024, 'GB': 1.0, 'TB': 1024.0}


def profile_config_errors(profiles, default_profile):
    """Cross-profile errors that make node-to-profile resolution ambiguous."""
    errors = []
    names = [profile.get('name') for profile in profiles]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        errors.append(f"Duplicate SKU profile name(s): {', '.join(map(str, duplicates))}")
    owners = {}
    for profile in profiles:
        for node in profile.get('nodes') or []:
            owners.setdefault(node, []).append(profile.get('name'))
    for node, owner_names in sorted(owners.items()):
        if len(owner_names) > 1:
            errors.append(f"Node {node} is listed in more than one SKU profile: {', '.join(owner_names)}")
    if default_profile and default_profile not in names:
        errors.append(f"default_profile {default_profile!r} does not name a SKU profile")
    return errors


def resolve_profile(node, product_name, profiles, default_profile):
    """Pick a node's profile: explicit nodes, then dmi_product_name, then default_profile."""
    for profile in profiles:
        if node in (profile.get('nodes') or []):
            return profile
    product = (product_name or '').strip().lower()
    if product:
        for profile in profiles:
            wanted = (profile.get('dmi_product_name') or '').strip().lower()
            if wanted and wanted == product:
                return profile
    for profile in profiles:
        if default_profile and profile.get('name') == default_profile:
            return profile
    return None


def parse_size_gb(text):
    """'64 GB' / '65536 MB' to GB; None for empty slots or unparseable text."""
    match = _SIZE_RE.match(str(text or ''))
    if not match:
        return None
    return round(float(match.group(1)) * _UNIT_GB[match.group(2).upper()], 2)


def parse_lscpu(markers):
    fields = {}
    for line in markers.get('LSCPU') or []:
        key, sep, value = line.partition(':')
        if sep:
            fields[key.strip()] = value.strip()
    sockets = to_number(fields.get('Socket(s)'))
    cores = to_number(fields.get('Core(s) per socket'))
    return {
        'model': fields.get('Model name', ''),
        'sockets': sockets,
        'cores_per_socket': cores,
        'total_cores': sockets * cores if sockets is not None and cores is not None else None,
        'threads_per_core': to_number(fields.get('Thread(s) per core')),
        'logical_cpus': to_number(fields.get('CPU(s)')),
    }


def parse_dimms(markers):
    """Return ``(dimms, source)``; source is dmidecode, edac or unavailable."""
    if markers.get('DIMM_FIELD'):
        devices = {}
        for item in markers['DIMM_FIELD']:
            index, _, field = item.partition('|')
            key, _, value = field.partition(':')
            devices.setdefault(index, {})[key.strip().lower()] = value.strip()
        dimms = []
        for index in sorted(devices, key=lambda value: to_number(value) or 0):
            device = devices[index]
            size = parse_size_gb(device.get('size'))
            if size is None:
                continue
            dimms.append(
                {
                    'locator': device.get('locator', ''),
                    'size_gb': size,
                    'type': device.get('type', ''),
                    'speed': device.get('speed', ''),
                }
            )
        return dimms, 'dmidecode'
    if markers.get('EDAC_DIMM'):
        dimms = []
        for item in markers['EDAC_DIMM']:
            path, _, size_mb = item.rpartition(' ')
            size = to_number(size_mb)
            if size:
                dimms.append({'locator': path, 'size_gb': round(size / 1024.0, 2), 'type': '', 'speed': ''})
        return dimms, 'edac'
    return [], 'unavailable'


def parse_block_devices(markers):
    nvme = []
    for line in markers.get('BLK') or []:
        fields = dict(_BLK_PAIR_RE.findall(line))
        name = fields.get('NAME', '')
        if fields.get('TYPE') != 'disk':
            continue
        if fields.get('TRAN') != 'nvme' and not name.startswith('nvme'):
            continue
        size = to_number(fields.get('SIZE'))
        nvme.append(
            {
                'name': name,
                'size_gb': round(size / 1e9, 1) if size else None,
                'model': fields.get('MODEL', '').strip(),
            }
        )
    return nvme


def parse_nics(markers):
    nics = []
    for line in markers.get('NIC') or []:
        parts = (line.split('|') + [''] * 6)[:6]
        name, operstate, speed, driver, rdma_device, is_vf = parts
        nics.append(
            {
                'name': name,
                'operstate': operstate,
                'speed_mbps': to_number(speed),
                'driver': driver,
                'rdma_device': rdma_device,
                'virtual_function': is_vf == '1',
            }
        )
    return nics


def frontend_nics(nics, backend_rdma_devices):
    """Physical, non-VF netdevs that are not backend RDMA ports.

    With no backend list, every RDMA-capable netdev counts as backend; list the
    frontend interfaces explicitly when the frontend NIC is RDMA-capable too.
    """
    backend = set(backend_rdma_devices or [])
    selected = []
    for nic in nics:
        if nic['virtual_function']:
            continue
        if nic['rdma_device'] and (not backend or nic['rdma_device'] in backend or nic['name'] in backend):
            continue
        selected.append(nic)
    return selected


def _equal(actual, expected):
    return actual is not None and abs(float(actual) - float(expected)) < 0.5


class HostInventoryCheck(HostReadinessCheck):
    """Gate nodes on the hardware inventory their SKU profile expects."""

    def __init__(
        self,
        orch,
        sku_profiles=None,
        default_profile='',
        backend_rdma_devices=None,
        on_mismatch='fail',
        config_dict=None,
    ):
        super().__init__(orch, on_mismatch, config_dict)
        self.sku_profiles = [dict(profile) for profile in sku_profiles or []]
        self.default_profile = default_profile or ''
        self.backend_rdma_devices = list(backend_rdma_devices or [])
        self.config_errors = profile_config_errors(self.sku_profiles, self.default_profile)

    def build_command(self):
        return PROBE_COMMAND

    def collect(self, markers):
        cpu = parse_lscpu(markers)
        dimms, dimm_source = parse_dimms(markers)
        nics = parse_nics(markers)
        frontend = {nic['name'] for nic in frontend_nics(nics, self.backend_rdma_devices)}
        for nic in nics:
            nic['role'] = 'frontend' if nic['name'] in frontend else ('backend' if nic['rdma_device'] else 'other')
        mem_kb = to_number(first_marker(markers, 'MEMTOTAL_KB'))
        return {
            'product_name': first_marker(markers, 'PRODUCT_NAME'),
            'cpu': cpu,
            'memory': {
                'memtotal_gib': round(mem_kb / (1024.0 * 1024.0), 1) if mem_kb else None,
                'dimm_source': dimm_source,
                'dimms': dimms,
            },
            'nvme': parse_block_devices(markers),
            'nics': nics,
        }

    def _cpu_and_memory(self, profile, inventory, mismatches, errors):
        cpu = inventory['cpu']
        if profile.get('cpu_sockets') is not None and not _equal(cpu['sockets'], profile['cpu_sockets']):
            mismatches.append(f"CPU sockets {cpu['sockets']} (expected {profile['cpu_sockets']})")
        min_cores = profile.get('min_cpu_cores')
        if min_cores is not None and (cpu['total_cores'] is None or cpu['total_cores'] < min_cores):
            mismatches.append(f"CPU cores {cpu['total_cores']} (expected at least {min_cores})")
        min_ram = profile.get('min_ram_gb')
        memtotal = inventory['memory']['memtotal_gib']
        if min_ram is not None and (memtotal is None or memtotal < min_ram):
            mismatches.append(f"Usable RAM {memtotal} GiB (MemTotal) is below the required {min_ram} GiB")

        memory = inventory['memory']
        wants_dimms = profile.get('dimm_count') is not None or profile.get('dimm_size_gb') is not None
        if wants_dimms and memory['dimm_source'] == 'unavailable':
            errors.append(
                "DIMM inventory is unreadable: 'sudo -n dmidecode' was denied and EDAC sysfs has no DIMM sizes"
            )
            return
        sizes = [dimm['size_gb'] for dimm in memory['dimms']]
        if profile.get('dimm_count') is not None and len(sizes) != profile['dimm_count']:
            mismatches.append(f"Populated DIMMs {len(sizes)} (expected {profile['dimm_count']})")
        if profile.get('dimm_size_gb') is not None:
            wrong = sorted({size for size in sizes if not _equal(size, profile['dimm_size_gb'])})
            if wrong:
                mismatches.append(
                    f"DIMM sizes {', '.join(f'{s:g} GB' for s in wrong)} (expected {profile['dimm_size_gb']} GB each)"
                )

    def _storage_and_network(self, profile, inventory, mismatches):
        min_size = profile.get('nvme_min_size_gb')
        counted = [
            drive for drive in inventory['nvme'] if min_size is None or (drive['size_gb'] or 0) >= float(min_size)
        ]
        if profile.get('nvme_count') is not None and len(counted) != profile['nvme_count']:
            qualifier = f" of at least {min_size} GB" if min_size is not None else ''
            mismatches.append(f"NVMe drives{qualifier}: {len(counted)} (expected {profile['nvme_count']})")

        nics = inventory['nics']
        listed = profile.get('frontend_nic_interfaces') or []
        if listed:
            by_name = {nic['name']: nic for nic in nics}
            for name in listed:
                nic = by_name.get(name)
                if nic is None:
                    mismatches.append(f"Frontend NIC {name} is missing")
                elif nic['operstate'] != 'up':
                    mismatches.append(f"Frontend NIC {name} is {nic['operstate'] or 'unknown'} (expected up)")
            count = len([name for name in listed if name in by_name])
        else:
            count = len([nic for nic in nics if nic['role'] == 'frontend'])
        if profile.get('frontend_nic_count') is not None and count != profile['frontend_nic_count']:
            mismatches.append(f"Frontend NICs {count} (expected {profile['frontend_nic_count']})")

    def evaluate_node(self, node, output):
        markers = parse_markers(output)
        inventory = self.collect(markers)
        warnings = []
        if any(line.startswith(f'{CVS_CMD_DENIED} dmidecode') for line in str(output).splitlines()):
            warnings.append(
                f"'sudo -n dmidecode' was denied; DIMM inventory read from {inventory['memory']['dimm_source']}"
            )
        if self.config_errors:
            return findings([], self.config_errors, warnings, {'inventory': inventory, 'profile': None})
        if not self.sku_profiles:
            warnings.append('No SKU profiles configured; inventory recorded only')
            return findings([], [], warnings, {'inventory': inventory, 'profile': None})

        profile = resolve_profile(node, inventory['product_name'], self.sku_profiles, self.default_profile)
        if profile is None:
            return findings(
                [],
                [
                    f"No SKU profile matches node {node} (product {inventory['product_name'] or 'unknown'}) and no default_profile is set"
                ],
                warnings,
                {'inventory': inventory, 'profile': None},
            )
        mismatches = []
        errors = []
        self._cpu_and_memory(profile, inventory, mismatches, errors)
        self._storage_and_network(profile, inventory, mismatches)
        return findings(mismatches, errors, warnings, {'inventory': inventory, 'profile': profile.get('name')})


def _safe_name(node):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(node)) or 'node'


def export_inventory(results, out_dir):
    """Write ``<node>.json`` per node and a combined ``host_inventory.json``; return the combined path."""
    os.makedirs(out_dir, exist_ok=True)
    combined = {}
    for node, result in results.items():
        details = result.get('details') or {}
        record = {
            'node': node,
            'status': result.get('status'),
            'profile': details.get('profile'),
            'inventory': details.get('inventory'),
            'errors': result.get('errors') or [],
            'warnings': result.get('warnings') or [],
        }
        combined[node] = record
        with open(os.path.join(out_dir, f"{_safe_name(node)}.json"), 'w', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
    combined_path = os.path.join(out_dir, 'host_inventory.json')
    with open(combined_path, 'w', encoding='utf-8') as handle:
        json.dump(combined, handle, indent=2, sort_keys=True)
    return combined_path
