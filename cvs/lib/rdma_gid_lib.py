"""Shared sysfs GID probing and validation for RDMA suites."""

import shlex

SYSFS_IB_ROOT = '/sys/class/infiniband'
DEFAULT_GID_TYPE = 'RoCE v2'
GID_TYPE_ANY = 'any'
ZERO_GID = '0000:0000:0000:0000:0000:0000:0000:0000'
IPV4_MAPPED_GID_PREFIX = '0000:0000:0000:0000:0000:ffff:'
LINK_LOCAL_GID_PREFIX = 'fe80:'

GID_OK = 'OK'
GID_EMPTY = 'EMPTY'
GID_MISSING = 'MISSING'
GID_DEVICE_MISSING = 'DEVICE_MISSING'
GID_WRONG_TYPE = 'WRONG_TYPE'
GID_NOT_REPORTED = 'NOT_REPORTED'


def normalize_gid_index(gid_index):
    """Return a safe, non-negative sysfs GID index."""
    index = str(gid_index).strip()
    if not index.isascii() or not index.isdigit():
        raise ValueError(f"gid_index must be a non-negative integer, got {gid_index!r}")
    return index


def is_auto_gid_index(gid_index):
    """Whether an index requests automatic selection."""
    return gid_index is None or str(gid_index).strip().lower() in ('', 'auto')


def gid_type_matches(actual, expected):
    """Compare sysfs GID types, including the configured escape hatch."""
    wanted = ' '.join((expected or '').split()).lower()
    return wanted in ('', GID_TYPE_ANY) or ' '.join((actual or '').split()).lower() == wanted


def gid_scope(gid):
    """Classify a GID address by its prefix."""
    value = (gid or '').lower()
    if value.startswith(IPV4_MAPPED_GID_PREFIX):
        return 'ipv4-mapped'
    if value.startswith(LINK_LOCAL_GID_PREFIX):
        return 'link-local'
    return 'global'


def build_gid_probe_cmd(gid_index, devices=None, port=1):
    """Build a shell command to read one GID and its type per device."""
    index = normalize_gid_index(gid_index)
    targets = (
        ' '.join(shlex.quote(f'{SYSFS_IB_ROOT}/{device}') for device in devices)
        if devices is not None
        else f'{SYSFS_IB_ROOT}/*'
    )
    skip_missing = '[ -d "$dev" ] || continue;' if devices is None else ''
    return f'''
for dev in {targets}; do
    {skip_missing}
    dev_name=$(basename "$dev"); echo "DEVICE:$dev_name"
    if [ ! -d "$dev" ]; then echo "DEVICE_MISSING"; continue; fi
    echo "LINK_LAYER:$(cat "$dev/ports/{port}/link_layer" 2>/dev/null)"
    if [ -f "$dev/ports/{port}/gids/{index}" ]; then
        echo "GID:$(cat "$dev/ports/{port}/gids/{index}" 2>/dev/null)"
        echo "GID_TYPE:$(cat "$dev/ports/{port}/gid_attrs/types/{index}" 2>/dev/null)"
    else
        echo "GID_MISSING"
    fi
done
'''


def parse_gid_probe_output(output):
    """Parse tagged per-device GID probe output."""
    entries = {}
    current = None
    for line in (output or '').splitlines():
        line = line.strip()
        if line.startswith('DEVICE:'):
            current = line.partition(':')[2].strip()
            entries[current] = {'present': True, 'link_layer': '', 'gid': None, 'gid_type': ''}
        elif current is not None:
            entry = entries[current]
            if line == 'DEVICE_MISSING':
                entry['present'] = False
            elif line == 'GID_MISSING':
                entry['gid'] = None
            elif line.startswith('LINK_LAYER:'):
                entry['link_layer'] = line.partition(':')[2].strip()
            elif line.startswith('GID:'):
                entry['gid'] = line.partition(':')[2].strip()
            elif line.startswith('GID_TYPE:'):
                entry['gid_type'] = line.partition(':')[2].strip()
    return entries


def check_gid_entry(device, entry, gid_index, expected_gid_type=DEFAULT_GID_TYPE):
    """Check presence, value, and type of one device's GID."""
    index = str(gid_index)
    if entry is None:
        return GID_NOT_REPORTED, f'RDMA device {device} was not reported by the GID probe'
    if not entry['present']:
        return GID_DEVICE_MISSING, f'RDMA device {device} not found under {SYSFS_IB_ROOT}'
    gid = entry['gid']
    if gid is None:
        return GID_MISSING, f'GID index {index} missing on {device}: no sysfs GID entry'
    if not gid or gid.lower() == ZERO_GID:
        return GID_EMPTY, f"GID index {index} is empty on {device} (value '{gid}')"
    if entry['link_layer'].lower() == 'infiniband':
        return GID_OK, ''
    if not gid_type_matches(entry['gid_type'], expected_gid_type):
        return GID_WRONG_TYPE, (
            f"GID index {index} on {device} has type '{entry['gid_type'] or 'unknown'}', "
            f"expected '{expected_gid_type}' (GID {gid})"
        )
    return GID_OK, ''


def build_gid_table_cmd(devices=None, port=1):
    """Build a shell command to dump populated GID table entries."""
    targets = (
        ' '.join(shlex.quote(f'{SYSFS_IB_ROOT}/{device}') for device in devices)
        if devices is not None
        else f'{SYSFS_IB_ROOT}/*'
    )
    return f'''
for dev in {targets}; do
    [ -d "$dev" ] || continue
    dev_name=$(basename "$dev")
    for entry in "$dev"/ports/{port}/gids/*; do
        [ -f "$entry" ] || continue
        idx=$(basename "$entry")
        gid=$(cat "$entry" 2>/dev/null)
        [ -n "$gid" ] && [ "$gid" != "{ZERO_GID}" ] || continue
        gid_type=$(cat "$dev/ports/{port}/gid_attrs/types/$idx" 2>/dev/null)
        echo "GIDENT|$dev_name|$idx|$gid|$gid_type"
    done
done
'''


def parse_gid_table_output(output):
    """Parse populated GID table entries from tagged output."""
    tables = {}
    for line in (output or '').splitlines():
        parts = line.strip().split('|')
        if (
            len(parts) != 5
            or parts[0] != 'GIDENT'
            or not parts[1]
            or not parts[2].isascii()
            or not parts[2].isdigit()
            or not parts[3]
        ):
            continue
        tables.setdefault(parts[1], {})[int(parts[2])] = {'gid': parts[3].strip(), 'gid_type': parts[4].strip()}
    return tables


def select_common_gid_index(tables_by_node, node_dev_dict, expected_gid_type=DEFAULT_GID_TYPE):
    """Find the lowest IPv4-mapped GID index usable on every selected NIC."""
    common = None
    errors = []
    candidates_by_nic = {}
    for node, devices in node_dev_dict.items():
        for device in devices:
            table = tables_by_node.get(node, {}).get(device, {})
            candidates = {
                index
                for index, entry in table.items()
                if gid_type_matches(entry['gid_type'], expected_gid_type) and gid_scope(entry['gid']) == 'ipv4-mapped'
            }
            candidates_by_nic[f'{node}/{device}'] = candidates
            if not candidates:
                summary = (
                    ', '.join(
                        f"idx={index} type={entry['gid_type']} gid={entry['gid']}"
                        for index, entry in sorted(table.items())
                    )
                    or 'none'
                )
                errors.append(
                    f'Node {node} NIC {device}: no {expected_gid_type} GID with an IPv4-mapped address (found: {summary})'
                )
            common = candidates if common is None else common & candidates
    if errors:
        return None, errors
    if not common:
        summary = '; '.join(f'{nic}={sorted(indices)}' for nic, indices in candidates_by_nic.items())
        return None, [
            f'No GID index is {expected_gid_type} with an IPv4-mapped address on every NIC; candidates per NIC: {summary}'
        ]
    return str(min(common)), []
