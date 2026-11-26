"""Per-node RDMA NIC inventory and the NIC-to-GPU map.

Topology tests address NICs individually, so they need every backend NIC on a
node and the GPU that drives each one -- not the one-NIC-per-GPU pairing that
``pair_gpus_with_nics`` produces.
"""

from cvs.lib.linux_utils import closest_gpu_for_nics, is_valid_pci_bdf

# Every probe line carries this marker because a login shell may print a banner
# or MOTD ahead of the command's own output.
PROBE_MARKER = 'CVSNIC'

# Every port is reported, not just port 1: a dual-port adapter can have its
# first port down and its second carrying the backend link.
PROBE_CMD = (
    'for d in /sys/class/infiniband/*; do '
    '[ -e "$d" ] || continue; '
    'n=$(basename "$d"); '
    'bdf=$(basename "$(readlink -f "$d/device")"); '
    'for p in "$d"/ports/*; do '
    '[ -e "$p" ] || continue; '
    'pn=$(basename "$p"); '
    's=$(cat "$p/state" 2>/dev/null); '
    'r=$(cat "$p/rate" 2>/dev/null); '
    f'printf \'{PROBE_MARKER}|%s|%s|%s|%s|%s\\n\' "$n" "$bdf" "$pn" "$s" "$r"; '
    'done; '
    'done'
)

EXCLUDED_LINK_DOWN = 'no ACTIVE port'
EXCLUDED_NO_LOCAL_GPU = 'no GPU in the same PCI domain (treated as a front-end NIC)'
EXCLUDED_INVALID_BDF = 'PCI address could not be read'
# Distinct from the front-end verdict: with no GPUs discovered, nothing can be
# said about a NIC's affinity, and reporting them all as front-end NICs would
# send the user after the wrong problem.
EXCLUDED_NO_GPUS_DETECTED = 'no GPUs were detected on this node, so NIC affinity is unknown'


def _port_sort_key(port):
    """Order ports numerically where possible, so port 2 follows port 1."""
    try:
        return (0, int(port))
    except (TypeError, ValueError):
        return (1, str(port))


def parse_nic_probe(output):
    """Parse marked probe lines into one record per device.

    A device reports one line per port. The record keeps the port that carries
    the link, so a caller drives traffic on the port that is actually up.
    """
    by_device = {}
    order = []
    if not isinstance(output, str):
        return []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith(PROBE_MARKER + '|'):
            continue
        fields = line.split('|')
        if len(fields) != 6:
            continue
        _, name, bdf, port, state, rate = fields
        if not name or not bdf:
            continue
        if name not in by_device:
            by_device[name] = []
            order.append(name)
        by_device[name].append(
            {
                'port': port.strip(),
                'state': state.strip(),
                'rate': rate.strip(),
                'active': 'ACTIVE' in state.upper(),
                'bdf': bdf,
            }
        )

    nics = []
    for name in order:
        ports = sorted(by_device[name], key=lambda entry: _port_sort_key(entry['port']))
        chosen = next((entry for entry in ports if entry['active']), ports[0])
        nics.append(
            {
                'nic': name,
                'bdf': chosen['bdf'],
                'port': chosen['port'],
                'state': chosen['state'],
                'rate': chosen['rate'],
                'active': chosen['active'],
                'ports': [entry['port'] for entry in ports],
            }
        )
    nics.sort(key=lambda nic: nic['nic'])
    return nics


def collect_nic_inventory(orch, timeout=60):
    """Probe every reachable node for its RDMA NICs.

    Returns ``(nics_by_node, failures_by_node)``. A node that returns nothing
    usable is reported rather than raised, so one bad node does not stop a run.
    """
    out_dict = orch.all.exec(PROBE_CMD, timeout=timeout, print_console=False)
    nics_by_node = {}
    failures = {}
    for node, output in (out_dict or {}).items():
        nics = parse_nic_probe(output)
        if nics:
            nics_by_node[node] = nics
        elif isinstance(output, str) and 'ABORT: Host Unreachable Error' in output:
            # An unreachable node is not a node without NICs; the two call for
            # different follow-up and must not read the same in a report.
            failures[node] = 'host unreachable'
        else:
            failures[node] = 'no RDMA devices found'
    return nics_by_node, failures


def select_backend_nics(nics, gpu_bdf_list):
    """Split a node's NICs into testable backend NICs and excluded ones.

    Returns ``(selected, excluded)``. Each selected entry gains a ``gpu`` index
    into ``gpu_bdf_list``. Excluded entries carry a ``reason`` so the report can
    say why a NIC was left out instead of silently shrinking the matrix.
    """
    if not gpu_bdf_list:
        return [], [dict(nic, reason=EXCLUDED_NO_GPUS_DETECTED) for nic in nics]

    # An address that cannot be parsed is set aside first: letting it reach the
    # pairing helper would raise and cost the node every other NIC.
    usable = [nic for nic in nics if is_valid_pci_bdf(nic['bdf'])]
    excluded = [dict(nic, reason=EXCLUDED_INVALID_BDF) for nic in nics if not is_valid_pci_bdf(nic['bdf'])]

    closest = closest_gpu_for_nics(gpu_bdf_list, [nic['bdf'] for nic in usable])
    selected = []
    for nic, gpu_index in zip(usable, closest):
        if gpu_index is None:
            excluded.append(dict(nic, reason=EXCLUDED_NO_LOCAL_GPU))
            continue
        if not nic['active']:
            excluded.append(dict(nic, reason=EXCLUDED_LINK_DOWN))
            continue
        selected.append(dict(nic, gpu=gpu_index))
    return selected, excluded
