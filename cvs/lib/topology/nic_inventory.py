"""Per-node RDMA NIC inventory and the NIC-to-GPU map.

Topology tests address NICs individually, so they need every backend NIC on a
node and the GPU that drives each one -- not the one-NIC-per-GPU pairing that
``pair_gpus_with_nics`` produces.
"""

from cvs.lib.linux_utils import closest_gpu_for_nics

# Every probe line carries this marker because a login shell may print a banner
# or MOTD ahead of the command's own output.
PROBE_MARKER = 'CVSNIC'

PROBE_CMD = (
    'for d in /sys/class/infiniband/*; do '
    '[ -e "$d" ] || continue; '
    'n=$(basename "$d"); '
    'bdf=$(basename "$(readlink -f "$d/device")"); '
    's=$(cat "$d/ports/1/state" 2>/dev/null); '
    'r=$(cat "$d/ports/1/rate" 2>/dev/null); '
    f'printf \'{PROBE_MARKER}|%s|%s|%s|%s\\n\' "$n" "$bdf" "$s" "$r"; '
    'done'
)

EXCLUDED_LINK_DOWN = 'link is not ACTIVE'
EXCLUDED_NO_LOCAL_GPU = 'no GPU in the same PCI domain (treated as a front-end NIC)'


def parse_nic_probe(output):
    """Parse marked probe lines into NIC records, ignoring any other output."""
    nics = []
    if not isinstance(output, str):
        return nics
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith(PROBE_MARKER + '|'):
            continue
        fields = line.split('|')
        if len(fields) != 5:
            continue
        _, name, bdf, state, rate = fields
        if not name or not bdf:
            continue
        nics.append(
            {
                'nic': name,
                'bdf': bdf,
                'state': state.strip(),
                'rate': rate.strip(),
                'active': 'ACTIVE' in state.upper(),
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
        else:
            failures[node] = 'no RDMA devices found'
    return nics_by_node, failures


def select_backend_nics(nics, gpu_bdf_list):
    """Split a node's NICs into testable backend NICs and excluded ones.

    Returns ``(selected, excluded)``. Each selected entry gains a ``gpu`` index
    into ``gpu_bdf_list``. Excluded entries carry a ``reason`` so the report can
    say why a NIC was left out instead of silently shrinking the matrix.
    """
    closest = closest_gpu_for_nics(gpu_bdf_list, [nic['bdf'] for nic in nics])
    selected = []
    excluded = []
    for nic, gpu_index in zip(nics, closest):
        if gpu_index is None:
            excluded.append(dict(nic, reason=EXCLUDED_NO_LOCAL_GPU))
            continue
        if not nic['active']:
            excluded.append(dict(nic, reason=EXCLUDED_LINK_DOWN))
            continue
        selected.append(dict(nic, gpu=gpu_index))
    return selected, excluded
