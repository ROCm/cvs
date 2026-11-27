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
EXCLUDED_NOT_IN_TB_TABLE = 'TransferBench does not list this device, so it cannot be addressed'
EXCLUDED_NO_TB_GPU = 'TransferBench reports no closest GPU for this device'


DEFAULT_IB_PORT = 1
EXCLUDED_OTHER_PORT = 'its active port is not the one the run addresses'


def select_common_port(nics_by_node):
    """Pick the one IB port the run can address, and drop NICs that disagree.

    TransferBench takes a single global ``IB_PORT_NUMBER``, so every NIC in
    one run has to be measured on the same port. A cluster whose adapters are
    all up on port 2 is served by setting it; a cluster that mixes ports
    cannot be, so the minority is excluded with a reason rather than measured
    on a port that is down.

    Returns ``(port, kept_by_node, dropped_by_node)``.
    """
    counts = {}
    for nics in nics_by_node.values():
        for nic in nics:
            counts[nic.get('port')] = counts.get(nic.get('port'), 0) + 1
    if not counts:
        return DEFAULT_IB_PORT, {}, {}

    # Most-used port first, lowest port number as the tie-break so the choice
    # does not move between runs on a symmetric cluster.
    chosen = sorted(counts, key=lambda port: (-counts[port], _port_sort_key(port)))[0]
    kept = {}
    dropped = {}
    for node, nics in nics_by_node.items():
        on_port = [nic for nic in nics if nic.get('port') == chosen]
        off_port = [dict(nic, reason=EXCLUDED_OTHER_PORT) for nic in nics if nic.get('port') != chosen]
        if on_port:
            kept[node] = on_port
        if off_port:
            dropped[node] = off_port
    try:
        port_number = int(str(chosen).strip())
    except (TypeError, ValueError):
        port_number = DEFAULT_IB_PORT
    return port_number, kept, dropped


def annotate_transferbench_indices(nics_by_node, table_by_node):
    """Attach TransferBench's own NIC and GPU indices to each selected NIC.

    Returns ``(annotated, dropped)``. A transfer file addresses a NIC as
    ``I<n>`` and a GPU as ``G<n>`` using TransferBench's own numbering, which
    is positional in its table and matches neither device-name order nor the
    order ``amd-smi`` lists GPUs in. The PCI-affinity pairing done earlier
    decides *which* NICs are backend NICs; this decides *how to address* them,
    and the two must not be conflated.

    A device TransferBench does not list cannot be addressed at all, so it is
    dropped with a reason rather than silently given a neighbour's index.
    """
    annotated = {}
    dropped = {}
    for node, nics in nics_by_node.items():
        by_name = {row['nic']: row for row in table_by_node.get(node, [])}
        kept = []
        for nic in nics:
            row = by_name.get(nic['nic'])
            if row is None:
                dropped.setdefault(node, []).append(dict(nic, reason=EXCLUDED_NOT_IN_TB_TABLE))
                continue
            closest = row.get('closest_gpus') or []
            if not closest:
                dropped.setdefault(node, []).append(dict(nic, reason=EXCLUDED_NO_TB_GPU))
                continue
            kept.append(dict(nic, index=row['index'], gpu=closest[0]))
        if kept:
            annotated[node] = kept
    return annotated, dropped


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
    out_dict = orch.exec(PROBE_CMD, timeout=timeout, detailed=True, print_console=False)
    nics_by_node = {}
    failures = {}
    for node, res in (out_dict or {}).items():
        output = res.get('output') if isinstance(res, dict) else res
        if isinstance(output, str) and 'ABORT: Host Unreachable Error' in output:
            # An unreachable node is not a node without NICs; the two call for
            # different follow-up and must not read the same in a report.
            failures[node] = 'host unreachable'
            continue
        if not isinstance(res, dict) or res.get('exit_code') != 0:
            # A node that times out or disconnects mid-probe can still have
            # printed a few real lines; accepting those as the full inventory
            # would silently understate how many NICs the node actually has.
            failures[node] = 'probe did not complete (timed out or disconnected before finishing)'
            continue
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
