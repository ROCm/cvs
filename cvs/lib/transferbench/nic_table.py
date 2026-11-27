"""Read TransferBench's own NIC table.

A NIC executor in a transfer file (``I<src>.<dst>``) is addressed by
TransferBench's index, and ``TB_NIC_FILTER`` renumbers that index to cover only
the NICs it matches. Deriving indices any other way lets them drift from the
ones the tool uses, which would silently measure the wrong pair, so the table
the tool prints is the only authoritative source.
"""

import re

from cvs.lib.env_lib import build_env_prefix
from cvs.lib.transferbench import multirank

# The table is printed by a run with no config, under whatever TB_NIC_FILTER is
# in effect:
#  NIC | Device Name | Active | PCIe Bus ID  | NUMA | Closest GPU(s) | ...
#  0   | ionic_0     | Yes    | 0000:04:00.0 | 0    | 0,1,2,3        | ...
_ROW_RE = re.compile(
    r'^\s*(?P<index>\d+)\s*\|'
    r'\s*(?P<name>\S+)\s*\|'
    r'\s*(?P<active>Yes|No)\s*\|'
    r'\s*(?P<bdf>\S+)\s*\|'
    r'\s*(?P<numa>\S*)\s*\|'
    r'\s*(?P<gpus>[^|]*)\|',
    re.IGNORECASE,
)

_HEADER_RE = re.compile(r'^\s*NIC\s*\|\s*Device Name\s*\|', re.IGNORECASE)


def parse_nic_table(output):
    """Parse the NIC table into records, ignoring banners and other output.

    Returns a list of dicts ordered by TransferBench's own index.
    """
    nics = []
    if not isinstance(output, str):
        return nics
    seen_header = False
    for line in output.splitlines():
        if _HEADER_RE.match(line):
            seen_header = True
            continue
        if not seen_header:
            continue
        match = _ROW_RE.match(line)
        if not match:
            # The table ends at the first line that is not a row; a separator
            # of dashes is not a row either, so it is skipped rather than
            # ending the table.
            if line.strip().startswith('-') or not line.strip():
                continue
            break
        gpus = [part.strip() for part in match.group('gpus').split(',') if part.strip()]
        nics.append(
            {
                'index': int(match.group('index')),
                'nic': match.group('name'),
                'active': match.group('active').lower() == 'yes',
                'bdf': match.group('bdf'),
                'numa': match.group('numa').strip(),
                'closest_gpus': [int(gpu) for gpu in gpus if gpu.isdigit()],
            }
        )
    return nics


def build_nic_filter(names):
    """Build a TB_NIC_FILTER regex matching exactly the given device names.

    Anchored alternation rather than a character class, so a name can never
    match a longer device name that happens to share a prefix.
    """
    cleaned = [str(name).strip() for name in names if str(name).strip()]
    if not cleaned:
        raise ValueError('cannot build a NIC filter from an empty device list')
    return '^(' + '|'.join(re.escape(name) for name in cleaned) + ')$'


def index_by_name(nics):
    """Map device name to TransferBench's index for the current filter."""
    return {nic['nic']: nic['index'] for nic in nics}


def query_nic_table(orch, binary, nic_filter=None, extra_env=None, use_sudo=False, timeout=120):
    """Run TransferBench with no config so it prints its NIC table.

    Returns ``(nics_by_node, failures_by_node)``. Run under the same
    ``TB_NIC_FILTER`` the measurement will use, because the filter decides the
    indices.
    """
    env = {}
    if nic_filter:
        env['TB_NIC_FILTER'] = nic_filter
    cmd = multirank.build_command(
        binary,
        [],
        env,
        use_sudo=use_sudo,
        # build_env_prefix quotes values but interpolates keys as shell
        # syntax; validating first is what makes passing it a raw dict safe.
        runtime_env_prefix=build_env_prefix(multirank.validate_extra_env(extra_env)),
        sudo_argv0='cvs-transferbench-topology',
    )
    out_dict = orch.exec(cmd, timeout=timeout, print_console=False)

    nics_by_node = {}
    failures = {}
    for node, output in (out_dict or {}).items():
        # An empty or partial table has more than one real cause, and each
        # needs different follow-up, so none of them is reported as the
        # others. The exit code is checked before any row is accepted: a
        # connection dropped mid-table can still have printed a few real
        # rows, and accepting those would silently understate NIC coverage.
        if isinstance(output, str) and 'ABORT: Host Unreachable Error' in output:
            failures[node] = 'host unreachable'
            continue
        exit_code = multirank.extract_exit_code(output)
        if exit_code is None:
            failures[node] = 'TransferBench did not complete (no exit status recovered)'
            continue
        if exit_code != 0:
            failures[node] = f'TransferBench exited {exit_code}'
            continue
        nics = parse_nic_table(output)
        if nics:
            nics_by_node[node] = nics
        else:
            failures[node] = 'TransferBench reported no NIC table'
    return nics_by_node, failures
