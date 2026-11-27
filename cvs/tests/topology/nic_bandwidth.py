'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json

import pytest

from cvs.lib import globals, rocm_plib
from cvs.lib.topology import nic_inventory, nic_matrix, nic_reporting, nic_runner
from cvs.lib.transferbench import nic_table
from cvs.lib.utils_lib import fail_test, get_passwordless_sudo_status, update_test_result
from cvs.lib.verify_lib import DmesgWindow
from cvs.tests.topology.conftest import launch_containers, teardown_containers

log = globals.log

TOPOLOGY_JSON = 'topology.json'


def test_launch_containers(orch, lifecycle):
    launch_containers(orch, lifecycle)


def _gpu_bdfs_by_node(orch):
    """GPU PCI addresses per node, in card order.

    Only the GPU addresses are needed, to decide which NICs share a PCI domain
    with a GPU. The full GPU-to-NIC mapping would also run lshw under sudo,
    which the shipped ``use_sudo: false`` config does not assume is available.
    """
    bus_dict = rocm_plib.get_gpu_pcie_bus_dict(orch, use_sudo=False)
    gpu_bdfs = {}
    for node, cards in (bus_dict or {}).items():
        ordered = sorted(cards, key=lambda card: int(''.join(ch for ch in card if ch.isdigit()) or 0))
        gpu_bdfs[node] = [cards[card].get('PCI Bus') for card in ordered if cards[card].get('PCI Bus')]
    return gpu_bdfs


def _collect_inventory(orch, settings):
    """Backend NICs per node, plus the NICs left out and why.

    Two different questions, answered by two different sources: PCI affinity
    decides which NICs are backend NICs, and TransferBench's own table decides
    how to address them in the transfer file.
    """
    nics_by_node, probe_failures = nic_inventory.collect_nic_inventory(orch)
    gpu_bdfs = _gpu_bdfs_by_node(orch)

    selected_by_node = {}
    excluded_by_node = {}
    for node, nics in nics_by_node.items():
        selected, excluded = nic_inventory.select_backend_nics(nics, gpu_bdfs.get(node, []))
        if selected:
            selected_by_node[node] = selected
        excluded_by_node[node] = excluded

    table_by_node, table_failures = nic_table.query_nic_table(
        orch,
        settings['tb_binary'],
        nic_filter=settings.get('nic_filter'),
        extra_env=settings.get('transferbench_env'),
        use_sudo=settings['use_sudo'],
    )
    inventory, unaddressable = nic_inventory.annotate_transferbench_indices(selected_by_node, table_by_node)
    for node, items in unaddressable.items():
        excluded_by_node.setdefault(node, []).extend(items)

    for node in sorted(set(selected_by_node) | set(excluded_by_node)):
        log.info(
            '%s: %d backend NIC(s) addressable, %d excluded',
            node,
            len(inventory.get(node, [])),
            len(excluded_by_node.get(node, [])),
        )
    for node, reason in sorted(probe_failures.items()):
        log.warning('%s: %s', node, reason)
    for node, reason in sorted(table_failures.items()):
        log.warning('%s: NIC table unavailable: %s', node, reason)
    return inventory, excluded_by_node, table_failures


def _dmesg_window(orch):
    """Kernel-log window for the measurement, or None when it cannot be read.

    This deliberately uses the host handle even under a container
    orchestrator: a GPU reset or a NIC/PCIe error is a host kernel event, and
    the container's view of the log is not where it is recorded. Reading it
    needs passwordless sudo, which the shipped ``use_sudo: false`` config does
    not assume, so an unreadable log is reported and the run continues rather
    than failing on an unrelated privilege gap.
    """
    sudo_status = get_passwordless_sudo_status(orch.all)
    if sudo_status and all(sudo_status.values()):
        return DmesgWindow(orch.all, 'nic_bandwidth', False)
    log.warning('skipping the dmesg error scan: the kernel log needs passwordless sudo on every node')
    return None


def _write_topology_json(document):
    """Write topology.json beside the run's other artefacts."""
    from cvs.core.run_layout import RunLayout

    path = RunLayout.get().run_dir / TOPOLOGY_JSON
    with open(path, 'w') as fp:
        json.dump(document, fp, indent=2, sort_keys=True)
    log.info('Wrote %s', path)
    return str(path)


def test_nic_bandwidth(orch, settings, cluster_dict, cvs_results_dict):
    """Measure cross-node NIC bandwidth and flag pairs below the cluster median."""
    globals.error_list = []
    if len(orch.hosts) < 2:
        pytest.skip('cross-node NIC bandwidth needs at least 2 nodes')

    inventory, excluded_by_node, table_failures = _collect_inventory(orch, settings)

    # A node CVS could not query was never validated. Reporting the cluster as
    # healthy because the remaining nodes measured well would answer a question
    # the user did not ask, so every such node fails the test whether or not
    # enough nodes survived to measure.
    for node, reason in sorted(table_failures.items()):
        fail_test(f'{node}: TransferBench could not report its NIC table, so the node was not validated ({reason})')

    if len(inventory) < 2:
        if table_failures:
            update_test_result()
            return
        pytest.skip(f'cross-node NIC bandwidth needs backend NICs on at least 2 nodes, found {len(inventory)}')

    # A GPU reset or a NIC/PCIe error during the run leaves bandwidth looking
    # fine, so the kernel log is scanned over the window the workload ran in.
    dmesg_window = _dmesg_window(orch)
    if dmesg_window:
        dmesg_window.open()
    try:
        matrix, pairs, problems = nic_runner.run_matrix(orch, inventory, settings)
    finally:
        # Bracketed in a finally so a failed run is still scanned: a host that
        # fell over mid-measurement is exactly when the kernel log matters.
        if dmesg_window:
            dmesg_window.close_and_scan()

    for problem in problems:
        log.warning('reduced coverage: %s', problem)
    if not matrix:
        fail_test(f'TransferBench returned no results for {len(pairs)} NIC pair(s)')
        update_test_result()
        return

    flags = nic_matrix.flag_outliers(matrix, settings['flag_below_pct_of_median'])
    log.info(
        'Measured %d cross-node NIC pair(s); median %.2f GB/s, flagging below %.2f GB/s (traffic class %s)',
        len(pairs),
        flags['median'],
        flags['threshold'],
        settings['traffic_class'],
    )

    meta = nic_reporting.make_meta(cluster_dict, 'nic_bandwidth')
    nic_reporting.record_results(cvs_results_dict, matrix, flags, meta=meta, excluded_by_node=excluded_by_node)
    _write_topology_json(
        nic_matrix.to_topology_json(
            matrix,
            flags,
            meta={
                'coverage': settings['coverage'],
                'traffic_class': settings['traffic_class'],
                'message_size': settings['message_size'],
                'queue_pairs': settings['queue_pairs'],
                'flag_below_pct_of_median': settings['flag_below_pct_of_median'],
                'pairs_measured': len(pairs),
            },
        ),
    )

    for message in nic_matrix.describe_flags(flags):
        fail_test(message)
    update_test_result()


def test_teardown_containers(orch, lifecycle):
    teardown_containers(orch, lifecycle)
