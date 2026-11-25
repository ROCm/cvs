'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

import json

import pytest

from cvs.lib import globals
from cvs.lib.linux_utils import get_gpu_nic_mapping_dict
from cvs.lib.topology import nic_inventory, nic_matrix, nic_reporting, nic_runner
from cvs.lib.utils_lib import fail_test, update_test_result

log = globals.log

TOPOLOGY_JSON = 'topology.json'


def _gpu_bdfs_by_node(orch):
    """GPU PCI addresses per node, in GPU order, from the shared mapping."""
    mapping = get_gpu_nic_mapping_dict(orch.all)
    gpu_bdfs = {}
    for node, cards in (mapping or {}).items():
        ordered = sorted(cards, key=lambda card: int(''.join(ch for ch in card if ch.isdigit()) or 0))
        gpu_bdfs[node] = [cards[card].get('gpu_bdf') for card in ordered if cards[card].get('gpu_bdf')]
    return gpu_bdfs


def _collect_inventory(orch):
    """Backend NICs per node, plus the NICs left out and why."""
    nics_by_node, probe_failures = nic_inventory.collect_nic_inventory(orch)
    gpu_bdfs = _gpu_bdfs_by_node(orch)

    inventory = {}
    excluded_by_node = {}
    for node, nics in nics_by_node.items():
        selected, excluded = nic_inventory.select_backend_nics(nics, gpu_bdfs.get(node, []))
        if selected:
            inventory[node] = selected
        excluded_by_node[node] = excluded
        log.info('%s: %d backend NIC(s) selected, %d excluded', node, len(selected), len(excluded))
    for node, reason in probe_failures.items():
        log.warning('%s: %s', node, reason)
    return inventory, excluded_by_node


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
    if len(orch.hosts) < 2:
        pytest.skip('cross-node NIC bandwidth needs at least 2 nodes')

    inventory, excluded_by_node = _collect_inventory(orch)
    if len(inventory) < 2:
        pytest.skip(f'cross-node NIC bandwidth needs backend NICs on at least 2 nodes, found {len(inventory)}')

    matrix, pairs, problems = nic_runner.run_matrix(orch, inventory, settings)
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
