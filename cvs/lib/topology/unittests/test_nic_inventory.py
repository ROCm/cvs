"""Unit tests for the per-node RDMA NIC inventory.

The NIC/GPU layout used here mirrors a real 4-GPU, 8-NIC accelerator node:
each GPU sits alone in its own PCI domain and the backend NICs share that
domain, while one management NIC sits in a domain with no GPU. Node names are
synthetic.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.topology import nic_inventory

GPU_BDFS = ['0001:01:00.0', '0002:01:00.0', '0003:01:00.0', '0004:01:00.0']

# nic name -> bdf, in the layout the hardware reports.
NIC_LAYOUT = [
    ('ionic_0', '0000:04:00.0'),
    ('ionic_1', '0001:44:00.0'),
    ('ionic_2', '0001:48:00.0'),
    ('ionic_3', '0002:44:00.0'),
    ('ionic_4', '0003:44:00.0'),
    ('ionic_5', '0003:48:00.0'),
    ('ionic_6', '0004:44:00.0'),
    ('ionic_7', '0004:48:00.0'),
]


def _port_line(name, bdf, port, active):
    state, rate = ('4: ACTIVE', '400 Gb/sec (4X NDR)') if active else ('1: DOWN', '2.5 Gb/sec (1X SDR)')
    return f'{nic_inventory.PROBE_MARKER}|{name}|{bdf}|{port}|{state}|{rate}'


def _probe_output(active_nics=None, banner=True):
    active_names = {name for name, _ in NIC_LAYOUT} if active_nics is None else set(active_nics)
    lines = []
    if banner:
        # A login shell prints a banner before the command's own output.
        lines += ['', '  ###  WELCOME BANNER  ###', 'Last login: never']
    for name, bdf in NIC_LAYOUT:
        lines.append(_port_line(name, bdf, '1', name in active_names))
    return '\n'.join(lines)


class TestParseNicProbe(unittest.TestCase):
    def test_banner_and_other_output_is_ignored(self):
        nics = nic_inventory.parse_nic_probe(_probe_output())
        self.assertEqual([nic['nic'] for nic in nics], [name for name, _ in NIC_LAYOUT])

    def test_link_state_is_interpreted(self):
        nics = nic_inventory.parse_nic_probe(_probe_output(active_nics={'ionic_0'}))
        by_name = {nic['nic']: nic for nic in nics}
        self.assertTrue(by_name['ionic_0']['active'])
        self.assertFalse(by_name['ionic_1']['active'])

    def test_malformed_and_empty_lines_are_skipped(self):
        text = '\n'.join(
            [
                f'{nic_inventory.PROBE_MARKER}|ionic_0',
                f'{nic_inventory.PROBE_MARKER}||0000:04:00.0|1|4: ACTIVE|400',
                _port_line('ionic_1', '0001:44:00.0', '1', True),
                'unrelated|line|with|pipes|here|now',
            ]
        )
        nics = nic_inventory.parse_nic_probe(text)
        self.assertEqual([nic['nic'] for nic in nics], ['ionic_1'])

    def test_non_string_output_yields_nothing(self):
        self.assertEqual(nic_inventory.parse_nic_probe(None), [])


class TestMultiPortDevices(unittest.TestCase):
    """A dual-port adapter may carry the backend link on its second port."""

    def test_device_is_active_on_its_second_port(self):
        text = '\n'.join(
            [
                _port_line('mlx5_0', '0001:44:00.0', '1', False),
                _port_line('mlx5_0', '0001:44:00.0', '2', True),
            ]
        )
        nics = nic_inventory.parse_nic_probe(text)

        self.assertEqual(len(nics), 1, 'one record per device, not per port')
        self.assertTrue(nics[0]['active'], 'port 2 is up, so the device is usable')
        self.assertEqual(nics[0]['port'], '2', 'the record must name the port that is up')
        self.assertEqual(nics[0]['ports'], ['1', '2'])

    def test_second_port_device_is_selected_not_excluded(self):
        text = '\n'.join(
            [
                _port_line('mlx5_0', '0001:44:00.0', '1', False),
                _port_line('mlx5_0', '0001:44:00.0', '2', True),
            ]
        )
        selected, excluded = nic_inventory.select_backend_nics(nic_inventory.parse_nic_probe(text), ['0001:01:00.0'])
        self.assertEqual([nic['nic'] for nic in selected], ['mlx5_0'])
        self.assertEqual(excluded, [])

    def test_device_with_every_port_down_is_excluded(self):
        text = '\n'.join(
            [
                _port_line('mlx5_0', '0001:44:00.0', '1', False),
                _port_line('mlx5_0', '0001:44:00.0', '2', False),
            ]
        )
        nics = nic_inventory.parse_nic_probe(text)
        self.assertFalse(nics[0]['active'])
        self.assertEqual(nics[0]['port'], '1', 'falls back to the lowest port for reporting')

    def test_ports_are_ordered_numerically_not_lexically(self):
        text = '\n'.join(
            [
                _port_line('mlx5_0', '0001:44:00.0', '10', False),
                _port_line('mlx5_0', '0001:44:00.0', '2', False),
            ]
        )
        self.assertEqual(nic_inventory.parse_nic_probe(text)[0]['ports'], ['2', '10'])

    def test_probe_enumerates_every_port(self):
        self.assertIn('"$d"/ports/*', nic_inventory.PROBE_CMD)
        self.assertNotIn('ports/1/state', nic_inventory.PROBE_CMD)


class TestUnreadablePciAddress(unittest.TestCase):
    """A device without a usable PCI address must not cost the node its other NICs."""

    def _probe_with_bad_bdf(self, bad):
        return '\n'.join(
            [
                _port_line('good_0', '0001:44:00.0', '1', True),
                _port_line('broken', bad, '1', True),
                _port_line('good_1', '0002:44:00.0', '1', True),
            ]
        )

    def test_one_unreadable_address_does_not_abort_selection(self):
        # A missing device symlink makes readlink -f resolve to '.../device'.
        nics = nic_inventory.parse_nic_probe(self._probe_with_bad_bdf('device'))
        selected, excluded = nic_inventory.select_backend_nics(nics, ['0001:01:00.0', '0002:01:00.0'])

        self.assertEqual([nic['nic'] for nic in selected], ['good_0', 'good_1'])
        self.assertEqual(
            [(nic['nic'], nic['reason']) for nic in excluded],
            [('broken', nic_inventory.EXCLUDED_INVALID_BDF)],
        )

    def test_several_shapes_of_unreadable_address_are_reported(self):
        for bad in ['device', 'not-a-bdf', '0001:4g:00.0', '0001:44:00.9', '::']:
            nics = nic_inventory.parse_nic_probe(self._probe_with_bad_bdf(bad))
            selected, excluded = nic_inventory.select_backend_nics(nics, ['0001:01:00.0', '0002:01:00.0'])
            self.assertEqual([nic['nic'] for nic in selected], ['good_0', 'good_1'], f'bad bdf {bad!r}')
            self.assertEqual([nic['nic'] for nic in excluded], ['broken'], f'bad bdf {bad!r}')

    def test_one_unreadable_gpu_address_does_not_abort_the_node(self):
        # The GPU list comes from a separate probe than the NIC list; a defect
        # on that side must be isolated exactly like a defect on the NIC side.
        nics = nic_inventory.parse_nic_probe(
            '\n'.join([_port_line('nic0', '0001:44:00.0', '1', True), _port_line('nic1', '0002:44:00.0', '1', True)])
        )
        selected, excluded = nic_inventory.select_backend_nics(nics, ['0001:01:00.0', 'not-a-bdf', '0002:01:00.0'])

        self.assertEqual({nic['nic']: nic['gpu'] for nic in selected}, {'nic0': 0, 'nic1': 2})
        self.assertEqual(excluded, [])


class TestSelectBackendNics(unittest.TestCase):
    def test_each_nic_maps_to_the_gpu_in_its_own_pci_domain(self):
        nics = nic_inventory.parse_nic_probe(_probe_output())
        selected, _ = nic_inventory.select_backend_nics(nics, GPU_BDFS)

        self.assertEqual(
            {nic['nic']: nic['gpu'] for nic in selected},
            {'ionic_1': 0, 'ionic_2': 0, 'ionic_3': 1, 'ionic_4': 2, 'ionic_5': 2, 'ionic_6': 3, 'ionic_7': 3},
        )

    def test_management_nic_without_a_local_gpu_is_excluded_with_a_reason(self):
        nics = nic_inventory.parse_nic_probe(_probe_output())
        selected, excluded = nic_inventory.select_backend_nics(nics, GPU_BDFS)

        self.assertNotIn('ionic_0', [nic['nic'] for nic in selected])
        reasons = {nic['nic']: nic['reason'] for nic in excluded}
        self.assertEqual(reasons['ionic_0'], nic_inventory.EXCLUDED_NO_LOCAL_GPU)

    def test_down_link_is_excluded_and_reported_not_dropped(self):
        nics = nic_inventory.parse_nic_probe(_probe_output(active_nics={'ionic_1', 'ionic_2'}))
        selected, excluded = nic_inventory.select_backend_nics(nics, GPU_BDFS)

        self.assertEqual([nic['nic'] for nic in selected], ['ionic_1', 'ionic_2'])
        down = {nic['nic'] for nic in excluded if nic['reason'] == nic_inventory.EXCLUDED_LINK_DOWN}
        self.assertEqual(down, {'ionic_3', 'ionic_4', 'ionic_5', 'ionic_6', 'ionic_7'})

    def test_node_without_gpus_says_so_instead_of_blaming_the_nics(self):
        # Seen on real hardware: the GPUs were present but ROCm could not
        # enumerate them. Reporting every NIC as a front-end NIC would send the
        # user after the wrong problem.
        nics = nic_inventory.parse_nic_probe(_probe_output())
        selected, excluded = nic_inventory.select_backend_nics(nics, [])

        self.assertEqual(selected, [])
        self.assertEqual(len(excluded), len(NIC_LAYOUT))
        self.assertEqual(
            {nic['reason'] for nic in excluded},
            {nic_inventory.EXCLUDED_NO_GPUS_DETECTED},
        )


def _done(output):
    return {'output': output, 'exit_code': 0}


def _failed(output, exit_code=1):
    return {'output': output, 'exit_code': exit_code}


class TestCollectNicInventory(unittest.TestCase):
    def _orch(self, exec_result):
        phdl = MagicMock()
        phdl.exec.return_value = exec_result
        orch = MagicMock()
        orch.all = phdl
        return orch

    def test_collects_per_node_and_reports_a_node_with_no_devices(self):
        orch = self._orch({'nodeA': _done(_probe_output()), 'nodeB': _done('no infiniband here')})
        nics_by_node, failures = nic_inventory.collect_nic_inventory(orch)

        self.assertEqual(len(nics_by_node['nodeA']), len(NIC_LAYOUT))
        self.assertNotIn('nodeB', nics_by_node)
        self.assertEqual(failures['nodeB'], 'no RDMA devices found')

    def test_uses_detailed_results_so_the_exit_code_is_available(self):
        orch = self._orch({'nodeA': _done(_probe_output())})
        nic_inventory.collect_nic_inventory(orch)
        self.assertTrue(orch.all.exec.call_args.kwargs.get('detailed'))

    def test_unreachable_node_is_distinguished_from_a_node_with_no_nics(self):
        # The two causes call for different follow-up and must not read alike.
        orch = self._orch({'nodeA': _failed('\nABORT: Host Unreachable Error', exit_code=-1)})
        _, failures = nic_inventory.collect_nic_inventory(orch)
        self.assertEqual(failures['nodeA'], 'host unreachable')

    def test_partial_output_from_a_dropped_connection_is_not_accepted_as_the_full_inventory(self):
        # The node really has 8 NICs; only 2 lines were ever printed before the
        # connection died, and no exit code was recovered for the attempt.
        partial = '\n'.join(_port_line(name, bdf, '1', True) for name, bdf in NIC_LAYOUT[:2])
        orch = self._orch({'nodeA': _failed(partial, exit_code=-1)})
        nics_by_node, failures = nic_inventory.collect_nic_inventory(orch)

        self.assertNotIn('nodeA', nics_by_node, 'a truncated probe must not be reported as a complete inventory')
        self.assertIn('nodeA', failures)

    def test_nonzero_exit_with_full_looking_output_is_still_rejected(self):
        # A completed-looking probe that nonetheless exited non-zero (for
        # example killed partway by a timeout after flushing its buffer) must
        # not be trusted either; only exit_code == 0 is accepted.
        orch = self._orch({'nodeA': _failed(_probe_output(), exit_code=137)})
        nics_by_node, failures = nic_inventory.collect_nic_inventory(orch)
        self.assertNotIn('nodeA', nics_by_node)
        self.assertIn('nodeA', failures)

    def test_probe_lines_are_marked_so_banners_cannot_be_mistaken_for_devices(self):
        orch = self._orch({'nodeA': _probe_output()})
        nic_inventory.collect_nic_inventory(orch)
        cmd = orch.all.exec.call_args[0][0]
        self.assertIn(nic_inventory.PROBE_MARKER, cmd)


if __name__ == '__main__':
    unittest.main()
