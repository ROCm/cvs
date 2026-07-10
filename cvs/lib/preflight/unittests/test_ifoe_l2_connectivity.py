"""Unit tests for IFoE L2 connectivity preflight check (AIMVT-180)."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.ifoe_l2_connectivity import (
    AfmctlPingParser,
    IfoeL2ConnectivityCheck,
    parse_afmctl_show_device,
)


PASSING_OUTPUT = """\
0001:01:00.1                   : Ping test results (1 pings per port pair)
Accel ID    Port#     IFoE Req        IFoE Rsp        Non-IFoE
--------    -----     --------        ---------       --------
0           0         1/1 PASS        1/1 PASS        1/1 PASS
0           1         1/1 PASS        1/1 PASS        1/1 PASS

Summary:
  IFoE Request    : 2/2 PASS, 0/2 fail (0.00% loss)
  IFoE Response   : 2/2 PASS, 0/2 fail (0.00% loss)
  Non-IFoE        : 2/2 PASS, 0/2 fail (0.00% loss)
"""


FAILING_OUTPUT = """\
0001:01:00.1                   : Ping test results (3 pings per port pair)
Accel ID    Port#     IFoE Req        IFoE Rsp        Non-IFoE
--------    -----     --------        ---------       --------
0           0         3/3 PASS        3/3 PASS        3/3 PASS
0           1         0/3 FAIL        3/3 PASS        1/3 FAIL
0           2         3/3 PASS        2/3 FAIL        3/3 PASS

Summary:
  IFoE Request    : 6/9 PASS, 3/9 fail (33.33% loss)
  IFoE Response   : 8/9 PASS, 1/9 fail (11.11% loss)
  Non-IFoE        : 7/9 PASS, 2/9 fail (22.22% loss)
"""


PARTIAL_LOSS_OUTPUT = """\
0001:01:00.1                   : Ping test results (10 pings per port pair)
Accel ID    Port#     IFoE Req        IFoE Rsp        Non-IFoE
--------    -----     --------        ---------       --------
0           0         9/10 FAIL       10/10 PASS      10/10 PASS

Summary:
  IFoE Request    : 9/10 PASS, 1/10 fail (10.00% loss)
  IFoE Response   : 10/10 PASS, 0/10 fail (0.00% loss)
  Non-IFoE        : 10/10 PASS, 0/10 fail (0.00% loss)
"""


SHOW_DEVICE_OUTPUT = """\
BDF                              : 0001:01:00.1
Spec:
  Accelerator id                 : 0
  Local accelerators             : 0, 1
  Capability:
    No. of network ports         : 72
"""


class TestAfmctlPingParser(unittest.TestCase):
    """Tests for the afmctl ping output parser."""

    def test_passing_output_parsed(self):
        parsed = AfmctlPingParser.parse(PASSING_OUTPUT)
        self.assertEqual(parsed['bdf'], '0001:01:00.1')
        self.assertEqual(parsed['pings_per_port'], 1)
        self.assertEqual(set(parsed['ports'].keys()), {'0', '1'})
        for port in ('0', '1'):
            for ttype in ('ifoe_req', 'ifoe_resp', 'non_ifoe'):
                self.assertEqual(parsed['ports'][port][ttype]['status'], 'PASS')
                self.assertEqual(parsed['ports'][port][ttype]['pass'], 1)
                self.assertEqual(parsed['ports'][port][ttype]['total'], 1)
        for ttype in ('ifoe_req', 'ifoe_resp', 'non_ifoe'):
            self.assertEqual(parsed['summary'][ttype]['status'], 'PASS')
            self.assertEqual(parsed['summary'][ttype]['loss_pct'], 0.0)
            self.assertEqual(parsed['summary'][ttype]['pass'], 2)
            self.assertEqual(parsed['summary'][ttype]['fail'], 0)
        self.assertFalse(parsed['parse_errors'])

    def test_failing_output_parsed(self):
        parsed = AfmctlPingParser.parse(FAILING_OUTPUT)
        self.assertEqual(parsed['pings_per_port'], 3)
        self.assertEqual(parsed['ports']['0']['ifoe_req']['status'], 'PASS')
        self.assertEqual(parsed['ports']['1']['ifoe_req']['status'], 'FAIL')
        self.assertEqual(parsed['ports']['1']['ifoe_req']['pass'], 0)
        self.assertEqual(parsed['ports']['2']['ifoe_resp']['status'], 'FAIL')
        self.assertAlmostEqual(parsed['summary']['ifoe_req']['loss_pct'], 33.33, places=2)
        self.assertEqual(parsed['summary']['ifoe_req']['status'], 'FAIL')
        self.assertEqual(parsed['summary']['ifoe_resp']['status'], 'FAIL')
        self.assertEqual(parsed['summary']['non_ifoe']['status'], 'FAIL')

    def test_empty_output(self):
        parsed = AfmctlPingParser.parse('')
        self.assertEqual(parsed['ports'], {})
        self.assertEqual(parsed['summary'], {})
        self.assertTrue(parsed['parse_errors'])

    def test_garbage_output(self):
        parsed = AfmctlPingParser.parse('command not found\nbash: afmctl: No such file\n')
        self.assertEqual(parsed['ports'], {})
        self.assertEqual(parsed['summary'], {})
        self.assertTrue(parsed['parse_errors'])

    def test_partial_loss_output(self):
        parsed = AfmctlPingParser.parse(PARTIAL_LOSS_OUTPUT)
        self.assertAlmostEqual(parsed['summary']['ifoe_req']['loss_pct'], 10.0)
        self.assertEqual(parsed['summary']['ifoe_req']['fail'], 1)
        self.assertEqual(parsed['summary']['ifoe_req']['status'], 'FAIL')
        self.assertEqual(parsed['summary']['ifoe_resp']['status'], 'PASS')


class TestParseAfmctlShowDevice(unittest.TestCase):
    """Tests for parse_afmctl_show_device()."""

    def test_single_device(self):
        devices = parse_afmctl_show_device(SHOW_DEVICE_OUTPUT)
        self.assertEqual(len(devices), 1)
        d = devices[0]
        self.assertEqual(d['bdf'], '0001:01:00.1')
        self.assertEqual(d['accelerator_id'], 0)
        self.assertEqual(d['local_accelerators'], [0, 1])
        self.assertEqual(d['num_network_ports'], 72)

    def test_multiple_devices(self):
        two_devs = SHOW_DEVICE_OUTPUT + "\n" + SHOW_DEVICE_OUTPUT.replace('0001:01:00.1', '0002:01:00.1')
        devices = parse_afmctl_show_device(two_devs)
        self.assertEqual(len(devices), 2)
        self.assertEqual({d['bdf'] for d in devices}, {'0001:01:00.1', '0002:01:00.1'})

    def test_empty(self):
        self.assertEqual(parse_afmctl_show_device(''), [])

    def test_garbage(self):
        self.assertEqual(parse_afmctl_show_device('bash: afmctl: command not found\n'), [])


class TestIfoeL2ConnectivityCheck(unittest.TestCase):
    """Tests for the IfoeL2ConnectivityCheck class."""

    def _make_phdl(self, reachable_hosts, exec_responses):
        """Build a MagicMock phdl that returns scripted exec() responses.

        Args:
            reachable_hosts: list of host names.
            exec_responses: list of {host: output} dicts returned by successive
                ``phdl.exec()`` calls.
        """
        phdl = MagicMock()
        phdl.reachable_hosts = list(reachable_hosts)
        phdl.exec = MagicMock(side_effect=exec_responses)
        return phdl

    def test_build_ping_command_defaults(self):
        check = IfoeL2ConnectivityCheck(MagicMock())
        cmd = check.build_ping_command('0001:01:00.1', 0)
        self.assertIn('afmctl test ping', cmd)
        self.assertIn('-b 0001:01:00.1', cmd)
        self.assertIn('-c 1', cmd)
        self.assertIn('--dst-accelerator 0', cmd)
        self.assertNotIn('-p ', cmd)
        self.assertNotIn('--traffic-type', cmd)

    def test_build_ping_command_with_ports_and_timeout(self):
        check = IfoeL2ConnectivityCheck(
            MagicMock(),
            afmctl_path='/usr/local/bin/afmctl',
            ports=[0, 1, 2],
            pings_per_port=5,
            per_ping_timeout=10,
            use_sudo=True,
            traffic_types=['ifoe_req'],
        )
        cmd = check.build_ping_command('0001:01:00.1', 3)
        self.assertTrue(cmd.startswith('sudo /usr/local/bin/afmctl test ping'))
        self.assertIn('-b 0001:01:00.1', cmd)
        self.assertIn('-c 5', cmd)
        self.assertIn('-p 0,1,2', cmd)
        self.assertIn('--dst-accelerator 3', cmd)
        self.assertIn('-t 10', cmd)
        self.assertIn('--traffic-type request', cmd)

    def test_build_ping_command_ports_string(self):
        check = IfoeL2ConnectivityCheck(MagicMock(), ports='0-7')
        cmd = check.build_ping_command('0001:01:00.1', 1)
        self.assertIn('-p 0-7', cmd)

    def test_traffic_type_subset_two(self):
        check = IfoeL2ConnectivityCheck(MagicMock(), traffic_types=['ifoe_req', 'non_ifoe'])
        cmd = check.build_ping_command('0001:01:00.1', 0)
        self.assertIn('--traffic-type request,non-ifoe', cmd)

    def test_traffic_type_aliases_normalized(self):
        check = IfoeL2ConnectivityCheck(MagicMock(), traffic_types=['REQUEST', 'response', 'non-ifoe'])
        self.assertEqual(set(check.traffic_types), {'ifoe_req', 'ifoe_resp', 'non_ifoe'})

    def test_run_passes_with_explicit_bdfs(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA', 'nodeB'],
            exec_responses=[
                {'nodeA': PASSING_OUTPUT, 'nodeB': PASSING_OUTPUT},
            ],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            dst_accelerators=[0],
            bdf_discovery='config',
        )
        results = check.run()
        self.assertEqual(set(results.keys()), {'nodeA', 'nodeB'})
        for node in ('nodeA', 'nodeB'):
            self.assertEqual(results[node]['status'], 'PASS')
            accel_block = results[node]['accelerators']['0001:01:00.1']
            self.assertEqual(accel_block['0']['status'], 'PASS')
            self.assertEqual(accel_block['0']['parsed']['summary']['ifoe_req']['loss_pct'], 0.0)
        self.assertEqual(phdl.exec.call_count, 1)

    def test_run_marks_failure_on_loss(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[{'nodeA': FAILING_OUTPUT}],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            bdf_discovery='config',
        )
        results = check.run()
        self.assertEqual(results['nodeA']['status'], 'FAIL')
        self.assertTrue(results['nodeA']['errors'])
        self.assertIn('0001:01:00.1 -> accel 0', results['nodeA']['errors'][0])
        invocation = results['nodeA']['accelerators']['0001:01:00.1']['0']
        self.assertEqual(invocation['status'], 'FAIL')

    def test_run_loss_threshold_allows_partial_loss(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[{'nodeA': PARTIAL_LOSS_OUTPUT}],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            bdf_discovery='config',
            loss_threshold_pct=15.0,
        )
        results = check.run()
        invocation = results['nodeA']['accelerators']['0001:01:00.1']['0']
        self.assertEqual(invocation['status'], 'FAIL')
        node_status = results['nodeA']['status']
        self.assertEqual(node_status, 'FAIL')
        self.assertTrue(
            any('Port 0 IFoE Request' in err for err in invocation['errors']),
            f"expected port-level error, got: {invocation['errors']}",
        )

    def test_run_traffic_type_subset_ignores_excluded(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[{'nodeA': FAILING_OUTPUT}],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            bdf_discovery='config',
            traffic_types=['ifoe_resp'],
        )
        results = check.run()
        invocation = results['nodeA']['accelerators']['0001:01:00.1']['0']
        self.assertEqual(invocation['status'], 'FAIL')
        for err in invocation['errors']:
            self.assertNotIn('IFoE Request', err)
            self.assertNotIn('Non-IFoE', err)

    def test_run_skipped_node_when_bdf_missing(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA', 'nodeB'],
            exec_responses=[
                {'nodeA': SHOW_DEVICE_OUTPUT, 'nodeB': 'bash: afmctl: command not found\n'},
                {'nodeA': PASSING_OUTPUT, 'nodeB': 'bash: afmctl: command not found\n'},
            ],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            dst_accelerators=[0],
            bdf_discovery='auto',
        )
        results = check.run()
        self.assertEqual(results['nodeA']['status'], 'PASS')
        self.assertEqual(results['nodeB']['status'], 'FAIL')
        self.assertTrue(
            any('No IFoE BDFs configured' in e for e in results['nodeB']['errors']),
            f"expected discovery failure message, got: {results['nodeB']['errors']}",
        )

    def test_run_multiple_dst_accelerators(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': PASSING_OUTPUT},
                {'nodeA': PASSING_OUTPUT},
            ],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            dst_accelerators=[0, 1],
            bdf_discovery='config',
        )
        results = check.run()
        self.assertEqual(results['nodeA']['status'], 'PASS')
        accel_block = results['nodeA']['accelerators']['0001:01:00.1']
        self.assertEqual(set(accel_block.keys()), {'0', '1'})
        self.assertEqual(phdl.exec.call_count, 2)


if __name__ == '__main__':
    unittest.main()
