"""Unit tests for IFoE L2 connectivity preflight check (AIMVT-180)."""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.ifoe_l2_connectivity import (
    AfmctlPortParser,
    AfmctlPingParser,
    IfoeL2ConnectivityCheck,
    expand_accelerator_ranges,
    parse_accelerator_ranges,
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


SHOW_DEVICE_VPOD_OUTPUT = """\
BDF                              : 0004:01:00.1
Spec:
  Accelerator id                 : 24
  Local accelerators             : 24-27, 31
  vPOD accelerators              : 24-31
  Capability:
    No. of network ports         : 72
"""


SHOW_PORT_JSON_OUTPUT = """\
{
  "devices": [
    {
      "bdf": "0004:01:00.1",
      "ports": [
        {"port": 0, "state": "UP"},
        {"port_id": "1", "status": "DOWN"},
        {"port_number": 2, "link_state": "UP"}
      ]
    }
  ]
}
"""


SHOW_PORT_TEXT_OUTPUT = """\
BDF : 0004:01:00.1
Port#    State
0        UP
1        DOWN
2        UP
"""


MI4XX_SHOW_PORT_JSON_OUTPUT = """\
{
  "device": [
    {
      "bdf": "0001:01:00.1",
      "port": [
        {
          "id": "0",
          "spec": {
            "device_bdf": "0001:01:00.1",
            "admin_state": "enabled"
          },
          "status": {
            "link_status": "LINK_UP",
            "ifcp": {"link_up": "yes"}
          }
        },
        {
          "id": "6",
          "spec": {
            "device_bdf": "0001:01:00.1",
            "admin_state": "enabled"
          },
          "status": {
            "link_status": "NO_PHY_LINK, PCS_NO_BLOCK_LOCK",
            "ifcp": {"link_up": "yes"}
          }
        },
        {
          "id": "29",
          "spec": {
            "device_bdf": "0001:01:00.1",
            "admin_state": "enabled"
          },
          "status": {
            "link_status": "NONE",
            "ifcp": {"link_up": "yes"}
          }
        }
      ]
    }
  ]
}
"""


MI4XX_SHOW_PORT_BRIEF_OUTPUT = """\
----------------------------------------------------------------------------------------
Port                 Name         Logical    Admin/Oper       N/W MAC              Speed
                                  index      state
----------------------------------------------------------------------------------------
0001:01:00.1:0/0     netport0     0          enabled/up       02:00:00:0e:03:00    400G
0001:01:00.1:3/6     netport6     6          enabled/down     02:00:00:0e:03:06    400G
0001:01:00.1:14/29   netport29    29         enabled/down     02:00:00:0e:03:1d    400G
----------------------------------------------------------------------------------------
"""


FULL_MESH_SHOW_DEVICE_OUTPUT = """\
BDF                              : 0001:01:00.1
Spec:
  Accelerator id                 : 0
  Local accelerators             : 0-2
  vPOD accelerators              : 0-2
  Capability:
    No. of network ports         : 72
"""


SHOW_PORT_ALL_DOWN_OUTPUT = """\
BDF : 0001:01:00.1
Port#    State
0        DOWN
1        DOWN
"""


def passing_output_for(destination_accelerator, ports=(0,), bdf='0001:01:00.1'):
    """Build internally consistent one-ping AFM output for a destination."""
    rows = '\n'.join(
        f'{destination_accelerator:<11}{port:<10}1/1 PASS        1/1 PASS        1/1 PASS' for port in ports
    )
    total = len(ports)
    return f"""\
{bdf}                   : Ping test results (1 pings per port pair)
Accel ID    Port#     IFoE Req        IFoE Rsp        Non-IFoE
--------    -----     --------        ---------       --------
{rows}

Summary:
  IFoE Request    : {total}/{total} PASS, 0/{total} fail (0.00% loss)
  IFoE Response   : {total}/{total} PASS, 0/{total} fail (0.00% loss)
  Non-IFoE        : {total}/{total} PASS, 0/{total} fail (0.00% loss)
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

    def test_duplicate_port_rows_fail_closed(self):
        duplicate_row = '0           0         1/1 PASS        1/1 PASS        1/1 PASS\n'
        parsed = AfmctlPingParser.parse(PASSING_OUTPUT.replace('\n\nSummary:', f'\n{duplicate_row}\nSummary:'))
        self.assertTrue(any('Duplicate afmctl ping result row' in error for error in parsed['parse_errors']))

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

    def test_range_and_vpod_accelerators_are_expanded(self):
        devices = parse_afmctl_show_device(SHOW_DEVICE_VPOD_OUTPUT)
        self.assertEqual(len(devices), 1)
        device = devices[0]
        self.assertEqual(device['accelerator_id'], 24)
        self.assertEqual(device['local_accelerators'], [24, 25, 26, 27, 31])
        self.assertEqual(device['vpod_accelerators'], list(range(24, 32)))
        self.assertFalse(device['parse_errors'])


class TestAcceleratorRangeParsing(unittest.TestCase):
    """Tests for safe accelerator-list/range parsing."""

    def test_expands_ranges_preserving_first_seen_order(self):
        values, errors = parse_accelerator_ranges(['24-27, 31', 24, '33 34-35'])
        self.assertEqual(values, [24, 25, 26, 27, 31, 33, 34, 35])
        self.assertEqual(errors, [])
        self.assertEqual(expand_accelerator_ranges('0-2, 4'), [0, 1, 2, 4])

    def test_malformed_ranges_are_reported_not_interpreted(self):
        values, errors = parse_accelerator_ranges('24-22, 3x, -1, 7')
        self.assertEqual(values, [7])
        self.assertEqual(len(errors), 3)
        self.assertTrue(any('Descending accelerator range' in error for error in errors))
        self.assertTrue(any('Malformed accelerator ID/range' in error for error in errors))


class TestAfmctlPortParser(unittest.TestCase):
    """Tests for JSON and versioned-text ``show port --brief`` parsing."""

    def test_json_port_inventory_is_scoped_to_bdf_and_selects_only_up(self):
        parsed = AfmctlPortParser.parse(SHOW_PORT_JSON_OUTPUT)
        self.assertEqual(parsed['format'], 'json')
        inventory = parsed['ports_by_bdf']['0004:01:00.1']
        self.assertEqual(set(inventory), {'0', '1', '2'})
        self.assertEqual([port for port, entry in inventory.items() if entry['is_up']], ['0', '2'])
        self.assertEqual(inventory['1']['state'], 'DOWN')
        self.assertFalse(parsed['parse_errors'])

    def test_text_v1_port_inventory_is_scoped_to_current_bdf(self):
        parsed = AfmctlPortParser.parse(SHOW_PORT_TEXT_OUTPUT)
        self.assertEqual(parsed['format'], 'text-v1')
        inventory = parsed['ports_by_bdf']['0004:01:00.1']
        self.assertEqual(inventory['0']['state'], 'UP')
        self.assertFalse(inventory['1']['is_up'])
        self.assertTrue(inventory['2']['is_up'])
        self.assertFalse(parsed['parse_errors'])

    def test_mi4xx_json_uses_physical_link_status_and_skips_ssh_preamble(self):
        parsed = AfmctlPortParser.parse(
            'Conductor SSH authentication banner\n'
            + MI4XX_SHOW_PORT_JSON_OUTPUT
            + '\n__CVS_AFMCTL_EXIT_STATUS__=0\ntransport trailer\n'
        )
        self.assertEqual(parsed['format'], 'json')
        inventory = parsed['ports_by_bdf']['0001:01:00.1']
        self.assertEqual(set(inventory), {'0', '6', '29'})
        self.assertTrue(inventory['0']['is_up'])
        self.assertEqual(inventory['0']['link_status'], 'LINK_UP')
        # IFCP says yes for these ports, but the physical link is not up.
        self.assertFalse(inventory['6']['is_up'])
        self.assertFalse(inventory['29']['is_up'])
        self.assertEqual(inventory['6']['state'], 'DOWN')
        self.assertFalse(parsed['parse_errors'])

    def test_mi4xx_brief_table_fallback_uses_logical_port_index(self):
        parsed = AfmctlPortParser.parse(MI4XX_SHOW_PORT_BRIEF_OUTPUT)
        self.assertEqual(parsed['format'], 'text-v1')
        inventory = parsed['ports_by_bdf']['0001:01:00.1']
        self.assertEqual(set(inventory), {'0', '6', '29'})
        self.assertTrue(inventory['0']['is_up'])
        self.assertFalse(inventory['6']['is_up'])
        self.assertFalse(parsed['parse_errors'])

    def test_unknown_or_malformed_port_formats_fail_closed(self):
        malformed_json = AfmctlPortParser.parse('{"ports": [{"port": 0, "state": "FLAPPING"}]}')
        self.assertEqual(malformed_json['format'], 'json')
        self.assertFalse(malformed_json['ports_by_bdf'])
        self.assertFalse(malformed_json['unscoped_ports'])
        self.assertTrue(any('Unknown port state' in error for error in malformed_json['parse_errors']))

        unknown_text = AfmctlPortParser.parse('afmctl version 99\nthere are some ports\n')
        self.assertEqual(unknown_text['format'], 'text-v1')
        self.assertFalse(unknown_text['ports_by_bdf'])
        self.assertFalse(unknown_text['unscoped_ports'])
        self.assertTrue(any('Could not locate' in error for error in unknown_text['parse_errors']))

    def test_mi4xx_unknown_physical_link_state_fails_closed(self):
        output = MI4XX_SHOW_PORT_JSON_OUTPUT.replace('"LINK_UP"', '"LINK_TRAINING"', 1)
        parsed = AfmctlPortParser.parse(output)
        self.assertFalse(parsed['ports_by_bdf']['0001:01:00.1'].get('0'))
        self.assertTrue(any('Unknown AFM physical link state' in error for error in parsed['parse_errors']))


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
        self.assertTrue(cmd.startswith('sudo -n afmctl test ping'))
        self.assertIn('afmctl test ping', cmd)
        self.assertIn('-b 0001:01:00.1', cmd)
        self.assertIn('-c 1', cmd)
        self.assertIn('--dst-accelerator 0', cmd)
        self.assertNotIn('-p ', cmd)
        self.assertNotIn('--traffic-type', cmd)

    def test_build_show_port_command_uses_validated_mi4xx_json_grammar(self):
        check = IfoeL2ConnectivityCheck(MagicMock())
        cmd = check.build_show_port_command('0001:01:00.1')
        self.assertEqual(cmd, 'sudo -n afmctl show port --json -b 0001:01:00.1')
        self.assertNotIn('--brief', cmd)

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
        self.assertTrue(cmd.startswith('sudo -n /usr/local/bin/afmctl test ping'))
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

    def test_invalid_bdf_or_bdf_discovery_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid IFoE source BDF'):
            IfoeL2ConnectivityCheck(MagicMock(), bdfs=['not-a-bdf'])
        with self.assertRaisesRegex(ValueError, "bdf_discovery must be 'auto' or 'config'"):
            IfoeL2ConnectivityCheck(MagicMock(), bdf_discovery='best-effort')

    def test_targeted_executor_runs_only_on_requested_node_and_extracts_exit_status(self):
        class TargetedPssh:
            reachable_hosts = ['nodeA', 'nodeB']

            def __init__(self):
                self.command_list = None

            def exec_cmd_list(self, commands, timeout=None, print_console=True):
                self.command_list = commands
                return {
                    'nodeA': '',
                    'nodeB': 'ping output\n__CVS_AFMCTL_EXIT_STATUS__=7\n',
                }

        phdl = TargetedPssh()
        check = IfoeL2ConnectivityCheck(phdl)
        execution = check._exec_on_node('nodeB', 'afmctl test ping --example')

        self.assertEqual(phdl.command_list[0], 'true')
        self.assertIn('afmctl test ping --example', phdl.command_list[1])
        self.assertIn('__CVS_AFMCTL_EXIT_STATUS__', phdl.command_list[1])
        self.assertEqual(execution, {'output': 'ping output', 'exit_status': 7})

    def test_exit_sentinel_is_extracted_with_trailing_transport_output(self):
        output, exit_status = IfoeL2ConnectivityCheck._extract_exit_sentinel(
            'afmctl json\n__CVS_AFMCTL_EXIT_STATUS__=3\nSSH transport footer\n'
        )
        self.assertEqual(output, 'afmctl json\nSSH transport footer\n')
        self.assertEqual(exit_status, 3)

    def test_run_passes_with_explicit_bdfs(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA', 'nodeB'],
            exec_responses=[
                {'nodeA': passing_output_for(0), 'nodeB': passing_output_for(0)},
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
                {'nodeA': passing_output_for(1)},
            ],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            dst_accelerators=[1],
            bdf_discovery='auto',
        )
        results = check.run()
        self.assertEqual(results['nodeA']['status'], 'PASS')
        self.assertEqual(results['nodeB']['status'], 'FAIL')
        self.assertTrue(
            any('No IFoE source BDFs available' in e for e in results['nodeB']['errors']),
            f"expected discovery failure message, got: {results['nodeB']['errors']}",
        )

    def test_run_multiple_dst_accelerators(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': passing_output_for(0)},
                {'nodeA': passing_output_for(1)},
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

    def test_full_mesh_plans_ordered_nonself_destinations(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': FULL_MESH_SHOW_DEVICE_OUTPUT},
                {'nodeA': passing_output_for(1)},
                {'nodeA': passing_output_for(2)},
            ],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            mesh_mode='full_mesh',
            ports=[0],
            bdf_discovery='auto',
        )

        results = check.run()

        node = results['nodeA']
        self.assertEqual(node['status'], 'PASS')
        self.assertEqual([cell['dst_accelerator'] for cell in node['plan']], [1, 2])
        self.assertTrue(all(cell['dst_accelerator'] != cell['source_accelerator'] for cell in node['plan']))
        self.assertEqual(
            node['coverage'],
            {
                'expected_pairs': 2,
                'planned_pairs': 2,
                'expected_invocations': 2,
                'completed_invocations': 2,
                'complete': True,
            },
        )
        executed_commands = [call.args[0] for call in phdl.exec.call_args_list]
        self.assertIn('--dst-accelerator 1', executed_commands[1])
        self.assertIn('--dst-accelerator 2', executed_commands[2])
        self.assertNotIn('--dst-accelerator 0', '\n'.join(executed_commands[1:]))

    def test_up_port_discovery_filters_down_ports_from_ping(self):
        port_inventory = """\
BDF : 0001:01:00.1
Port#    State
0        UP
1        DOWN
2        UP
"""
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': FULL_MESH_SHOW_DEVICE_OUTPUT},
                {'nodeA': port_inventory},
                {'nodeA': passing_output_for(1, ports=(0, 2))},
            ],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            dst_accelerators=[1],
            ports='up',
            port_discovery='auto',
            bdf_discovery='config',
        )

        results = check.run()

        node = results['nodeA']
        self.assertEqual(node['status'], 'PASS')
        self.assertEqual(node['port_inventory']['0001:01:00.1']['up_ports'], [0, 2])
        invocation = node['accelerators']['0001:01:00.1']['1']
        self.assertEqual(invocation['selected_ports'], [0, 2])
        self.assertIn('-p 0,2', invocation['command'])
        self.assertEqual(set(invocation['parsed']['ports']), {'0', '2'})
        self.assertTrue(node['coverage']['complete'])

    def test_mi4xx_json_up_port_discovery_builds_a_scoped_ping(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': FULL_MESH_SHOW_DEVICE_OUTPUT},
                {'nodeA': 'Conductor SSH authentication banner\n' + MI4XX_SHOW_PORT_JSON_OUTPUT},
                {'nodeA': passing_output_for(1, ports=(0,))},
            ],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            dst_accelerators=[1],
            ports='up',
            port_discovery='auto',
            bdf_discovery='config',
        )

        results = check.run()

        node = results['nodeA']
        self.assertEqual(node['status'], 'PASS')
        self.assertEqual(node['port_inventory']['0001:01:00.1']['up_ports'], [0])
        invocation = node['accelerators']['0001:01:00.1']['1']
        self.assertIn('-p 0', invocation['command'])
        self.assertIn('show port --json -b 0001:01:00.1', phdl.exec.call_args_list[1].args[0])

    def test_no_up_ports_fails_closed_without_issuing_ping(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': FULL_MESH_SHOW_DEVICE_OUTPUT},
                {'nodeA': SHOW_PORT_ALL_DOWN_OUTPUT},
            ],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            dst_accelerators=[1],
            ports='up',
            port_discovery='auto',
            bdf_discovery='config',
        )

        results = check.run()

        node = results['nodeA']
        self.assertEqual(node['status'], 'FAIL')
        self.assertEqual(node['port_inventory']['0001:01:00.1']['up_ports'], [])
        self.assertEqual(node['plan'], [])
        self.assertFalse(node['coverage']['complete'])
        self.assertTrue(any(issue['category'] == 'PORT_DISCOVERY_ERROR' for issue in node['error_details']))
        self.assertTrue(any(issue['category'] == 'COVERAGE_ERROR' for issue in node['error_details']))
        self.assertEqual(phdl.exec.call_count, 2)

    def test_nonzero_ping_exit_status_fails_even_when_output_is_passing(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': {'output': passing_output_for(1), 'exit_code': 3}},
            ],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            dst_accelerators=[1],
            bdf_discovery='config',
        )

        results = check.run()

        invocation = results['nodeA']['accelerators']['0001:01:00.1']['1']
        self.assertEqual(results['nodeA']['status'], 'FAIL')
        self.assertEqual(invocation['failure_category'], 'COMMAND_ERROR')
        self.assertTrue(any('exited with status 3' in error for error in invocation['errors']))

    def test_missing_result_bdf_fails_closed(self):
        output_without_banner = '\n'.join(passing_output_for(1).splitlines()[1:])
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[{'nodeA': output_without_banner}],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            dst_accelerators=[1],
            bdf_discovery='config',
        )

        results = check.run()

        invocation = results['nodeA']['accelerators']['0001:01:00.1']['1']
        self.assertEqual(results['nodeA']['status'], 'FAIL')
        self.assertEqual(invocation['failure_category'], 'PARSE_ERROR')
        self.assertTrue(any('did not identify the requested source BDF' in error for error in invocation['errors']))

    def test_strict_full_mesh_requires_vpod_membership(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[{'nodeA': SHOW_DEVICE_OUTPUT}],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            mesh_mode='full_mesh',
            ports=[0],
            bdf_discovery='auto',
            strict_discovery=True,
        )

        results = check.run()

        node = results['nodeA']
        self.assertEqual(node['status'], 'FAIL')
        self.assertEqual(node['plan'], [])
        self.assertFalse(node['coverage']['complete'])
        self.assertTrue(
            any('vPOD accelerator membership is unavailable' in error for error in node['errors'])
        )


if __name__ == '__main__':
    unittest.main()
