"""Unit tests for IFoE L2 connectivity preflight check (AIMVT-180)."""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.ifoe_l2_connectivity import (
    AfmctlPortParser,
    AfmctlPingParser,
    IfoeL2ConnectivityCheck,
    expand_accelerator_ranges,
    parse_accelerator_ranges,
    parse_afmctl_show_device,
    parse_afmctl_show_device_json,
)
from cvs.lib.preflight.report import PreflightReportGenerator
from cvs.parsers.schemas import PreflightConfigFile


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


SKIP_PASS_OUTPUT = """\
0001:01:00.1                   : Ping test results (1 pings per port pair)
Accel ID    Port#     IFoE Req        IFoE Rsp        Non-IFoE
--------    -----     --------        ---------       --------

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


PING_JSON_OUTPUT = json.dumps(
    {
        "bdf": "0001:01:00.1",
        "pings_per_port": 1,
        "results": [
            {
                "destination_accelerator": 0,
                "port": 0,
                "ifoe_req": {"pass": 1, "total": 1, "status": "PASS"},
                "ifoe_resp": {"pass": 1, "total": 1, "status": "PASS"},
                "non_ifoe": {"pass": 1, "total": 1, "status": "PASS"},
            }
        ],
        "summary": {
            "ifoe_req": {"pass": 1, "total": 1, "fail": 0, "loss_pct": 0.0},
            "ifoe_resp": {"pass": 1, "total": 1, "fail": 0, "loss_pct": 0.0},
            "non_ifoe": {"pass": 1, "total": 1, "fail": 0, "loss_pct": 0.0},
        },
    }
)


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

    def test_skip_pass_output_retains_summary_without_success_rows(self):
        parsed = AfmctlPingParser.parse(SKIP_PASS_OUTPUT)
        self.assertEqual(parsed['bdf'], '0001:01:00.1')
        self.assertEqual(parsed['pings_per_port'], 1)
        self.assertEqual(parsed['ports'], {})
        self.assertEqual(parsed['summary']['ifoe_req']['total'], 2)
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

    def test_json_output_is_parsed_without_text_fallback(self):
        parsed = AfmctlPingParser.parse(PING_JSON_OUTPUT, allow_text_fallback=False)
        self.assertEqual(parsed['format'], 'json')
        self.assertEqual(parsed['bdf'], '0001:01:00.1')
        self.assertEqual(parsed['ports']['0']['accelerator_id'], 0)
        self.assertEqual(parsed['summary']['ifoe_req']['status'], 'PASS')
        self.assertEqual(parsed['parse_errors'], [])

    def test_text_output_is_rejected_when_compatibility_fallback_is_disabled(self):
        parsed = AfmctlPingParser.parse(PASSING_OUTPUT, allow_text_fallback=False)
        self.assertTrue(any('fallback is disabled' in error for error in parsed['parse_errors']))


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

    def test_json_device_inventory_is_first_class(self):
        payload = json.dumps(
            {
                'devices': [
                    {
                        'bdf': '0001:01:00.1',
                        'accelerator_id': 0,
                        'local_accelerators': [0, 1],
                        'vpod_accelerators': [0, 1],
                        'num_network_ports': 36,
                    }
                ]
            }
        )
        devices, errors = parse_afmctl_show_device_json(payload)
        self.assertEqual(errors, [])
        self.assertEqual(devices[0]['bdf'], '0001:01:00.1')
        self.assertEqual(devices[0]['vpod_accelerators'], [0, 1])


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
    """Tests for JSON and versioned-text AFM port inventory parsing."""

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

    def test_mi4xx_json_uses_id_and_nested_link_status(self):
        output = json.dumps(
            {
                "device": [
                    {
                        "bdf": "0001:01:00.1",
                        "port": [
                            {
                                "id": "0",
                                "spec": {"station_id": "0"},
                                "status": {"link_status": "LINK_UP"},
                            },
                            {
                                "id": "1",
                                "spec": {"station_id": "0"},
                                "status": {"link_status": "NO_PHY_LINK, PCS_NO_BLOCK_LOCK"},
                            },
                        ],
                    }
                ]
            }
        )
        parsed = AfmctlPortParser.parse(output, allow_text_fallback=False)
        inventory = parsed['ports_by_bdf']['0001:01:00.1']
        self.assertTrue(inventory['0']['is_up'])
        self.assertFalse(inventory['1']['is_up'])
        self.assertEqual(inventory['0']['station_id'], 0)
        self.assertEqual(inventory['1']['station_id'], 0)
        self.assertEqual(parsed['parse_errors'], [])

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


class TestIfoeL2ConnectivityCheck(unittest.TestCase):
    """Tests for the IfoeL2ConnectivityCheck class."""

    def _make_phdl(self, reachable_hosts, exec_responses, port_artifacts=None):
        """Build a MagicMock phdl that returns scripted exec() responses.

        Args:
            reachable_hosts: list of host names.
            exec_responses: list of {host: output} dicts returned by successive
                ``phdl.exec()`` calls.
            port_artifacts: optional ``{host: JSON}`` mapping written by the
                SFTP mock when AFM port discovery retrieves its private file.
        """
        phdl = MagicMock()
        phdl.reachable_hosts = list(reachable_hosts)
        responses = iter(exec_responses)

        def exec_side_effect(*_args, **_kwargs):
            try:
                return next(responses)
            except StopIteration:
                return {host: '' for host in reachable_hosts}

        def download_file(_remote_file, local_prefix, recurse=False, suffix_separator='_', hosts=None):
            result = {}
            for host in hosts or reachable_hosts:
                if host not in (port_artifacts or {}):
                    raise IOError(f'no SFTP artifact configured for {host}')
                local_path = f'{local_prefix}{suffix_separator}{host}'
                with open(local_path, 'w', encoding='utf-8') as artifact_file:
                    artifact_file.write(port_artifacts[host])
                result[host] = local_path
            return result

        phdl.exec = MagicMock(side_effect=exec_side_effect)
        phdl.download_file = MagicMock(side_effect=download_file)
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
        self.assertNotIn('--json', cmd)
        self.assertIn('--skip-pass', cmd)

    def test_build_show_port_command_uses_json_without_incompatible_brief_flag(self):
        check = IfoeL2ConnectivityCheck(MagicMock())
        command = check.build_show_port_command('0001:01:00.1')
        self.assertIn('show port -b 0001:01:00.1 --json', command)
        self.assertNotIn('--brief', command)

    def test_privileged_commands_resolve_afmctl_before_sudo(self):
        check = IfoeL2ConnectivityCheck(MagicMock(), use_sudo=True)

        commands = [
            check.build_show_device_command(),
            check.build_show_port_command('0001:01:00.1'),
            check.build_ping_command('0001:01:00.1', 0),
        ]
        for command in commands:
            self.assertIn('_cvs_afmctl="$(command -v afmctl)"', command)
            self.assertIn('case "$_cvs_afmctl" in /*)', command)
            self.assertIn('sudo -n "$_cvs_afmctl"', command)

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
        self.assertIn('--skip-pass', cmd)

    def test_skip_pass_summary_proves_selected_port_coverage(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[{'nodeA': SKIP_PASS_OUTPUT}],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            dst_accelerators=[1],
            ports=[0, 2],
            bdf_discovery='config',
            skip_pass=True,
        )

        results = check.run()

        invocation = results['nodeA']['accelerators']['0001:01:00.1']['1']
        self.assertEqual(results['nodeA']['status'], 'PASS')
        self.assertEqual(invocation['parsed']['ports'], {})
        self.assertIn('--skip-pass', invocation['command'])

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
            allow_text_fallback=False,
        )
        results = check.run()
        self.assertEqual(set(results.keys()), {'nodeA', 'nodeB'})
        for node in ('nodeA', 'nodeB'):
            self.assertEqual(results[node]['status'], 'PASS')
            accel_block = results[node]['accelerators']['0001:01:00.1']
            self.assertEqual(accel_block['0']['status'], 'PASS')
            self.assertEqual(accel_block['0']['parsed']['summary']['ifoe_req']['loss_pct'], 0.0)
        self.assertEqual(phdl.exec.call_count, 1)

    def test_run_accepts_json_ping_output_if_a_future_afm_version_emits_it(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[{'nodeA': PING_JSON_OUTPUT}],
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            dst_accelerators=[0],
            bdf_discovery='config',
            allow_text_fallback=False,
        )
        results = check.run()

        self.assertEqual(results['nodeA']['status'], 'PASS')
        invocation = results['nodeA']['accelerators']['0001:01:00.1']['0']
        self.assertEqual(invocation['parsed']['format'], 'json')
        self.assertNotIn('--json', invocation['command'])

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
                {'nodeA': ''},
                {'nodeA': passing_output_for(1, ports=(0, 2))},
            ],
            port_artifacts={'nodeA': port_inventory},
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
        self.assertEqual(invocation['parsed']['summary']['ifoe_req']['total'], 2)
        self.assertEqual(node['port_inventory']['0001:01:00.1']['artifact_transport'], 'sftp')
        self.assertTrue(node['coverage']['complete'])

    def test_up_port_discovery_excludes_physically_up_masked_stations(self):
        port_inventory = json.dumps(
            {
                'device': [
                    {
                        'bdf': '0001:01:00.1',
                        'port': [
                            {
                                'id': '0',
                                'spec': {'station_id': '0', 'admin_state': 'enabled'},
                                'status': {'link_status': 'LINK_UP'},
                            },
                            {
                                'id': '1',
                                'spec': {'station_id': '1', 'admin_state': 'enabled'},
                                'status': {'link_status': 'LINK_UP'},
                            },
                            {
                                'id': '2',
                                'spec': {'station_id': '2', 'admin_state': 'enabled'},
                                'status': {'link_status': 'LINK_UP'},
                            },
                        ],
                    }
                ]
            }
        )
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': FULL_MESH_SHOW_DEVICE_OUTPUT},
                {'nodeA': port_inventory},
                {'nodeA': ''},
                {'nodeA': passing_output_for(1, ports=(0, 2))},
            ],
            port_artifacts={'nodeA': port_inventory},
        )
        check = IfoeL2ConnectivityCheck(
            phdl,
            bdfs=['0001:01:00.1'],
            dst_accelerators=[1],
            ports='up',
            port_discovery='auto',
            bdf_discovery='config',
            admitted_port_ids_by_node={'nodeA': {'0001:01:00.1': [0, 2]}},
        )

        results = check.run()
        inventory = results['nodeA']['port_inventory']['0001:01:00.1']
        invocation = results['nodeA']['accelerators']['0001:01:00.1']['1']
        self.assertEqual(results['nodeA']['status'], 'PASS')
        self.assertEqual(inventory['physical_up_ports'], [0, 1, 2])
        self.assertEqual(inventory['excluded_masked_up_ports'], [1])
        self.assertEqual(inventory['up_ports'], [0, 2])
        self.assertEqual(invocation['selected_ports'], [0, 2])
        self.assertIn('-p 0,2', invocation['command'])

    def test_no_up_ports_fails_closed_without_issuing_ping(self):
        phdl = self._make_phdl(
            reachable_hosts=['nodeA'],
            exec_responses=[
                {'nodeA': FULL_MESH_SHOW_DEVICE_OUTPUT},
                {'nodeA': SHOW_PORT_ALL_DOWN_OUTPUT},
            ],
            port_artifacts={'nodeA': SHOW_PORT_ALL_DOWN_OUTPUT},
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
        self.assertGreaterEqual(phdl.exec.call_count, 3)

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
        self.assertTrue(any('vPOD accelerator membership is unavailable' in error for error in node['errors']))


class TestL2PingConfigContract(unittest.TestCase):
    def test_schema_accepts_only_the_two_customer_facing_options(self):
        config = PreflightConfigFile.model_validate(
            {
                'connectivity_check': {
                    'ifoe': {
                        'l2ping': {
                            'enabled': True,
                            'pings_per_port': 5,
                        }
                    }
                }
            }
        )

        self.assertTrue(config.connectivity_check.ifoe.l2ping.enabled)
        self.assertEqual(config.connectivity_check.ifoe.l2ping.pings_per_port, 5)

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate(
                {
                    'connectivity_check': {
                        'ifoe': {
                            'l2ping': {
                                'enabled': True,
                                'pings_per_port': 3,
                                'loss_threshold_pct': 1.0,
                            }
                        }
                    }
                }
            )

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate(
                {
                    'l2ping': {
                        'enabled': True,
                        'pings_per_port': 3,
                    }
                }
            )

    def test_preflight_entrypoint_uses_fixed_strict_policy(self):
        from cvs.tests.preflight import preflight_checks

        phdl = MagicMock()
        phdl.reachable_hosts = ['nodeA']
        config = {
            'connectivity_check': {
                'ifoe': {
                    'l2ping': {
                        'enabled': True,
                        'pings_per_port': 5,
                    }
                }
            }
        }
        cluster = {'node_dict': {'nodeA': {}}}
        checker_results = {
            'nodeA': {
                'status': 'PASS',
                'errors': [],
                'accelerators': {},
                'coverage': {'complete': True},
            }
        }

        previous_results = dict(preflight_checks.preflight_results)
        preflight_checks.preflight_results.clear()
        try:
            with (
                patch.object(preflight_checks, 'IfoeL2ConnectivityCheck') as checker_cls,
                patch.object(preflight_checks, 'preflight_update_test_result'),
            ):
                checker_cls.return_value.run.return_value = checker_results
                preflight_checks.test_ifoe_l2_connectivity(phdl, config, cluster)

            kwargs = checker_cls.call_args.kwargs
            self.assertEqual(kwargs['pings_per_port'], 5)
            self.assertEqual(kwargs['afmctl_path'], 'afmctl')
            self.assertEqual(kwargs['bdfs'], [])
            self.assertEqual(kwargs['mesh_mode'], 'full_mesh')
            self.assertEqual(kwargs['ports'], 'up')
            self.assertEqual(kwargs['traffic_types'], ['ifoe_req', 'ifoe_resp', 'non_ifoe'])
            self.assertEqual(kwargs['loss_threshold_pct'], 0.0)
            self.assertTrue(kwargs['require_complete_coverage'])
            self.assertTrue(kwargs['strict_discovery'])
            self.assertFalse(kwargs['allow_text_fallback'])
            self.assertTrue(kwargs['skip_pass'])
            self.assertTrue(kwargs['use_sudo'])
            result = preflight_checks.preflight_results['ifoe_l2_connectivity']
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(result['failure_mode'], 'gate')
        finally:
            preflight_checks.preflight_results.clear()
            preflight_checks.preflight_results.update(previous_results)

    def test_disabled_l2ping_skips_without_constructing_checker(self):
        from cvs.tests.preflight import preflight_checks

        phdl = MagicMock()
        phdl.reachable_hosts = ['nodeA']
        previous_results = dict(preflight_checks.preflight_results)
        preflight_checks.preflight_results.clear()
        try:
            with (
                patch.object(preflight_checks, 'IfoeL2ConnectivityCheck') as checker_cls,
                patch.object(preflight_checks, 'preflight_update_test_result'),
            ):
                preflight_checks.test_ifoe_l2_connectivity(
                    phdl,
                    {
                        'connectivity_check': {
                            'ifoe': {
                                'l2ping': {'enabled': False, 'pings_per_port': 3},
                            }
                        }
                    },
                    {'node_dict': {'nodeA': {}}},
                )

            checker_cls.assert_not_called()
            result = preflight_checks.preflight_results['ifoe_l2_connectivity']
            self.assertTrue(result['skipped'])
            self.assertEqual(result['mode'], 'skip')
        finally:
            preflight_checks.preflight_results.clear()
            preflight_checks.preflight_results.update(previous_results)

    def test_report_includes_customer_ping_count(self):
        results = {
            'status': 'PASS',
            'pings_per_port': 5,
            'node_results': {
                'nodeA': {
                    'status': 'PASS',
                    'errors': [],
                    'accelerators': {},
                    'port_inventory': {},
                    'coverage': {'complete': True},
                }
            },
            'coverage': {'complete': True},
        }
        generator = PreflightReportGenerator(MagicMock(), {'ifoe_l2_connectivity': results}, {})

        html_report = generator._generate_ifoe_l2_html(results)

        self.assertIn('pings per port: <code>5</code>', html_report)


if __name__ == '__main__':
    unittest.main()
