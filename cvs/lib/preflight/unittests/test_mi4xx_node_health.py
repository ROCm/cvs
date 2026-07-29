"""Unit tests for the read-only MI4XX node-health admission gate."""

import json
import os
import re
import sys
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from cvs.lib.preflight.mi4xx_node_health import (  # noqa: E402
    Mi4xxNodeHealthCheck,
    NodeHealthCheck,
    parse_afmctl_device_json,
    parse_amd_smi_gpu_json,
    parse_station_masks,
)
from cvs.lib.preflight.report import PreflightReportGenerator  # noqa: E402
from cvs.parsers.schemas import PreflightConfigFile  # noqa: E402


def _afm_devices(*, phase="ACTIVE", vpod=None):
    membership = list(vpod if vpod is not None else range(4))
    return {
        "devices": [
            {
                "bdf": f"0000:{index + 1:02x}:00.1",
                "accelerator_id": index,
                "config_phase": phase,
                "virtualization_mode": "bare-metal",
                "local_accelerators": list(range(4)),
                "vpod_accelerators": membership,
                "num_network_ports": 36,
            }
            for index in range(4)
        ]
    }


def _gpu_inventory():
    return {"gpus": [{"bdf": f"0000:{index + 1:02x}:00.0", "gpu_id": index} for index in range(4)]}


def _port_inventory(bdfs, down_ports=()):
    return {
        "devices": [
            {
                "bdf": bdf,
                "ports": [
                    {
                        "port": port,
                        "state": "DOWN" if port in down_ports else "UP",
                        "spec": {"station_id": port // 2},
                    }
                    for port in range(36)
                ],
            }
            for bdf in bdfs
        ]
    }


class FakePssh:
    """Small command-aware PSSH substitute; no host actions are performed."""

    def __init__(
        self,
        hosts=("nodeA", "nodeB"),
        *,
        phase_by_host=None,
        vpod_by_host=None,
        station_mask='f' * 18,
        down_ports_by_host=None,
    ):
        self.reachable_hosts = list(hosts)
        self.phase_by_host = dict(phase_by_host or {})
        self.vpod_by_host = dict(vpod_by_host or {})
        self.station_mask = station_mask
        self.down_ports_by_host = {host: set(ports) for host, ports in (down_ports_by_host or {}).items()}
        self.commands = []

    def exec(self, command, **_kwargs):
        self.commands.append(command)
        result = {}
        for host in self.reachable_hosts:
            if "/sys/module/amdgpu" in command or "/sys/module/ifoe" in command:
                output = "1"
            elif "/dev/kfd" in command:
                output = "1"
            elif "pgrep -af" in command:
                output = "102 inb-node-agent --slot-id 4\n"
            elif "journalctl -k" in command:
                output = "amdgpu: initialized\n"
            elif "amd-smi" in command and " list " in f" {command} ":
                output = json.dumps(_gpu_inventory())
            elif "afmctl" in command and "show device" in command:
                output = json.dumps(
                    _afm_devices(
                        phase=self.phase_by_host.get(host, "ACTIVE"),
                        vpod=self.vpod_by_host.get(host),
                    )
                )
            elif "lane_en_bitmap" in command:
                output = "\n".join(f"0000:{index + 1:02x}:00.0 {self.station_mask}" for index in range(4))
            elif "afmctl" in command and "show port" in command:
                match = re.search(r"-b\s+([0-9a-f:.,]+)", command)
                bdfs = match.group(1).split(',') if match else ["0000:01:00.1"]
                output = json.dumps(_port_inventory(bdfs, self.down_ports_by_host.get(host, set())))
            else:
                output = ""
            result[host] = output
        return result


class TestMi4xxParsers(unittest.TestCase):
    def test_amd_smi_gpu_inventory_requires_four_descriptors(self):
        gpus, errors = parse_amd_smi_gpu_json(json.dumps(_gpu_inventory()))
        self.assertEqual(len(gpus), 4)
        self.assertEqual(errors, [])

        gpus, errors = parse_amd_smi_gpu_json("not json")
        self.assertEqual(gpus, [])
        self.assertTrue(errors)

    def test_afm_device_json_preserves_accelerator_zero_and_vpod(self):
        devices, errors = parse_afmctl_device_json(json.dumps(_afm_devices()))
        self.assertEqual(errors, [])
        self.assertEqual(len(devices), 4)
        self.assertEqual(devices[0]["accelerator_id"], 0)
        self.assertEqual(devices[0]["config_phase"], "ACTIVE")
        self.assertEqual(devices[0]["virtualization_mode"], "BARE-METAL")
        self.assertEqual(devices[0]["vpod_accelerators"], [0, 1, 2, 3])

    def test_afm_device_json_accepts_live_singular_network_port_field(self):
        payload = _afm_devices()
        for device in payload['devices']:
            device.pop('num_network_ports')
            device['capability'] = {'num_network_port': '36'}
        devices, errors = parse_afmctl_device_json(json.dumps(payload))
        self.assertEqual(errors, [])
        self.assertTrue(all(device['num_network_ports'] == 36 for device in devices))

    def test_station_masks_allow_intentional_disablement_and_reject_partial(self):
        masks, errors = parse_station_masks("0000:01:00.0 f0f0f0f0f0f0f0f0f0\n")
        self.assertEqual(errors, [])
        self.assertEqual(masks["0000:01:00.0"], "f0f0f0f0f0f0f0f0f0")

        masks, errors = parse_station_masks("0000:01:00.0 f0c0f0f0f0f0f0f0f0\n")
        self.assertEqual(errors, [])
        self.assertIn("c", masks["0000:01:00.0"])


class TestMi4xxNodeHealthCheck(unittest.TestCase):
    def _check(self, phdl, **kwargs):
        return Mi4xxNodeHealthCheck(
            phdl,
            use_sudo=False,
            readiness_timeout_seconds=0,
            poll_interval_seconds=1,
            **kwargs,
        )

    def test_run_passes_when_platform_and_afm_admission_are_healthy(self):
        check = self._check(FakePssh())
        results = check.run()

        self.assertEqual(results["status"], "PASS")
        self.assertEqual(results["vpod_membership"]["status"], "PASS")
        self.assertEqual(results["vpod_membership"]["vpod_accelerators"], [0, 1, 2, 3])
        for node in ("nodeA", "nodeB"):
            self.assertEqual(results["node_results"][node]["status"], "PASS")
            self.assertEqual(len(results["node_results"][node]["afm_port_inventory"]), 4)

    def test_generic_mode_runs_only_generation_independent_gpu_checks(self):
        phdl = FakePssh()
        check = self._check(phdl, fabric_checks=False)

        results = check.run()

        self.assertEqual(results["status"], "PASS")
        self.assertFalse(results["fabric_checks"])
        self.assertEqual(results["vpod_membership"]["status"], "SKIPPED")
        commands = "\n".join(phdl.commands)
        self.assertIn("/sys/module/amdgpu", commands)
        self.assertIn("/dev/kfd", commands)
        self.assertIn("amd-smi", commands)
        self.assertNotIn("/sys/module/ifoe", commands)
        self.assertNotIn("pgrep -af", commands)
        self.assertNotIn("afmctl", commands)
        self.assertNotIn("lane_en_bitmap", commands)

    def test_expected_afm_device_count_is_derived_from_gpu_count(self):
        check = self._check(FakePssh(hosts=("nodeA",)), expected_gpus_per_node=3)

        _devices, errors = check._evaluate_afm_devices(json.dumps(_afm_devices()))

        self.assertTrue(any("Expected 3 AFM devices, found 4" in error for error in errors))

    def test_provider_phase_fails_the_admission_gate(self):
        check = self._check(FakePssh(phase_by_host={"nodeB": "PROVIDER"}))
        results = check.run()

        self.assertEqual(results["status"], "FAIL")
        self.assertEqual(results["node_results"]["nodeB"]["status"], "FAIL")
        self.assertTrue(
            any("config_phase is PROVIDER" in error for error in results["node_results"]["nodeB"]["errors"])
        )

    def test_cluster_vpod_membership_mismatch_fails(self):
        check = self._check(FakePssh(vpod_by_host={"nodeB": [4, 5, 6, 7]}))
        results = check.run()

        self.assertEqual(results["status"], "FAIL")
        self.assertTrue(any("same AFM vPOD membership" in error for error in results["vpod_membership"]["errors"]))

    def test_host_prerequisite_detects_agent_slot_and_kernel_failure(self):
        check = self._check(FakePssh(hosts=("nodeA",)), agent_slot_ids={"nodeA": 9})
        check.results = {"nodeA": {"status": "PASS", "errors": []}}
        check._evaluate_host_prerequisites(
            {
                "nodeA": {
                    "amdgpu": "1",
                    "kfd": "1",
                    "module:ifoe": "1",
                    "agent": "102 inb-node-agent --slot-id 4\n",
                    "kernel": "amdgpu: Fatal error during GPU init\n",
                }
            }
        )
        errors = check.results["nodeA"]["errors"]
        self.assertTrue(any("slot-id 9" in error for error in errors))
        self.assertTrue(any("Current-boot amdgpu initialization failures" in error for error in errors))

    def test_agent_probe_does_not_match_its_own_remote_shell_command(self):
        command = Mi4xxNodeHealthCheck._agent_command('inb-node-agent')
        self.assertIn('[i]nb\\-node\\-agent', command)
        self.assertNotIn("'inb-node-agent'", command)

    def test_station_mask_command_uses_sudo_when_configured(self):
        command = Mi4xxNodeHealthCheck(
            FakePssh(hosts=('nodeA',)), use_sudo=True, readiness_timeout_seconds=0
        ).build_station_mask_command()
        self.assertIn('sudo -n cat', command)

    def test_partial_station_is_rejected_by_default_policy(self):
        check = self._check(FakePssh(hosts=("nodeA",)))
        check.results = {"nodeA": {"status": "PASS", "errors": []}}
        check._exec_all = lambda _command: {
            "nodeA": "\n".join(
                ["0000:01:00.0 f0c0f0f0f0f0f0f0f0"] + [f"0000:{index + 1:02x}:00.0 {'f' * 18}" for index in range(1, 4)]
            )
        }
        check._evaluate_station_masks()
        self.assertEqual(check.results["nodeA"]["status"], "FAIL")
        self.assertTrue(any("partial IFoE station" in error for error in check.results["nodeA"]["errors"]))

    def test_masked_station_links_do_not_fail_the_gate(self):
        # ``afmctl`` may report a physical LINK_UP on a deliberately masked
        # station.  Only the ports belonging to ``f`` stations are admitted.
        check = self._check(FakePssh(hosts=("nodeA",), station_mask='fff000000fff000000'))
        results = check.run()

        self.assertEqual(results['status'], 'PASS')
        inventory = results['node_results']['nodeA']['afm_port_inventory']['0000:01:00.1']
        self.assertEqual(inventory['expected_enabled_ports'], 12)
        self.assertEqual(inventory['enabled_station_up_ports'], 12)
        self.assertEqual(inventory['up_ports'], 36)

    def test_down_port_in_mask_enabled_station_fails_the_gate(self):
        check = self._check(
            FakePssh(
                hosts=("nodeA",),
                station_mask='fff000000fff000000',
                down_ports_by_host={"nodeA": [0]},
            )
        )
        results = check.run()

        self.assertEqual(results['status'], 'FAIL')
        self.assertTrue(
            any(
                'enabled station 0 has non-UP port(s): 0 (DOWN)' in error
                for error in results['node_results']['nodeA']['errors']
            )
        )

    def test_port_inventory_is_batched_once_per_node(self):
        class BatchedPortPssh:
            reachable_hosts = ['nodeA', 'nodeB']

            def __init__(self):
                self.command_lists = []

            def exec_cmd_list(self, commands, **_kwargs):
                self.command_lists.append(commands)
                results = {}
                for host, command in zip(self.reachable_hosts, commands):
                    match = re.search(r"-b\s+([0-9a-f:.,]+)", command)
                    bdfs = match.group(1).split(',') if match else []
                    results[host] = json.dumps(_port_inventory(bdfs))
                return results

        phdl = BatchedPortPssh()
        check = Mi4xxNodeHealthCheck(phdl, use_sudo=False, readiness_timeout_seconds=0)
        masks = {f'0000:{index + 1:02x}:00.0': 'f' * 18 for index in range(4)}
        check.results = {
            host: {
                'status': 'PASS',
                'errors': [],
                'station_masks': dict(masks),
            }
            for host in phdl.reachable_hosts
        }

        check._cross_check_port_inventory()

        self.assertEqual(len(phdl.command_lists), 1)
        self.assertEqual(len(phdl.command_lists[0]), 2)
        for command in phdl.command_lists[0]:
            self.assertIn('-b 0000:01:00.1,0000:02:00.1,0000:03:00.1,0000:04:00.1 --json', command)
        for host in phdl.reachable_hosts:
            self.assertEqual(check.results[host]['status'], 'PASS')
            self.assertEqual(len(check.results[host]['afm_port_inventory']), 4)


class TestNodeHealthConfigContract(unittest.TestCase):
    def test_schema_accepts_only_the_three_customer_facing_options(self):
        config = PreflightConfigFile.model_validate(
            {
                "node_health": {
                    "enabled": True,
                    "gpus_per_node": 4,
                    "fabric_checks": True,
                }
            }
        )

        self.assertTrue(config.node_health.enabled)
        self.assertEqual(config.node_health.gpus_per_node, 4)
        self.assertTrue(config.node_health.fabric_checks)

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate(
                {
                    "node_health": {
                        "enabled": True,
                        "gpus_per_node": 4,
                        "fabric_checks": True,
                        "failure_mode": "report",
                    }
                }
            )

        with self.assertRaises(ValidationError):
            PreflightConfigFile.model_validate(
                {
                    "node_check": {
                        "mi4xx_node_health": {
                            "enabled": True,
                        }
                    }
                }
            )

    def test_preflight_entrypoint_wires_the_new_shape(self):
        from cvs.tests.preflight import preflight_checks

        phdl = MagicMock()
        phdl.reachable_hosts = ["nodeA"]
        config = {
            "node_health": {
                "enabled": True,
                "gpus_per_node": 8,
                "fabric_checks": False,
            }
        }
        cluster = {"node_dict": {"nodeA": {}}}
        checker_results = {
            "status": "PASS",
            "fabric_checks": False,
            "node_results": {"nodeA": {"status": "PASS", "errors": []}},
            "failed_nodes": [],
            "vpod_membership": {"status": "SKIPPED", "errors": []},
            "errors": [],
        }

        previous_results = dict(preflight_checks.preflight_results)
        try:
            with (
                patch.object(preflight_checks, "NodeHealthCheck") as checker_cls,
                patch.object(preflight_checks, "preflight_update_test_result"),
            ):
                checker_cls.return_value.run.return_value = checker_results
                preflight_checks.test_node_health(phdl, config, cluster)

            kwargs = checker_cls.call_args.kwargs
            self.assertEqual(kwargs["expected_gpus_per_node"], 8)
            self.assertFalse(kwargs["fabric_checks"])
            self.assertEqual(preflight_checks.preflight_results["node_health"]["status"], "PASS")
        finally:
            preflight_checks.preflight_results.clear()
            preflight_checks.preflight_results.update(previous_results)

    def test_generation_neutral_class_name_preserves_legacy_alias(self):
        self.assertIs(NodeHealthCheck, Mi4xxNodeHealthCheck)

    def test_report_distinguishes_generic_health_from_fabric_admission(self):
        results = {
            "status": "PASS",
            "fabric_checks": False,
            "node_results": {
                "nodeA": {
                    "status": "PASS",
                    "errors": [],
                    "gpu_inventory": _gpu_inventory()["gpus"],
                }
            },
            "failed_nodes": [],
            "vpod_membership": {"status": "SKIPPED", "errors": []},
            "coverage": {"missing_nodes": []},
        }
        generator = PreflightReportGenerator(MagicMock(), {"node_health": results}, {})

        summary = generator._summarize_node_health_results(results)
        html_report = generator._generate_node_health_html(results)

        self.assertEqual(summary["status"], "PASS")
        self.assertFalse(summary["fabric_checks"])
        self.assertIn("passed node-health admission", summary["summary"])
        self.assertIn("Node-Health Admission", html_report)
        self.assertIn("fabric checks disabled", html_report)
        self.assertNotIn("AFM devices</th>", html_report)


if __name__ == "__main__":
    unittest.main()
