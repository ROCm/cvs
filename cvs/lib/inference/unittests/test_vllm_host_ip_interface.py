'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for server_params.host_ip_interface: config validation, the
per-node VLLM_HOST_IP export in the server env script, and Ray bootstrap
node addressing.
'''

import unittest

from pydantic import ValidationError

from cvs.lib.inference.utils.vllm_config_loader import VariantConfig
from cvs.lib.inference.vllm_job import VllmJob

HEAD = "10.0.0.1"
WORKER = "10.0.0.2"


class _Orch:
    """Records every exec call; `responder(cmd, hosts, detailed)` controls returns."""

    def __init__(self, hosts, responder=None):
        self.hosts = list(hosts)
        self.calls = []
        self._responder = responder

    def exec(self, cmd, hosts=None, detailed=False, **kwargs):
        self.calls.append((cmd, hosts))
        if self._responder is not None:
            return self._responder(cmd, hosts, detailed)
        return {}

    def exec_on_head(self, cmd, *args, **kwargs):
        return {}


def _resolving_responder(ips):
    """Answer the VLLM_HOST_IP probe with `ips[host]`; ray bootstrap succeeds."""

    def r(cmd, hosts, detailed):
        targets = hosts if hosts is not None else list(ips)
        if detailed:
            return {h: {"exit_code": 0, "output": "", "stdout": ""} for h in targets}
        if "source /tmp/server_env_script.sh" in cmd and "printf" in cmd:
            return {h: ips.get(h, "") for h in targets}
        return {h: "" for h in targets}

    return r


def _variant(pp=2, backend="ray", **server_extra):
    cell = f"ISL=1024,OSL=1024,TP=8,PP={pp},CONC=16"
    return VariantConfig(
        enforce_thresholds=False,
        threshold_json="threshold.json",
        paths={"shared_fs": "/logs", "models_dir": "/models", "log_dir": "/logs", "hf_token_file": "/logs/.hf"},
        container={"name": "test", "image": "test", "env": {}, "runtime": {"name": "docker", "args": {}}},
        server_params={
            "model": "/models/test-model",
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": pp,
            "distributed_executor_backend": backend,
            **server_extra,
        },
        sweeps={cell: {}},
        runs=[cell],
    )


def _job(orch, **server_extra):
    pp = 2 if len(orch.hosts) > 1 else 1
    return VllmJob(
        orch=orch,
        variant=_variant(pp=pp, **server_extra),
        hf_token="tok",
        isl="1024",
        osl="1024",
        concurrency=16,
        num_prompts="160",
    )


def _env_script_cmd(orch):
    return next(cmd for cmd, _ in orch.calls if "/tmp/server_env_script.sh" in cmd and "printf" in cmd)


def _ray_starts(orch, host):
    return [cmd for cmd, hosts in orch.calls if hosts == [host] and "ray start" in cmd]


class TestHostIpInterfaceConfig(unittest.TestCase):
    def test_defaults_to_unset(self):
        self.assertIsNone(_variant().server_params.host_ip_interface)

    def test_accepts_interface_name_and_keeps_it_out_of_serve_options(self):
        params = _variant(host_ip_interface="ens3").server_params
        self.assertEqual(params.host_ip_interface, "ens3")
        self.assertNotIn("host_ip_interface", params.extra_options())

    def test_rejects_invalid_interface_names(self):
        for name in ("", "ens3; reboot", "$(id)", "eth 0", "a" * 16):
            with self.subTest(name=name):
                with self.assertRaises(ValidationError):
                    _variant(host_ip_interface=name)


class TestHostIpExport(unittest.TestCase):
    def test_unset_writes_no_host_ip_and_skips_probe(self):
        orch = _Orch([HEAD, WORKER])
        _job(orch).build_server_cmd()
        self.assertTrue(all("VLLM_HOST_IP" not in cmd for cmd, _ in orch.calls))

    def test_set_exports_host_ip_from_named_interface(self):
        orch = _Orch([HEAD, WORKER], _resolving_responder({HEAD: "10.0.0.1", WORKER: "10.0.0.2"}))
        _job(orch, host_ip_interface="ens3").build_server_cmd()
        script_cmd = _env_script_cmd(orch)
        self.assertIn("export VLLM_HOST_IP=", script_cmd)
        self.assertIn("ens3", script_cmd)

    def test_set_probes_every_host_after_writing_script(self):
        orch = _Orch([HEAD, WORKER], _resolving_responder({HEAD: "10.0.0.1", WORKER: "10.0.0.2"}))
        _job(orch, host_ip_interface="ens3").build_server_cmd()
        cmds = [cmd for cmd, _ in orch.calls]
        write_idx = cmds.index(_env_script_cmd(orch))
        probe_idx = next(i for i, cmd in enumerate(cmds) if "source /tmp/server_env_script.sh" in cmd)
        self.assertGreater(probe_idx, write_idx)
        self.assertIsNone(orch.calls[probe_idx][1], "probe must broadcast to every host")

    def test_set_raises_when_a_host_has_no_ipv4_address(self):
        for bad in ("", "Traceback (most recent call last)", "fe80::1"):
            with self.subTest(bad=bad):
                orch = _Orch([HEAD, WORKER], _resolving_responder({HEAD: "10.0.0.1", WORKER: bad}))
                with self.assertRaisesRegex(RuntimeError, rf"ens3.*{WORKER}"):
                    _job(orch, host_ip_interface="ens3").build_server_cmd()

    def test_set_raises_when_a_host_does_not_answer(self):
        orch = _Orch([HEAD, WORKER], _resolving_responder({HEAD: "10.0.0.1"}))
        with self.assertRaisesRegex(RuntimeError, WORKER):
            _job(orch, host_ip_interface="ens3").build_server_cmd()

    def test_serve_argv_has_no_host_ip_flag(self):
        orch = _Orch([HEAD, WORKER])
        argv = _job(orch, host_ip_interface="ens3")._server_argv(0)
        self.assertFalse([a for a in argv if "host-ip" in a or "host_ip" in a])


class TestRayBootstrapNodeAddress(unittest.TestCase):
    def test_unset_keeps_legacy_ray_start_commands(self):
        orch = _Orch([HEAD, WORKER], _resolving_responder({}))
        _job(orch).start_server()
        self.assertEqual(_ray_starts(orch, HEAD), ["ray start --head --port=29501"])
        self.assertEqual(_ray_starts(orch, WORKER), [f"ray start --address={HEAD}:29501"])

    def test_set_pins_ray_node_ip_on_every_node(self):
        orch = _Orch([HEAD, WORKER], _resolving_responder({}))
        _job(orch, host_ip_interface="ens3").start_server()
        for host, token in ((HEAD, "--head --port=29501"), (WORKER, f"--address={HEAD}:29501")):
            with self.subTest(host=host):
                (cmd,) = _ray_starts(orch, host)
                self.assertIn("source /tmp/server_env_script.sh", cmd)
                self.assertIn(token, cmd)
                self.assertIn('--node-ip-address="$VLLM_HOST_IP"', cmd)


if __name__ == "__main__":
    unittest.main()
