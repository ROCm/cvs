import os
import shlex
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from cvs.lib.inference.xdit.pytorch_xdit_benchmark_job import (
    BenchmarkLaunchPlan,
    PytorchXditBenchmarkJob,
)


class _StubBenchmarkJob(PytorchXditBenchmarkJob):
    def validate_parallelism(self):
        return None

    def _build_env_args(self) -> str:
        return "-e STUB=1"

    def _build_torchrun_cmd(self, *, node_rank, host_output_dir, master_addr, master_port) -> str:
        return f"torchrun --node_rank={node_rank} --master_addr={master_addr}"

    def _host_output_dir(self, output_base_dir: str, hostname: str) -> str:
        return f"{output_base_dir}/stub_{hostname}_outputs"

    def _benchmark_name(self) -> str:
        return "STUB"


def _wire_phdl(phdl, hosts, hostnames=None):
    hostnames = hostnames or {host: f"host-{idx}" for idx, host in enumerate(hosts)}

    def _exec_side(cmd, timeout=None, print_console=False, detailed=False):
        text = str(cmd)
        out = {}
        for host in hosts:
            if "test -e /dev/kfd" in text:
                value = "KFD_OK"
            elif text.strip() == "hostname":
                value = hostnames[host]
            elif "docker run" in text:
                value = "benchmark ok"
                if detailed:
                    out[host] = {"output": value, "exit_code": 0}
                    continue
            elif text.startswith("mkdir -p"):
                value = ""
            else:
                value = ""
            if detailed:
                out[host] = {"output": value, "exit_code": 0}
            else:
                out[host] = value
        return out

    phdl.exec.side_effect = _exec_side
    phdl.exec_cmd_list = MagicMock(return_value={host: "" for host in hosts})


def _make_job(hosts=None, *, distributed=False, cluster_dict=None, hf_token=""):
    hosts = hosts or ["10.0.0.1"]
    phdl = MagicMock()
    phdl.host_list = list(hosts)
    _wire_phdl(phdl, hosts)

    inference_dict = {
        "container_image": "test/xdit:latest",
        "container_name": "stub-benchmark",
        "hf_home": "/home/user/.cache/huggingface",
        "output_base_dir": "/home/user/stub_output",
        "container_config": {
            "device_list": ["/dev/dri", "/dev/kfd"],
            "volume_dict": {},
            "env_dict": {},
        },
    }
    return _StubBenchmarkJob(
        phdl,
        inference_dict,
        hf_token=hf_token,
        nproc_per_node=8,
        distributed=distributed,
        cluster_dict=cluster_dict or {},
    )


def _rm_cmds(phdl):
    return [
        str(call.args[0]) for call in phdl.exec.call_args_list if call.args and str(call.args[0]).startswith("rm -f ")
    ]


def _long_token(fill):
    return "hf_" + (fill * 24)


class TestBenchmarkLaunchPlan(unittest.TestCase):
    def test_defaults(self):
        plan = BenchmarkLaunchPlan()
        self.assertEqual(plan.mkdir_cmds, [])
        self.assertEqual(plan.docker_cmds, [])
        self.assertFalse(plan.distributed)
        self.assertEqual(plan.world_size, 0)


class TestPytorchXditBenchmarkJob(unittest.TestCase):
    def test_check_kfd_all_present(self):
        job = _make_job(["10.0.0.1", "10.0.0.2"])
        self.assertEqual(job.check_kfd(), [])

    def test_build_launch_plan_single_node(self):
        job = _make_job(["10.0.0.1"])
        plan = job.build_launch_plan()
        self.assertEqual(len(plan.docker_cmds), 1)
        self.assertIn("stub_host-0_outputs", plan.output_dirs_by_node["10.0.0.1"])
        self.assertIn("docker run", plan.docker_cmds[0])
        self.assertIn("torchrun", plan.docker_cmds[0])

    def test_store_output_dir_hint_single_node(self):
        job = _make_job(["10.0.0.1"])
        plan = job.build_launch_plan()
        job.store_output_dir_hint(plan)
        self.assertIn("_test_output_dir", job.inference_dict)

    def test_run_success(self):
        job = _make_job(["10.0.0.1"])
        results, plan, errors = job.run(timeout=60)
        self.assertEqual(errors, [])
        self.assertEqual(len(plan.docker_cmds), 1)
        self.assertIn("10.0.0.1", results)

    def test_wrap_docker_cmd_chmods_and_removes_token_file(self):
        token = _long_token("d")
        job = _make_job(hf_token=token)
        path = shlex.quote(job.hf_token_path)
        wrapped = job._wrap_docker_cmd("docker run img")
        self.assertEqual(wrapped, f"chmod 600 {path} && docker run img; rm -f {path}")
        self.assertNotIn(token, wrapped)

    def test_wrap_docker_cmd_unchanged_without_token(self):
        job = _make_job()
        self.assertEqual(job._wrap_docker_cmd("docker run img"), "docker run img")

    def test_stage_uploads_token_bytes_to_remote_path(self):
        token = _long_token("f")
        job = _make_job(hf_token=token)
        captured = {}

        def _upload(local_path, remote_path):
            with open(local_path, "rb") as handle:
                captured["body"] = handle.read()
            captured["remote"] = remote_path
            captured["local_exists"] = os.path.exists(local_path)

        job.s_phdl.upload_file.side_effect = _upload
        self.assertIsNone(job._stage_hf_token())
        self.assertEqual(captured["body"], token.encode("utf-8"))
        self.assertEqual(captured["remote"], job.hf_token_path)
        self.assertTrue(captured["local_exists"])

    def test_stage_removes_token_file_when_upload_fails(self):
        token = _long_token("e")
        job = _make_job(hf_token=token)
        job.s_phdl.upload_file.side_effect = OSError("sftp failed")
        err = job._stage_hf_token()
        self.assertIn("Failed to stage HF token file", err)
        self.assertIn("sftp failed", err)
        self.assertNotIn(token, err)
        self.assertIn(f"rm -f {shlex.quote(job.hf_token_path)}", _rm_cmds(job.s_phdl))

    def test_run_removes_token_file_when_exec_fails(self):
        token = _long_token("c")
        job = _make_job(hf_token=token)
        original = job.s_phdl.exec.side_effect

        def _exec_side(cmd, timeout=None, print_console=False, detailed=False):
            if "docker run" in str(cmd):
                raise TimeoutError("ssh channel closed")
            return original(cmd, timeout=timeout, print_console=print_console, detailed=detailed)

        job.s_phdl.exec.side_effect = _exec_side
        _results, _plan, errors = job.run(timeout=30)
        self.assertTrue(errors)
        self.assertTrue(any("ssh channel closed" in err for err in errors))
        self.assertIn(f"rm -f {shlex.quote(job.hf_token_path)}", _rm_cmds(job.s_phdl))
        for call in job.s_phdl.exec.call_args_list:
            if call.args:
                self.assertNotIn(token, str(call.args[0]))

    def test_publish_rejects_remote_handle_without_upload_file(self):
        job = _make_job(hf_token=_long_token("a"))
        dest_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, dest_dir, ignore_errors=True)
        job.hf_token_path = os.path.join(dest_dir, "token")
        remote = type("RemoteHandle", (), {})()
        remote.host_list = list(job.server_nodes)
        remote.exec = MagicMock(return_value={host: "" for host in job.server_nodes})
        job.s_phdl = remote
        fd, local_path = tempfile.mkstemp()
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(local_path) and os.remove(local_path))
        with self.assertRaises(RuntimeError) as ctx:
            job._publish_hf_token_file(local_path)
        self.assertIn("upload_file", str(ctx.exception))
        self.assertFalse(os.path.exists(job.hf_token_path))

    def test_stage_cleans_up_when_remote_handle_has_no_upload_file(self):
        token = _long_token("b")
        job = _make_job(hf_token=token)
        dest_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, dest_dir, ignore_errors=True)
        job.hf_token_path = os.path.join(dest_dir, "token")
        remote = type("RemoteHandle", (), {})()
        remote.host_list = list(job.server_nodes)
        remote.exec = MagicMock(return_value={host: "" for host in job.server_nodes})
        job.s_phdl = remote
        err = job._stage_hf_token()
        self.assertIn("upload_file", err)
        self.assertNotIn(token, err)
        self.assertFalse(os.path.exists(job.hf_token_path))
        self.assertIn(f"rm -f {shlex.quote(job.hf_token_path)}", _rm_cmds(remote))

    def test_publish_copies_file_for_local_pssh(self):
        job = _make_job(hf_token="hf_local_only")
        dest_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, dest_dir, ignore_errors=True)
        job.hf_token_path = os.path.join(dest_dir, "token")
        job.s_phdl = type("LocalPssh", (), {"host_list": ["127.0.0.1"]})()
        fd, local_path = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"hf_local_only")
        self.addCleanup(lambda: os.path.exists(local_path) and os.remove(local_path))
        job._publish_hf_token_file(local_path)
        with open(job.hf_token_path, "rb") as handle:
            self.assertEqual(handle.read(), b"hf_local_only")
        if os.name == "posix":
            self.assertEqual(os.stat(job.hf_token_path).st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
