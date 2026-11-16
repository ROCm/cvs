'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/orchestrators/baremetal.py: BaremetalOrchestrator construction
# and command-dispatch surface (exec, exec_cmd_list, exec_on_head, cleanup, sudo_prefix) used by the
# migrated rvs_cvs.py orch fixture. Mocks Pssh so tests run with no SSH.

import os
import shlex
import subprocess  # nosec B404
import unittest
import tempfile
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.factory import OrchestratorConfig
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.lib.parallel.config import ParallelConfig
from cvs.unittests.support import FAKE_CRED, TMP


def _make_orch_config():
    """Minimal OrchestratorConfig that satisfies BaremetalOrchestrator.__init__
    without touching disk or SSH."""
    return OrchestratorConfig(
        orchestrator="baremetal",
        node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
        username="testuser",
        priv_key_file="/dev/null",
        password=None,
        head_node_dict={"mgmt_ip": "10.0.0.1"},
        container={},
    )


def _run_in_sh(cmd, **kwargs):
    proc = subprocess.run(["sh", "-c", cmd], capture_output=True, text=True, check=False)  # nosec B603 B607
    return {"10.0.0.1": {"output": proc.stdout, "exit_code": proc.returncode}}


class TestBaremetalOrchestrator(unittest.TestCase):
    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_init_constructs_pssh_handles(self, mock_pssh):
        BaremetalOrchestrator(MagicMock(), _make_orch_config())
        # __init__ creates two Pssh handles: self.head and self.all.
        self.assertEqual(mock_pssh.call_count, 2)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_init_forwards_cluster_env_vars(self, mock_pssh):
        cfg = _make_orch_config()
        cfg.env_vars = {"PATH": "/opt/rocm/bin", "LD_LIBRARY_PATH": "/opt/rocm/lib"}
        BaremetalOrchestrator(MagicMock(), cfg)
        for call in mock_pssh.call_args_list:
            self.assertEqual(call.kwargs.get("env_vars"), cfg.env_vars)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_parallel_handle_settings_apply_to_both_handles(self, mock_pssh):
        cfg = _make_orch_config()
        cfg.parallel_handle = {
            "config": {"hosts_per_shard": 8},
            "transport_kwargs": {"timeout": 60, "num_retries": 2, "retry_delay": 2},
        }
        BaremetalOrchestrator(MagicMock(), cfg)
        for call in mock_pssh.call_args_list:
            self.assertIsInstance(call.kwargs["config"], ParallelConfig)
            self.assertEqual(call.kwargs["config"].hosts_per_shard, 8)
            self.assertEqual(call.kwargs["timeout"], 60)
            self.assertEqual(call.kwargs["num_retries"], 2)
            self.assertEqual(call.kwargs["retry_delay"], 2)
            self.assertEqual(call.kwargs["transport"], "ssh")

    @patch("cvs.core.orchestrators.baremetal.is_managed_compute", return_value=True)
    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_parallel_handle_transport_kwargs_are_forwarded_on_http(self, mock_pssh, _managed):
        cfg = _make_orch_config()
        cfg.parallel_handle = {
            "config": {"hosts_per_shard": 8},
            "transport_kwargs": {"timeout": 60, "num_retries": 2, "retry_delay": 2},
        }
        BaremetalOrchestrator(MagicMock(), cfg)
        for call in mock_pssh.call_args_list:
            self.assertEqual(call.kwargs["transport"], "http")
            self.assertEqual(call.kwargs["config"].hosts_per_shard, 8)
            self.assertEqual(call.kwargs["timeout"], 60)
            self.assertEqual(call.kwargs["num_retries"], 2)
            self.assertEqual(call.kwargs["retry_delay"], 2)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_init_sets_orchestrator_type(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        self.assertEqual(orch.orchestrator_type, "baremetal")

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_init_picks_first_node_as_head(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        # _make_orch_config inserts 10.0.0.1 first.
        self.assertEqual(orch.head_node, "10.0.0.1")

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_delegates_to_all_when_targeting_full_set(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.all = MagicMock()
        orch.all.exec.return_value = {"10.0.0.1": "ok", "10.0.0.2": "ok"}
        result = orch.exec("ls", timeout=5)
        orch.all.exec.assert_called_once_with(
            "ls", timeout=5, detailed=False, print_console=True, inactivity_timeout=None
        )
        self.assertEqual(result, {"10.0.0.1": "ok", "10.0.0.2": "ok"})

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_cmd_list_delegates_to_all(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.all = MagicMock()
        result = orch.exec_cmd_list(["echo a", "echo b"], timeout=7, print_console=False)
        orch.all.exec_cmd_list.assert_called_once_with(["echo a", "echo b"], timeout=7, print_console=False)
        self.assertIs(result, orch.all.exec_cmd_list.return_value)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_cmd_list_defaults_match_all_handle(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.all = MagicMock()
        orch.exec_cmd_list(["echo a", "echo b"])
        orch.all.exec_cmd_list.assert_called_once_with(["echo a", "echo b"], timeout=None, print_console=True)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_distribute_using_mpi_logs_masked_command(self, _mock_pssh):
        log = MagicMock()
        orch = BaremetalOrchestrator(log, _make_orch_config())
        cmd = f"mpirun -x HF_TOKEN={FAKE_CRED} hostname"
        with patch.object(orch, "build_mpi_cmd", return_value=cmd), patch.object(orch, "exec_on_head") as run:
            orch.distribute_using_mpi("hostname", ["10.0.0.1"], 1, {"HF_TOKEN": FAKE_CRED}, "/opt/ompi")
        run.assert_called_once_with(cmd, timeout=500)
        logged = [c.args[0] for c in log.debug.call_args_list if str(c.args[0]).startswith("MPI command: ")]
        self.assertEqual(logged, ["MPI command: mpirun -x HF_TOKEN=<redacted> hostname"])

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_on_head_delegates_to_head_handle(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.head = MagicMock()
        orch.head.exec.return_value = {"10.0.0.1": "ok"}
        result = orch.exec_on_head("hostname", timeout=10)
        orch.head.exec.assert_called_once_with("hostname", timeout=10, detailed=False, print_console=True)
        self.assertEqual(result, {"10.0.0.1": "ok"})

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_download_file_delegates_to_host_transport(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.all = MagicMock()
        orch.all.download_file.return_value = {"10.0.0.2": f"{TMP}/result.log"}
        result = orch.download_file("/remote/result.log", f"{TMP}/result.log", hosts=["10.0.0.2"])
        orch.all.download_file.assert_called_once_with("/remote/result.log", f"{TMP}/result.log", hosts=["10.0.0.2"])
        self.assertEqual(result, {"10.0.0.2": f"{TMP}/result.log"})

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_on_host_targets_requested_host_subset(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        with patch.object(orch, "exec", return_value={"10.0.0.2": "ok"}) as exec_mock:
            result = orch.exec_on_host("date", hosts=["10.0.0.2"], timeout=5)

        exec_mock.assert_called_once_with(
            "date",
            hosts=["10.0.0.2"],
            timeout=5,
            detailed=False,
            print_console=True,
        )
        self.assertEqual(result, {"10.0.0.2": "ok"})

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_head_file_transfers_delegate_to_head_handle(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.head = MagicMock()
        orch.head.upload_file.return_value = {"10.0.0.1": "/remote/result.json"}
        orch.head.download_file.return_value = {"10.0.0.1": "/local/result.json"}

        uploaded = orch.upload_to_head("/local/result.json", "/remote/result.json")
        downloaded = orch.download_from_head("/remote/result.json", "/local/result.json")

        orch.head.upload_file.assert_called_once_with("/local/result.json", "/remote/result.json")
        orch.head.download_file.assert_called_once_with("/remote/result.json", "/local/result.json")
        self.assertEqual(uploaded, {"10.0.0.1": "/remote/result.json"})
        self.assertEqual(downloaded, {"10.0.0.1": "/local/result.json"})

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_forwards_print_console_false_to_all(self, _mock_pssh):
        """print_console=False must reach the pssh handle, not be swallowed here.

        A dropped kwarg is silent -- the command still works, it just logs
        hundreds of MB -- so this is pinned explicitly.
        """
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.all = MagicMock()
        orch.exec("cat /tmp/huge", print_console=False)
        self.assertIs(orch.all.exec.call_args.kwargs["print_console"], False)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_forwards_print_console_false_to_host_subset(self, mock_pssh):
        """The subset branch builds its own Pssh; it must forward too.

        orch.all is stubbed with a distinct mock so that falling through to the
        all-hosts branch would fail this test rather than silently satisfy it
        -- both handles would otherwise be the same patched Pssh return value.
        """
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.all = MagicMock()
        mock_pssh.reset_mock()
        orch.exec("cat /tmp/huge", hosts=["10.0.0.2"], print_console=False)
        # A subset handle was constructed for exactly the requested host...
        mock_pssh.assert_called_once()
        self.assertEqual(mock_pssh.call_args.args[1], ["10.0.0.2"])
        # ...the all-hosts handle was bypassed...
        orch.all.exec.assert_not_called()
        # ...and the kwarg reached the subset handle.
        subset_handle = mock_pssh.return_value
        self.assertIs(subset_handle.exec.call_args.kwargs["print_console"], False)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_forwards_inactivity_timeout_to_all(self, _mock_pssh):
        """inactivity_timeout must reach the pssh handle (ANC relies on it: abort a
        group only after N seconds with no new output, no total wall-clock cap)."""
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.all = MagicMock()
        orch.exec("sudo ./anc.py -g hbm_lvl1", inactivity_timeout=900)
        self.assertEqual(orch.all.exec.call_args.kwargs["inactivity_timeout"], 900)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_forwards_inactivity_timeout_to_host_subset(self, mock_pssh):
        """The subset branch builds its own Pssh; inactivity_timeout must forward too."""
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.all = MagicMock()
        mock_pssh.reset_mock()
        orch.exec("sudo ./anc.py -g hbm_lvl1", hosts=["10.0.0.2"], inactivity_timeout=900)
        orch.all.exec.assert_not_called()
        subset_handle = mock_pssh.return_value
        self.assertEqual(subset_handle.exec.call_args.kwargs["inactivity_timeout"], 900)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_on_head_forwards_print_console_false(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.head = MagicMock()
        orch.exec_on_head("cat /tmp/huge", print_console=False)
        self.assertIs(orch.head.exec.call_args.kwargs["print_console"], False)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_cleanup_returns_true(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        self.assertTrue(orch.cleanup(orch.hosts))

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_build_mpi_cmd_creates_private_hostfile(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        env_patch = patch.dict(os.environ, {"TMPDIR": temp.name})
        env_patch.start()
        self.addCleanup(env_patch.stop)

        with patch.object(orch, "exec_on_head", side_effect=_run_in_sh):
            cmd = orch.build_mpi_cmd("echo hi", ["10.0.0.1", "10.0.0.2"], 8, {}, "/opt/mpi")
            self.assertEqual(len(orch._mpi_hostfiles), 1)
            hostfile = orch._mpi_hostfiles[0]
            self.assertEqual(os.path.dirname(hostfile), temp.name)
            self.assertTrue(os.path.basename(hostfile).startswith("mpi_hosts_"))
            self.assertEqual(os.stat(hostfile).st_mode & 0o777, 0o600)
            with open(hostfile) as stream:
                self.assertEqual(stream.read(), "10.0.0.1 slots=8\n10.0.0.2 slots=8\n")
            tokens = shlex.split(cmd)
            self.assertEqual(tokens[tokens.index("--hostfile") + 1], hostfile)
            self.assertTrue(orch.release_mpi_hostfiles())
            self.assertFalse(os.path.exists(hostfile))

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_build_mpi_cmd_writes_host_names_verbatim(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        env_patch = patch.dict(os.environ, {"TMPDIR": temp.name})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        marker = os.path.join(temp.name, "injected")
        hosts = ["a'b", "$(id -un)", chr(96) + "id -un" + chr(96), f"x'; touch {marker}; echo '"]

        with patch.object(orch, "exec_on_head", side_effect=_run_in_sh):
            for host in hosts:
                with self.subTest(host=host):
                    orch.build_mpi_cmd("echo hi", [host], 1, {}, "/opt/mpi")
                    with open(orch._mpi_hostfiles[-1]) as stream:
                        self.assertEqual(stream.read(), f"{host} slots=1\n")
                    self.assertFalse(os.path.exists(marker))
            self.assertTrue(orch.release_mpi_hostfiles())

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_build_mpi_cmd_hostfile_does_not_use_sudo(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.head = MagicMock()
        orch.head.exec.return_value = {"10.0.0.1": {"output": "/tmp/mpi_hosts_abc123", "exit_code": 0}}
        with patch.object(orch, "sudo_prefix", return_value="sudo -n ") as sudo_prefix:
            orch.build_mpi_cmd("echo hi", ["10.0.0.1"], 1, {}, "/opt/mpi")
        creation_cmd = orch.head.exec.call_args.args[0]
        self.assertIn("umask 077", creation_cmd)
        self.assertIn("mktemp", creation_cmd)
        self.assertNotIn("sudo", creation_cmd)
        sudo_prefix.assert_not_called()

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_build_mpi_cmd_rejects_empty_host_list(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.head = MagicMock()

        with self.assertRaises(ValueError):
            orch.build_mpi_cmd(
                rank_cmd="echo hi",
                mpi_hosts=[],
                ranks_per_host=1,
                env_vars={},
                mpi_install_dir="/opt/mpi",
            )
        orch.head.exec.assert_not_called()

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_build_mpi_cmd_rejects_bad_hostfile_path(self, _mock_pssh):
        valid = {"output": "/tmp/mpi_hosts_abc", "exit_code": 0}
        cases = {
            "nonzero_exit": {"10.0.0.1": {**valid, "exit_code": 1}},
            "missing_exit": {"10.0.0.1": {"output": valid["output"]}},
            "empty_output": {"10.0.0.1": {"output": "", "exit_code": 0}},
            "relative_path": {"10.0.0.1": {"output": "mpi_hosts_abc", "exit_code": 0}},
            "wrong_prefix": {"10.0.0.1": {"output": "/tmp/other_abc", "exit_code": 0}},
            "two_lines": {"10.0.0.1": {"output": "/tmp/mpi_hosts_a\n/tmp/mpi_hosts_b", "exit_code": 0}},
            "empty_result": {},
            "missing_head": {"10.0.0.9": valid},
        }
        for label, result in cases.items():
            with self.subTest(label=label):
                orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
                with patch.object(orch, "exec_on_head", return_value=result):
                    with self.assertRaises(RuntimeError):
                        orch.build_mpi_cmd("echo hi", ["10.0.0.1"], 1, {}, "/opt/mpi")
                self.assertEqual(orch._mpi_hostfiles, [])

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_build_mpi_cmd_fails_if_any_host_fails(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        result = {
            "10.0.0.1": {"output": "/tmp/mpi_hosts_abc", "exit_code": 0},
            "10.0.0.2": {"output": "", "exit_code": 1},
        }
        with patch.object(orch, "exec_on_head", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "10.0.0.2"):
                orch.build_mpi_cmd("echo hi", ["10.0.0.1", "10.0.0.2"], 1, {}, "/opt/mpi")
        self.assertEqual(orch._mpi_hostfiles, [])

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_build_mpi_cmd_raises_on_hostfile_write_failure(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        result = {"10.0.0.1": {"output": "/tmp/mpi_hosts_abc", "exit_code": 1}}
        with patch.object(orch, "exec_on_head", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "10.0.0.1"):
                orch.build_mpi_cmd("echo hi", ["10.0.0.1"], 1, {}, "/opt/mpi")
        self.assertEqual(orch._mpi_hostfiles, [])

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_build_mpi_cmd_quotes_hostfile_path(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        hostfile = "/tmp/my dir/mpi_hosts_abc"
        result = {"10.0.0.1": {"output": hostfile, "exit_code": 0}}
        with patch.object(orch, "exec_on_head", return_value=result):
            cmd = orch.build_mpi_cmd("echo hi", ["10.0.0.1"], 1, {}, "/opt/mpi")
        self.assertIn("--hostfile '/tmp/my dir/mpi_hosts_abc'", cmd)
        tokens = shlex.split(cmd)
        self.assertEqual(tokens[tokens.index("--hostfile") + 1], hostfile)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_release_mpi_hostfiles_removes_tracked_files(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch._mpi_hostfiles = ["/tmp/mpi_hosts_a b"]
        result = {"10.0.0.1": {"output": "", "exit_code": 0}}
        with patch.object(orch, "sudo_prefix", return_value="sudo -n ") as sudo_prefix:
            with patch.object(orch, "exec_on_head", return_value=result) as run:
                self.assertTrue(orch.release_mpi_hostfiles())
        run.assert_called_once_with("rm -f -- '/tmp/mpi_hosts_a b'", timeout=30, detailed=True)
        sudo_prefix.assert_not_called()
        self.assertEqual(orch._mpi_hostfiles, [])

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_release_mpi_hostfiles_keeps_path_on_failure(self, _mock_pssh):
        failure = {"10.0.0.1": {"output": "", "exit_code": 1}}
        for label, response in [("nonzero_exit", failure), ("exception", RuntimeError("offline"))]:
            with self.subTest(label=label):
                log = MagicMock()
                orch = BaremetalOrchestrator(log, _make_orch_config())
                orch._mpi_hostfiles = ["/tmp/mpi_hosts_old"]
                with patch.object(orch, "exec_on_head") as run:
                    if isinstance(response, Exception):
                        run.side_effect = response
                    else:
                        run.return_value = response
                    self.assertFalse(orch.release_mpi_hostfiles())
                self.assertEqual(orch._mpi_hostfiles, ["/tmp/mpi_hosts_old"])
                log.warning.assert_called_once()

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_build_mpi_cmd_retries_leftover_hostfile(self, _mock_pssh):
        old_path = "/tmp/mpi_hosts_old"
        new_path = "/tmp/mpi_hosts_new"
        new_result = {"10.0.0.1": {"output": new_path, "exit_code": 0}}
        for label, removal_exit in [("success", 0), ("failure", 1)]:
            with self.subTest(label=label):
                orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
                orch._mpi_hostfiles = [old_path]
                remove_result = {"10.0.0.1": {"output": "", "exit_code": removal_exit}}
                with patch.object(orch, "exec_on_head", side_effect=[remove_result, new_result]) as run:
                    orch.build_mpi_cmd("echo hi", ["10.0.0.1"], 1, {}, "/opt/mpi")
                self.assertEqual(run.call_args_list[0].args[0], "rm -f -- /tmp/mpi_hosts_old")
                self.assertEqual(orch._mpi_hostfiles, [new_path] if removal_exit == 0 else [old_path, new_path])

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_distribute_using_mpi_releases_hostfile_on_success_and_failure(self, _mock_pssh):
        for should_fail in (False, True):
            with self.subTest(should_fail=should_fail):
                orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
                result = {"10.0.0.1": {"output": "", "exit_code": 0}}

                def build(*_args):
                    orch._mpi_hostfiles.append("/tmp/mpi_hosts_x")
                    return "mpirun x"

                def run(cmd, **kwargs):
                    if cmd.startswith("rm "):
                        return result
                    if should_fail:
                        raise RuntimeError("mpirun failed")
                    return result

                with patch.object(orch, "build_mpi_cmd", side_effect=build):
                    with patch.object(orch, "exec_on_head", side_effect=run) as exec_on_head:
                        if should_fail:
                            with self.assertRaisesRegex(RuntimeError, "mpirun failed"):
                                orch.distribute_using_mpi("hostname", ["10.0.0.1"], 1, {}, "/opt/mpi")
                        else:
                            self.assertEqual(
                                orch.distribute_using_mpi("hostname", ["10.0.0.1"], 1, {}, "/opt/mpi"), result
                            )
                self.assertEqual(
                    [call.args[0] for call in exec_on_head.call_args_list], ["mpirun x", "rm -f -- /tmp/mpi_hosts_x"]
                )
                self.assertEqual(orch._mpi_hostfiles, [])

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_distribute_using_mpi_build_failure_runs_no_mpirun(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        with patch.object(orch, "build_mpi_cmd", side_effect=RuntimeError("creation failed")):
            with patch.object(orch, "exec_on_head") as run:
                with patch.object(orch, "release_mpi_hostfiles", wraps=orch.release_mpi_hostfiles) as release:
                    with self.assertRaisesRegex(RuntimeError, "creation failed"):
                        orch.distribute_using_mpi("hostname", ["10.0.0.1"], 1, {}, "/opt/mpi")
        run.assert_not_called()
        release.assert_called_once_with()
        self.assertEqual(orch._mpi_hostfiles, [])


class TestBaremetalOrchestratorSudoPrefix(unittest.TestCase):
    """Covers BaremetalOrchestrator.sudo_prefix(): the probe-once mechanism that
    replaced with_sudo_fallback's per-command `cmd || sudo -n cmd` retry."""

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_sudo_prefix_returns_sudo_prefix_when_passwordless_sudo_available(self, mock_pssh):
        pssh_instance = MagicMock()
        pssh_instance.exec.return_value = {"10.0.0.1": "0", "10.0.0.2": "0"}
        mock_pssh.return_value = pssh_instance

        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())

        self.assertEqual(orch.sudo_prefix(), "sudo -n ")

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_sudo_prefix_returns_empty_when_sudo_unavailable(self, mock_pssh):
        pssh_instance = MagicMock()
        pssh_instance.exec.return_value = {"10.0.0.1": "1", "10.0.0.2": "1"}
        mock_pssh.return_value = pssh_instance

        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())

        self.assertEqual(orch.sudo_prefix(), "")

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_sudo_prefix_warns_on_host_disagreement_but_still_returns_a_value(self, mock_pssh):
        # AC: hosts disagreeing on sudo need must log a warning but must NOT
        # raise or branch per-host -- the fleet-wide answer is the head node's
        # (10.0.0.1 per _make_orch_config), not an arbitrary dict-order pick.
        pssh_instance = MagicMock()
        pssh_instance.exec.return_value = {"10.0.0.1": "0", "10.0.0.2": "1"}
        mock_pssh.return_value = pssh_instance

        log = MagicMock()
        orch = BaremetalOrchestrator(log, _make_orch_config())

        self.assertEqual(orch.sudo_prefix(), "sudo -n ")
        log.warning.assert_called_once()

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_sudo_prefix_disagreement_uses_head_node_not_dict_order(self, mock_pssh):
        # Regression test: the fleet-wide answer must come specifically from
        # the head node, not from whichever host the probe dict happens to
        # iterate first. Here the head node (10.0.0.1) says sudo is NOT
        # needed while a non-head worker says it IS -- if this incorrectly
        # picked "first in dict order" or "any True wins", the result would
        # flip to 'sudo -n ' and privileged commands on the head node (e.g.
        # exec_on_head, used by build_mpi_cmd) would run under an
        # unnecessary and potentially unavailable sudo.
        pssh_instance = MagicMock()
        pssh_instance.exec.return_value = {"10.0.0.1": "1", "10.0.0.2": "0"}
        mock_pssh.return_value = pssh_instance

        log = MagicMock()
        orch = BaremetalOrchestrator(log, _make_orch_config())

        self.assertEqual(orch.sudo_prefix(), "")
        log.warning.assert_called_once()

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_sudo_prefix_probes_at_most_once_across_multiple_calls(self, mock_pssh):
        # Regression test for the bug being fixed: the passwordless-sudo probe
        # must fire ONCE per orchestrator instance for its whole lifetime, no
        # matter how many times sudo_prefix() is subsequently called.
        pssh_instance = MagicMock()
        pssh_instance.exec.return_value = {"10.0.0.1": "0", "10.0.0.2": "0"}
        mock_pssh.return_value = pssh_instance

        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())

        results = [orch.sudo_prefix() for _ in range(3)]

        self.assertEqual(results, ["sudo -n "] * 3)
        pssh_instance.exec.assert_called_once_with("sudo -n true >/dev/null 2>&1; echo $?")


class TestBaremetalOrchestratorSubsetHandleCleanup(unittest.TestCase):
    """The subset branch builds a throwaway Pssh; it must be destroyed.

    Left to refcounting, a call whose exec timed out keeps its SSH session
    open on the target host, so a polling suite accumulates sessions until
    sshd's limit is hit.
    """

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_destroys_subset_handle(self, mock_pssh):
        # The timeout path is the one that leaks, so cleanup must not depend
        # on a clean return.
        for label, side_effect in (("returns", None), ("raises", RuntimeError("timed out"))):
            with self.subTest(label):
                orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
                orch.all = MagicMock()
                mock_pssh.reset_mock()
                mock_pssh.return_value.exec.side_effect = side_effect

                if side_effect is None:
                    orch.exec("hostname", hosts=["10.0.0.2"])
                else:
                    with self.assertRaises(RuntimeError):
                        orch.exec("sleep 300", hosts=["10.0.0.2"], timeout=1)

                mock_pssh.return_value.destroy_clients.assert_called_once_with()

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_setup_env_destroys_subset_handle(self, mock_pssh):
        # Same subset branch as exec(); it has no callers today, but it is an
        # abstractmethod on the base class, so an implementation could reach it.
        for label, side_effect in (("returns", None), ("raises", RuntimeError("timed out"))):
            with self.subTest(label):
                orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
                orch.all = MagicMock()
                mock_pssh.reset_mock()
                if side_effect is None:
                    mock_pssh.return_value.exec.return_value = {"10.0.0.2": {"exit_code": 0}}
                    mock_pssh.return_value.exec.side_effect = None
                    orch.setup_env(["10.0.0.2"], env_script=f"{TMP}/env.sh")
                else:
                    mock_pssh.return_value.exec.side_effect = side_effect
                    with self.assertRaises(RuntimeError):
                        orch.setup_env(["10.0.0.2"], env_script=f"{TMP}/env.sh")

                mock_pssh.return_value.destroy_clients.assert_called_once_with()

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_exec_does_not_destroy_shared_all_handle(self, _mock_pssh):
        # self.all is long-lived and reused; tearing it down would break
        # every later call.
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.all = MagicMock()

        orch.exec("hostname")

        orch.all.destroy_clients.assert_not_called()


class TestBaremetalOrchestratorHttpTransport(unittest.TestCase):
    """Managed jobs use HTTP when the cluster file has agent endpoints."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.token_file = f"{self.temp_dir.name}/secret"
        with open(self.token_file, "w", encoding="utf-8") as stream:
            stream.write("tok\n")
        self.config = _make_orch_config()
        self.config.node_dict = {
            "10.0.0.1": {"agent_port": 9000},
            "10.0.0.2": {"agent_port": 9001},
        }
        self.config.agent_token_file = self.token_file
        self.expected_ports = {"10.0.0.1": 9000, "10.0.0.2": 9001}

    @patch("cvs.core.orchestrators.baremetal.is_managed_compute", return_value=True)
    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_init_uses_http_transport_when_agent_config_present(self, mock_pssh, _managed):
        BaremetalOrchestrator(MagicMock(), self.config)
        self.assertEqual(mock_pssh.call_count, 2)
        for call in mock_pssh.call_args_list:
            self.assertEqual(call.kwargs["transport"], "http")
            self.assertEqual(call.kwargs["token_file"], self.token_file)
            self.assertEqual(call.kwargs["agent_port_map"], self.expected_ports)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_unmanaged_stays_on_ssh_even_with_agent_config(self, mock_pssh):
        BaremetalOrchestrator(MagicMock(), self.config)
        for call in mock_pssh.call_args_list:
            self.assertNotIn("token_file", call.kwargs)
            self.assertEqual(call.kwargs.get("transport", "ssh"), "ssh")

    @patch("cvs.core.orchestrators.baremetal.is_managed_compute", return_value=True)
    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_subset_exec_forwards_http_kwargs(self, mock_pssh, _managed):
        orch = BaremetalOrchestrator(MagicMock(), self.config)
        orch.all = MagicMock()
        mock_pssh.reset_mock()
        orch.exec("hostname", hosts=["10.0.0.2"])
        mock_pssh.assert_called_once()
        self.assertEqual(mock_pssh.call_args.args[1], ["10.0.0.2"])
        self.assertEqual(mock_pssh.call_args.kwargs["transport"], "http")
        self.assertEqual(mock_pssh.call_args.kwargs["token_file"], self.token_file)
        self.assertEqual(mock_pssh.call_args.kwargs["agent_port_map"], self.expected_ports)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_close_destroys_both_persistent_handles(self, mock_pssh):
        mock_pssh.side_effect = lambda *args, **kwargs: MagicMock()
        orch = BaremetalOrchestrator(MagicMock(), self.config)
        orch.close()
        orch.head.destroy_clients.assert_called_once_with()
        orch.all.destroy_clients.assert_called_once_with()

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_close_survives_a_handle_that_raises(self, mock_pssh):
        mock_pssh.side_effect = lambda *args, **kwargs: MagicMock()
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.head.destroy_clients.side_effect = OSError("already gone")
        orch.close()
        orch.all.destroy_clients.assert_called_once_with()

    @patch("cvs.core.orchestrators.baremetal.is_managed_compute", return_value=True)
    @patch("cvs.core.orchestrators.container.RuntimeFactory")
    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_container_orchestrator_uses_http_with_agent_config(self, mock_pssh, _runtime, _managed):
        from cvs.core.orchestrators.container import ContainerOrchestrator

        cfg = OrchestratorConfig(
            orchestrator="container",
            node_dict={"10.0.0.1": {}, "10.0.0.2": {}},
            username="testuser",
            priv_key_file="/dev/null",
            password=None,
            head_node_dict={"mgmt_ip": "10.0.0.1"},
            container={
                "lifetime": "per_run",
                "image": "rocm/cvs:test",
                "name": "cvs_iter_test",
                "runtime": {"name": "docker", "args": {}},
            },
            agent_token_file=self.token_file,
        )
        cfg.node_dict["10.0.0.1"]["agent_port"] = 9000
        cfg.node_dict["10.0.0.2"]["agent_port"] = 9001
        ContainerOrchestrator(MagicMock(), cfg)
        self.assertEqual(mock_pssh.call_args.kwargs["transport"], "http")
        self.assertEqual(mock_pssh.call_args.kwargs["token_file"], self.token_file)


if __name__ == "__main__":
    unittest.main()
