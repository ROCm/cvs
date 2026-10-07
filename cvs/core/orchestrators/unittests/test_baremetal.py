'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# Unit tests for cvs/core/orchestrators/baremetal.py: BaremetalOrchestrator construction
# and command-dispatch surface (exec, exec_on_head, cleanup, sudo_prefix, build_mpi_cmd)
# used by the migrated rvs_cvs.py orch fixture. Mocks Pssh so tests run with no SSH; the
# build_mpi_cmd tests run its head-node commands in a local bash instead.

import os
import shlex
import stat
import subprocess
import unittest
import tempfile
from unittest.mock import MagicMock, patch

from cvs.core.orchestrators.factory import OrchestratorConfig
from cvs.core.orchestrators.baremetal import BaremetalOrchestrator
from cvs.lib.parallel.config import ParallelConfig


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
        orch.all.exec.assert_called_once_with("ls", timeout=5, detailed=False, print_console=True)
        self.assertEqual(result, {"10.0.0.1": "ok", "10.0.0.2": "ok"})

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
        orch.all.download_file.return_value = {"10.0.0.2": "/tmp/result.log"}
        result = orch.download_file("/remote/result.log", "/tmp/result.log", hosts=["10.0.0.2"])
        orch.all.download_file.assert_called_once_with("/remote/result.log", "/tmp/result.log", hosts=["10.0.0.2"])
        self.assertEqual(result, {"10.0.0.2": "/tmp/result.log"})

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
    def test_exec_on_head_forwards_print_console_false(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        orch.head = MagicMock()
        orch.exec_on_head("cat /tmp/huge", print_console=False)
        self.assertIs(orch.head.exec.call_args.kwargs["print_console"], False)

    @patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
    def test_cleanup_returns_true(self, _mock_pssh):
        orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        self.assertTrue(orch.cleanup(orch.hosts))


# Stands in for mpirun: prints the --hostfile it was given, then exits with
# FAKE_MPIRUN_STATUS.
_FAKE_MPIRUN = """#!/bin/sh
while [ "$#" -gt 0 ]; do
    if [ "$1" = --hostfile ]; then cat "$2"; fi
    shift
done
exit "${FAKE_MPIRUN_STATUS:-0}"
"""


def _write_executable(path, body):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(body)
    os.chmod(path, 0o700)


class _LocalHead:
    """Head-node handle stand-in that runs each command in a local bash.

    TMPDIR points at the test's scratch directory, so mktemp never writes
    outside it. banner and warning are wrapped around the real output the way
    a login banner (stdout) and a sudo or docker warning (stderr) reach the
    output of a real handle. exit_code, when set, replaces the real status, as
    a handle does when it gives up on a command (e.g. on a read timeout).
    """

    def __init__(self, test, host, tmpdir, path_dir=None, banner="", warning="", exit_code=None):
        self.test = test
        self.host = host
        self.env = dict(os.environ, TMPDIR=tmpdir)
        if path_dir:
            self.env["PATH"] = path_dir + os.pathsep + self.env["PATH"]
        self.banner = banner
        self.warning = warning
        self.exit_code = exit_code
        self.calls = []

    def exec(self, cmd, timeout=None, detailed=False, print_console=True):
        self.calls.append((cmd, detailed))
        # Checked before running, so neither can execute on the machine running the tests.
        self.test.assertNotIn("sudo", cmd)
        self.test.assertNotIn("mpi_hosts.txt", cmd)
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=self.env, check=False)
        output = self.banner + proc.stdout + proc.stderr + self.warning
        exit_code = proc.returncode if self.exit_code is None else self.exit_code
        return {self.host: {"output": output, "exit_code": exit_code}}


class TestBaremetalOrchestratorMpiHostfile(unittest.TestCase):
    """build_mpi_cmd's hostfile: a private mktemp file written by one head-node
    command as the SSH user, quoted wherever it is used, and removed by the
    returned command once mpirun exits."""

    def setUp(self):
        pssh_patcher = patch("cvs.core.orchestrators.baremetal.MultiProcessParallelHandle")
        pssh_patcher.start()
        self.addCleanup(pssh_patcher.stop)
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.scratch = scratch.name
        # The space and the quote break any unquoted use of the hostfile path.
        self.tmpdir = os.path.join(self.scratch, "run's tmp")
        os.mkdir(self.tmpdir)
        self.orch = BaremetalOrchestrator(MagicMock(), _make_orch_config())
        self.head = self._use_local_head()

    def _use_local_head(self, **kwargs):
        kwargs.setdefault("tmpdir", self.tmpdir)
        self.orch.head = _LocalHead(self, self.orch.head_node, **kwargs)
        return self.orch.head

    def _build(self, **overrides):
        kwargs = {
            "rank_cmd": "echo hi",
            "mpi_hosts": ["10.0.0.1", "10.0.0.2"],
            "ranks_per_host": 1,
            "env_vars": {},
            "mpi_install_dir": "/opt/mpi",
        }
        kwargs.update(overrides)
        return self.orch.build_mpi_cmd(**kwargs)

    def _created_files(self):
        return [os.path.join(self.tmpdir, name) for name in os.listdir(self.tmpdir)]

    @staticmethod
    def _hostfile_arg(cmd):
        tokens = shlex.split(cmd)
        return tokens[tokens.index("--hostfile") + 1]

    def test_build_mpi_cmd_writes_private_hostfile_named_by_hostfile_arg(self):
        # Quotes in a host name must land in the file verbatim instead of
        # ending the quoting of the command that writes it.
        cmd = self._build(mpi_hosts=["10.0.0.1", "node'2", 'node"3'], ranks_per_host=4)

        created = self._created_files()
        self.assertEqual(len(created), 1, created)
        with open(created[0], encoding="utf-8") as stream:
            self.assertEqual(stream.read(), "10.0.0.1 slots=4\nnode'2 slots=4\nnode\"3 slots=4\n")
        self.assertEqual(stat.S_IMODE(os.stat(created[0]).st_mode), 0o600)
        self.assertEqual(self._hostfile_arg(cmd), created[0])
        # A single exec_on_head(detailed=True) call creates and fills the file.
        self.assertEqual([detailed for _, detailed in self.head.calls], [True])

    def test_build_mpi_cmd_result_removes_hostfile_and_keeps_mpirun_status(self):
        mpi_dir = os.path.join(self.scratch, "mpi")
        os.mkdir(mpi_dir)
        _write_executable(os.path.join(mpi_dir, "mpirun"), _FAKE_MPIRUN)

        for status in (0, 7):
            with self.subTest(mpirun_status=status):
                cmd = self._build(mpi_install_dir=mpi_dir)
                # Composed the way a caller might: `&&` must see mpirun's status,
                # not that of the cleanup that runs after it.
                proc = subprocess.run(
                    ["bash", "-c", f"{cmd} && echo after-mpirun"],
                    capture_output=True,
                    text=True,
                    env=dict(os.environ, FAKE_MPIRUN_STATUS=str(status)),
                    check=False,
                )
                self.assertEqual(proc.returncode, status, proc.stderr)
                self.assertEqual("after-mpirun" in proc.stdout, status == 0)
                self.assertIn("10.0.0.2 slots=1\n", proc.stdout)
                self.assertEqual(self._created_files(), [])

    def test_build_mpi_cmd_hostfile_commands_do_not_use_sudo(self):
        # mpirun runs as the SSH user, so a sudo-created 0600 file would be
        # unreadable to it, and container images often have no sudo at all.
        with patch.object(self.orch, "sudo_prefix", return_value="sudo -n "):
            cmd = self._build()

        self.assertNotIn("sudo", cmd)
        for sent, _ in self.head.calls:
            self.assertNotIn("sudo", sent)

    def test_build_mpi_cmd_raises_when_hostfile_cannot_be_created(self):
        fake_bin = os.path.join(self.scratch, "bin")
        os.mkdir(fake_bin)
        # Prints the file it was asked for (its last argument), but under a
        # directory that does not exist: mktemp "succeeds", the write fails.
        _write_executable(
            os.path.join(fake_bin, "mktemp"),
            '#!/bin/sh\nfor template; do :; done\necho "$TMPDIR/missing/${template##*/}"\n',
        )
        cases = {
            "mktemp_fails": {"tmpdir": os.path.join(self.scratch, "missing")},
            "write_fails": {"path_dir": fake_bin},
            # The path is printed, but the handle reports the command as failed.
            "head_reports_failure": {"exit_code": -1},
        }
        for label, head_kwargs in cases.items():
            with self.subTest(label):
                self._use_local_head(**head_kwargs)
                with self.assertRaises(RuntimeError):
                    self._build()

    def test_build_mpi_cmd_finds_hostfile_path_among_other_head_output(self):
        self._use_local_head(
            banner="Welcome to node0\n",
            warning="sudo: unable to resolve host node0: Name or service not known\n",
        )

        cmd = self._build()

        self.assertEqual([self._hostfile_arg(cmd)], self._created_files())

    def test_build_mpi_cmd_raises_unless_head_reports_exactly_one_hostfile_path(self):
        # Without exactly one path in the output, the hostfile is unknown; mpirun
        # must not be pointed at a guess, such as a file another run created.
        with self.subTest("no_path"):
            self.orch.head = MagicMock()
            self.orch.head.exec.return_value = {"10.0.0.1": {"output": "Welcome to node0\n", "exit_code": 0}}
            with self.assertRaises(RuntimeError):
                self._build()
        with self.subTest("two_paths"):
            self._use_local_head(banner=os.path.join(self.tmpdir, "cvs_mpi_hosts.other") + "\n")
            with self.assertRaises(RuntimeError):
                self._build()


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
        # flip to 'sudo -n ' and privileged commands sent to the head node
        # through exec_on_head would run under an unnecessary and potentially
        # unavailable sudo.
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
                    orch.setup_env(["10.0.0.2"], env_script="/tmp/env.sh")
                else:
                    mock_pssh.return_value.exec.side_effect = side_effect
                    with self.assertRaises(RuntimeError):
                        orch.setup_env(["10.0.0.2"], env_script="/tmp/env.sh")

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
