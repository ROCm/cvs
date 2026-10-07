'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/inference/vllm_job.py:
  - start_server clears each rank's stale log and PID file, then records the
    server's PID in server.pid beside the rank's log
  - a launch exec that returns no exit status is retried, at most twice, and
    only after a probe shows it left no log or PID file; other failures raise
  - wait_ready fails before the readiness poll when a rank's log is missing or
    its server PID is gone, and a known log pattern still names the cause
  - FATAL_LOG_RE names pydantic ValidationErrors and ports already in use
  - the probe, launch and fatal-pattern grep, run in a local bash against real
    processes, behave as they would in the container; a zombie counts as dead

Server reuse and the Ray backend are covered in test_vllm_job_server_reuse.py
and test_vllm_job_ray_backend.py.
'''

import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
import unittest.mock as mock

from cvs.lib.inference.unittests.fake_orch import FakeOrch
from cvs.lib.inference.utils.vllm_config_loader import VariantConfig
from cvs.lib.inference.vllm_job import VllmJob

HEAD = "10.0.0.1"
WORKER = "10.0.0.2"
TRANSPORT_ERROR = "HTTPConnectionError('')\n"
VALIDATION_ERROR = "pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig"


def _result(output="", exit_code=0):
    return {"output": output, "exit_code": exit_code}


def _state(log, pid):
    return _result(f"server-state log={log} pid={pid}\n")


LAUNCHED = _result()
LOST = _result(TRANSPORT_ERROR, -1)
NOTHING_ON_DISK = _state("missing", "none")
ALIVE = _state("present", "alive")
DEAD = _state("present", "dead")


class ServerScript:
    """Answers the commands VllmJob issues for one server, by kind of command.

    ``launch`` and ``state`` are lists of results used in order, the last one
    repeating; ``state`` may instead map each host to its own list. ``events``
    records the kind and host of every command, so tests can check the order.
    """

    def __init__(self, launch=(LAUNCHED,), state=(ALIVE,), tail="INFO Loading weights\n", fatal_line="", ready=False):
        self.launch = list(launch)
        self.state = {h: list(s) for h, s in state.items()} if isinstance(state, dict) else list(state)
        self.tail = tail
        self.fatal_line = fatal_line
        self.ready = ready
        self.events = []

    def __call__(self, cmd, hosts, detailed=False, **kwargs):
        host = hosts[0] if hosts else HEAD
        if "nohup" in cmd:
            kind, result = "launch", self._next(self.launch)
        elif "server-state" in cmd:
            kind, result = "probe", self._next(self.state[host] if isinstance(self.state, dict) else self.state)
        elif cmd.startswith("tail "):
            kind, result = "tail", self.tail
        elif cmd.startswith("grep -m1 "):
            kind, result = "fatal_grep", _result(self.fatal_line, 0 if self.fatal_line else 1)
        elif cmd.startswith("grep -qiE "):
            kind, result = "ready_grep", _result("", 0 if self.ready else 1)
        else:
            kind, result = "other", _result() if detailed else ""
        self.events.append((kind, host))
        return {host: result}

    def kinds(self):
        return [kind for kind, _ in self.events]

    @staticmethod
    def _next(queue):
        return queue.pop(0) if len(queue) > 1 else queue[0]


def _variant(log_dir, pp):
    cell = f"ISL=1024,OSL=1024,TP=8,PP={pp},CONC=16"
    return VariantConfig(
        enforce_thresholds=False,
        threshold_json="threshold.json",
        paths={"shared_fs": log_dir, "models_dir": "/models", "log_dir": log_dir, "hf_token_file": "/models/.hf"},
        container={"name": "test", "image": "test"},
        server_params={
            "model": "/models/test-model",
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": pp,
            "port": 8000,
        },
        sweeps={cell: {}},
        runs=[cell],
    )


def _job(responder=None, hosts=(HEAD,), log_dir="/logs"):
    return VllmJob(
        orch=FakeOrch(hosts=list(hosts), responder=responder or ServerScript()),
        variant=_variant(log_dir, pp=len(hosts)),
        hf_token="tok",
        isl="1024",
        osl="1024",
        concurrency="16",
        num_prompts="64",
    )


def _calls(job, marker):
    """(hosts, kwargs, bash script) for each exec whose command contains marker."""
    calls = []
    for (cmd, hosts), kwargs in zip(job.orch.commands, job.orch.exec_kwargs):
        if marker in cmd:
            argv = shlex.split(cmd)
            calls.append((hosts, kwargs, argv[2] if argv[:2] == ["bash", "-c"] else cmd))
    return calls


@mock.patch("cvs.lib.inference.vllm_job.time.sleep")
class TestStartServer(unittest.TestCase):
    def test_each_rank_launches_on_its_host_and_writes_its_own_pid_file(self, sleep):
        job = _job(hosts=(HEAD, WORKER))
        job.start_server()
        launches = _calls(job, "nohup")
        self.assertEqual([hosts for hosts, _, _ in launches], [[HEAD], [WORKER]])
        for rank, (_, kwargs, script) in enumerate(launches):
            with self.subTest(rank=rank):
                rank_dir = f"/logs/vllm/out-node{rank}/isl1024_osl1024_conc16"
                self.assertTrue(script.startswith(f"rm -f {rank_dir}/vllm_serve_server.log {rank_dir}/server.pid && "))
                self.assertIn(f"> {rank_dir}/vllm_serve_server.log 2>&1 & echo $! > {rank_dir}/server.pid;", script)
                # Without exit codes a lost launch looks like a successful one.
                self.assertTrue(kwargs.get("detailed"))

    def test_lost_launch_that_left_no_files_is_retried_once(self, sleep):
        script = ServerScript(launch=[LOST, LAUNCHED], state=[NOTHING_ON_DISK])
        sleep.side_effect = lambda seconds: script.events.append(("sleep", seconds))
        job = _job(script)
        job.start_server()
        # The probe waits first, so a launch that did get through has time to write its files.
        self.assertEqual(script.kinds(), ["launch", "sleep", "probe", "launch"])
        self.assertIn(("sleep", job._launch_retry_wait), script.events)

    def test_three_lost_launches_raise_not_started(self, sleep):
        script = ServerScript(launch=[LOST], state=[NOTHING_ON_DISK])
        with self.assertRaises(RuntimeError) as ctx:
            _job(script).start_server()
        self.assertIn(f"vllm server not started on {HEAD} (rank 0)", str(ctx.exception))
        self.assertIn("HTTPConnectionError", str(ctx.exception))
        self.assertEqual(script.kinds().count("launch"), 3)

    def test_lost_launch_that_may_have_run_is_not_repeated(self, sleep):
        # A second launch over a server that did start would race it for the port.
        for state in (
            ALIVE,
            DEAD,
            _state("present", "none"),
            _state("missing", "alive"),
            _state("missing", "dead"),
            LOST,
        ):
            with self.subTest(probe=state["output"].strip()):
                script = ServerScript(launch=[LOST, LAUNCHED], state=[state])
                _job(script).start_server()
                self.assertEqual(script.kinds(), ["launch", "probe"])

    def test_launch_that_ran_and_failed_raises_without_retry(self, sleep):
        # Only a missing exit status is ambiguous; a real non-zero exit is a failed launch.
        script = ServerScript(launch=[_result("Error response from daemon: container test is not running\n", 1)])
        with self.assertRaises(RuntimeError) as ctx:
            _job(script).start_server()
        self.assertIn(f"vllm server failed to launch on {HEAD} (rank 0), exit code 1", str(ctx.exception))
        self.assertIn("is not running", str(ctx.exception))
        self.assertEqual(script.kinds(), ["launch"])
        sleep.assert_not_called()


@mock.patch("cvs.lib.inference.vllm_job.time.sleep")
class TestWaitReadyFailsFast(unittest.TestCase):
    def test_dead_server_raises_with_its_log_tail_before_the_readiness_poll(self, sleep):
        # A server killed without a known log line used to be polled for the whole
        # readiness budget, about 66 minutes with the defaults.
        script = ServerScript(state=[DEAD], tail="INFO Loading weights\nKilled\n", ready=True)
        job = _job(script)
        with self.assertRaises(RuntimeError) as ctx:
            job.wait_ready()
        self.assertIn(f"vllm server exited during startup on {HEAD} (rank 0)", str(ctx.exception))
        self.assertIn("Killed", str(ctx.exception))
        self.assertNotIn("ready_grep", script.kinds())
        sleep.assert_called_once_with(job._precheck_wait)

    def test_missing_log_raises_not_started_instead_of_a_generic_early_failure(self, sleep):
        # tail's own "No such file or directory" used to be reported as an early failure.
        script = ServerScript(
            state=[NOTHING_ON_DISK], tail="tail: cannot open 'x' for reading: No such file or directory\n"
        )
        job = _job(script)
        with self.assertRaises(RuntimeError) as ctx:
            job.wait_ready()
        self.assertEqual(str(ctx.exception), f"vllm server not started on {HEAD} (rank 0): no log at {job.server_log}")
        self.assertNotIn("tail", script.kinds())

    def test_known_log_pattern_names_the_cause_when_the_pid_is_gone(self, sleep):
        script = ServerScript(state=[DEAD], fatal_line=VALIDATION_ERROR)
        with self.assertRaises(RuntimeError) as ctx:
            _job(script).wait_ready()
        self.assertIn(f"vllm server fatal error on {HEAD} (rank 0)", str(ctx.exception))
        self.assertIn(VALIDATION_ERROR, str(ctx.exception))

    def test_dead_worker_is_reported_with_its_host_and_rank(self, sleep):
        script = ServerScript(state={HEAD: [ALIVE], WORKER: [DEAD]})
        with self.assertRaises(RuntimeError) as ctx:
            _job(script, hosts=(HEAD, WORKER)).wait_ready()
        self.assertIn(f"vllm server exited during startup on {WORKER} (rank 1)", str(ctx.exception))

    def test_states_that_prove_no_failure_let_the_poll_run(self, sleep):
        # Only a missing log or a dead PID is proof; a missing PID file or a probe
        # that got no answer must not fail a server that is starting.
        for state in (ALIVE, _state("present", "none"), LOST):
            with self.subTest(probe=state["output"].strip()):
                script = ServerScript(state=[state], ready=True)
                _job(script).wait_ready()
                self.assertEqual(script.kinds().count("ready_grep"), 1)

    def test_probe_runs_quietly_on_each_ranks_own_host(self, sleep):
        # The probe runs on every readiness poll; logging it would repeat it each time.
        job = _job(hosts=(HEAD, WORKER))
        job._check_early_failure()
        probes = _calls(job, "server-state")
        self.assertEqual([hosts for hosts, _, _ in probes], [[HEAD], [WORKER]])
        for _, kwargs, _ in probes:
            self.assertTrue(kwargs.get("detailed"))
            self.assertFalse(kwargs.get("print_console"))


class TestFatalLogPatterns(unittest.TestCase):
    FATAL_LINES = (
        VALIDATION_ERROR,
        "pydantic_core._pydantic_core.ValidationError: 2 validation errors for VllmConfig",
        "OSError: [Errno 98] Address already in use",
        "ERROR:    [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): address already in use",
    )

    def test_startup_failures_match(self):
        for line in self.FATAL_LINES:
            with self.subTest(line=line):
                self.assertRegex(line, VllmJob.FATAL_LOG_RE)

    def test_lines_that_only_mention_the_words_do_not_match(self):
        # The whole log is grepped, so a bare mention must not fail a healthy start.
        for line in (
            "WARNING [config.py:120] Ignoring ValidationError from an optional plugin",
            "INFO [api_server.py:1] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)",
        ):
            with self.subTest(line=line):
                self.assertIsNone(VllmJob.FATAL_LOG_RE.search(line))


@unittest.skipUnless(sys.platform.startswith("linux"), "the server probe reads /proc")
class TestServerCommandsInBash(unittest.TestCase):
    """Runs VllmJob's real commands in a local bash, as docker exec runs them in the container."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.env = dict(os.environ)
        self.env_script = None
        self.job = _job(responder=self._bash, log_dir=self.tmp)
        os.makedirs(self.job.out_dir)

    def _bash(self, cmd, hosts, detailed=False, **kwargs):
        if self.env_script:
            # The launch sources a fixed path; use a temp copy so the test never touches the real one.
            cmd = cmd.replace("/tmp/server_env_script.sh", self.env_script)
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False, env=self.env)
        output = proc.stdout + proc.stderr
        return {hosts[0]: {"output": output, "exit_code": proc.returncode} if detailed else output}

    def _write(self, path, text):
        with open(path, "w") as f:
            f.write(text)

    def _server(self, *argv):
        proc = subprocess.Popen(list(argv))
        self.addCleanup(proc.wait)
        return proc

    def _state_with_pid(self, pid):
        self._write(self.job.server_log, "INFO Loading weights\n")
        self._write(self.job.server_pid_file, f"{pid}\n")
        return self.job._server_state(0, HEAD)

    def test_running_server_is_alive(self):
        proc = self._server("sleep", "300")
        self.addCleanup(proc.kill)
        self.assertEqual(self._state_with_pid(proc.pid), ("present", "alive"))

    def test_exited_and_reaped_server_is_dead(self):
        proc = self._server("true")
        proc.wait()
        self.assertEqual(self._state_with_pid(proc.pid), ("present", "dead"))

    def test_exited_but_unreaped_server_is_dead(self):
        # CVS containers run `sleep infinity` as PID 1, so an exited server stays a
        # zombie there, and kill -0 succeeds on a zombie.
        proc = self._server("true")
        os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
        self.assertEqual(self._state_with_pid(proc.pid), ("present", "dead"))

    def test_log_without_pid_file(self):
        self._write(self.job.server_log, "INFO Loading weights\n")
        self.assertEqual(self.job._server_state(0, HEAD), ("present", "none"))

    def test_nothing_on_disk(self):
        self.assertEqual(self.job._server_state(0, HEAD), ("missing", "none"))

    def test_fatal_lines_are_found_by_the_jobs_own_grep(self):
        # The job hands FATAL_LOG_RE to grep -E, which lacks Python-only syntax such as \d.
        proc = self._server("sleep", "300")
        self.addCleanup(proc.kill)
        for line in TestFatalLogPatterns.FATAL_LINES:
            with self.subTest(line=line):
                self._write(self.job.server_log, f"INFO Loading weights\n{line}\nINFO Shutting down\n")
                self._write(self.job.server_pid_file, f"{proc.pid}\n")
                with self.assertRaises(RuntimeError) as ctx:
                    self.job._check_early_failure()
                self.assertEqual(str(ctx.exception), f"vllm server fatal error on {HEAD} (rank 0): {line}")

    def test_launch_replaces_stale_files_and_records_a_live_server(self):
        bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(bin_dir)
        self._write(os.path.join(bin_dir, "vllm"), "#!/bin/sh\nexec sleep 300\n")
        os.chmod(os.path.join(bin_dir, "vllm"), 0o700)
        self.env["PATH"] = f"{bin_dir}:{self.env['PATH']}"
        self.env_script = os.path.join(self.tmp, "server_env_script.sh")
        self._write(self.env_script, "export CVS_TEST_ENV=1\n")
        # A stale log from an earlier run would satisfy is_ready without a server.
        self._write(self.job.server_log, "INFO Application startup complete.\n")
        self._write(self.job.server_pid_file, "1\n")

        self.job.start_server()

        with open(self.job.server_pid_file) as f:
            pid = int(f.read())
        self.addCleanup(self._kill, pid)
        self.assertNotEqual(pid, 1)
        self.assertEqual(self.job._server_state(0, HEAD)[1], "alive")
        if os.path.exists(self.job.server_log):
            with open(self.job.server_log) as f:
                self.assertNotIn("Application startup complete", f.read())

    @staticmethod
    def _kill(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    unittest.main()
