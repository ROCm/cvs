'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/inference/vllm_job.py: server PID file, lost-launch retry,
and fail-fast startup checks. Server reuse and the Ray backend are covered in
test_vllm_job_server_reuse.py and test_vllm_job_ray_backend.py.
'''

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from cvs.lib.inference.unittests.fake_orch import FakeOrch
from cvs.lib.inference.utils.vllm_config_loader import VariantConfig
from cvs.lib.inference.vllm_job import VllmJob

HEAD = "10.0.0.1"
WORKER = "10.0.0.2"
TRANSPORT_ERROR = "HTTPConnectionError('')\n"
FATAL_LINES = (
    "pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig",
    "OSError: [Errno 98] Address already in use",
)


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
        self.events.append(kind)
        return {host: result}

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


@mock.patch("cvs.lib.inference.vllm_job.time.sleep")
class TestStartServer(unittest.TestCase):
    def test_each_rank_clears_and_writes_its_own_pid_file(self, sleep):
        job = _job(hosts=(HEAD, WORKER))
        job.start_server()
        launches = [cmd for cmd, _ in job.orch.commands if "nohup" in cmd]
        self.assertEqual(len(launches), 2)
        for rank, cmd in enumerate(launches):
            rank_dir = f"/logs/vllm/out-node{rank}/isl1024_osl1024_conc16"
            self.assertIn(f"rm -f {rank_dir}/vllm_serve_server.log {rank_dir}/server.pid && ", cmd)
            self.assertIn(f"echo $! > {rank_dir}/server.pid;", cmd)

    def test_lost_launch_that_left_no_files_is_retried_after_a_settle_wait(self, sleep):
        script = ServerScript(launch=[LOST, LAUNCHED], state=[NOTHING_ON_DISK])
        sleep.side_effect = lambda seconds: script.events.append("sleep")
        _job(script).start_server()
        self.assertEqual(script.events, ["launch", "sleep", "probe", "launch"])

    def test_three_lost_launches_raise_not_started(self, sleep):
        script = ServerScript(launch=[LOST], state=[NOTHING_ON_DISK])
        with self.assertRaisesRegex(RuntimeError, "vllm server not started"):
            _job(script).start_server()
        self.assertEqual(script.events.count("launch"), 3)

    def test_lost_launch_that_may_have_run_is_not_repeated(self, sleep):
        # A second launch over a server that did start would race it for the port.
        for state in (_state("present", "none"), _state("missing", "alive"), LOST):
            with self.subTest(probe=state["output"].strip()):
                script = ServerScript(launch=[LOST, LAUNCHED], state=[state])
                _job(script).start_server()
                self.assertEqual(script.events, ["launch", "probe"])

    def test_launch_that_ran_and_failed_raises_without_retry(self, sleep):
        script = ServerScript(launch=[_result("Error response from daemon: container test is not running\n", 1)])
        with self.assertRaisesRegex(RuntimeError, "vllm server failed to launch"):
            _job(script).start_server()
        self.assertEqual(script.events, ["launch"])


@mock.patch("cvs.lib.inference.vllm_job.time.sleep")
class TestWaitReadyFailsFast(unittest.TestCase):
    def test_dead_server_raises_with_its_log_tail(self, sleep):
        script = ServerScript(state=[DEAD], tail="INFO Loading weights\nKilled\n", ready=True)
        with self.assertRaisesRegex(RuntimeError, r"(?s)vllm server exited during startup on .*Killed"):
            _job(script).wait_ready()

    def test_missing_log_raises_not_started(self, sleep):
        script = ServerScript(
            state=[NOTHING_ON_DISK], tail="tail: cannot open 'x' for reading: No such file or directory\n"
        )
        with self.assertRaisesRegex(RuntimeError, "vllm server not started"):
            _job(script).wait_ready()

    def test_known_log_pattern_names_the_cause_when_the_pid_is_gone(self, sleep):
        script = ServerScript(state=[DEAD], fatal_line=FATAL_LINES[0])
        with self.assertRaisesRegex(RuntimeError, "vllm server fatal error"):
            _job(script).wait_ready()

    def test_dead_worker_is_reported_with_its_host_and_rank(self, sleep):
        script = ServerScript(state={HEAD: [ALIVE], WORKER: [DEAD]})
        with self.assertRaisesRegex(RuntimeError, f"exited during startup on {WORKER} \\(rank 1\\)"):
            _job(script, hosts=(HEAD, WORKER)).wait_ready()

    def test_missing_pid_file_lets_the_poll_run(self, sleep):
        _job(ServerScript(state=[_state("present", "none")], ready=True)).wait_ready()


@unittest.skipUnless(sys.platform.startswith("linux"), "the server probe reads /proc")
class TestServerCommandsInBash(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.env = dict(os.environ)
        self.env_script = None
        self.job = _job(responder=self._bash, log_dir=self.tmp)
        os.makedirs(self.job.out_dir)

    def _bash(self, cmd, hosts, detailed=False, **kwargs):
        if self.env_script:
            cmd = cmd.replace("/tmp/server_env_script.sh", self.env_script)
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False, env=self.env)
        output = proc.stdout + proc.stderr
        return {hosts[0]: {"output": output, "exit_code": proc.returncode} if detailed else output}

    def _write(self, path, text):
        with open(path, "w") as f:
            f.write(text)

    def _state_of_exited(self, reap):
        proc = subprocess.Popen(["true"])
        self.addCleanup(proc.wait)
        if reap:
            proc.wait()
        else:
            os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
        self._write(self.job.server_log, "")
        self._write(self.job.server_pid_file, f"{proc.pid}\n")
        return self.job._server_state(0, HEAD)

    def test_exited_server_is_dead_whether_reaped_or_a_zombie(self):
        # CVS containers run `sleep infinity` as PID 1, so an exited server stays a
        # zombie there, and kill -0 succeeds on a zombie.
        for reap in (True, False):
            with self.subTest(reap=reap):
                self.assertEqual(self._state_of_exited(reap), ("present", "dead"))

    def test_nothing_on_disk(self):
        self.assertEqual(self.job._server_state(0, HEAD), ("missing", "none"))

    def test_fatal_lines_are_found_by_the_jobs_own_grep(self):
        # The job hands FATAL_LOG_RE to grep -E, which lacks Python-only syntax such as \d.
        for line in FATAL_LINES:
            with self.subTest(line=line):
                self._write(self.job.server_log, f"INFO Loading weights\n{line}\n")
                with self.assertRaisesRegex(RuntimeError, "vllm server fatal error"):
                    self.job._check_early_failure()

    def test_launch_records_a_live_server_pid(self):
        bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(bin_dir)
        self._write(os.path.join(bin_dir, "vllm"), "#!/bin/sh\nexec sleep 300\n")
        os.chmod(os.path.join(bin_dir, "vllm"), 0o700)
        self.env["PATH"] = f"{bin_dir}:{self.env['PATH']}"
        self.env_script = os.path.join(self.tmp, "server_env_script.sh")
        self._write(self.env_script, "")
        self._write(self.job.server_pid_file, "1\n")

        self.job.start_server()

        with open(self.job.server_pid_file) as f:
            pid = int(f.read())
        self.addCleanup(self._kill, pid)
        self.assertNotEqual(pid, 1)
        self.assertEqual(self.job._server_state(0, HEAD)[1], "alive")

    @staticmethod
    def _kill(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    unittest.main()
