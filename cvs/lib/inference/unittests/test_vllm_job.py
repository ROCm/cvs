'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/inference/vllm_job.py: server PID file, lost-launch retry,
fail-fast startup checks, and the server and client time budgets.
'''

import os
import shutil
import signal
import subprocess  # nosec B404 - Popen gives the test a real child PID to inspect
import sys
import tempfile
import unittest
from unittest import mock

from cvs.config_validation.schemas.inference.engines import VllmConfig
from cvs.lib.inference.unittests.fake_orch import FakeOrch
from cvs.lib.inference.vllm_job import VllmJob
from cvs.unittests.support import FAKE_CRED, TMP, run_trusted

FAKE_CRED_FILE = "/models/.hf"
HEAD = "10.0.0.1"
WORKER = "10.0.0.2"


def _result(output="", exit_code=0):
    return {"output": output, "exit_code": exit_code}


def _state(log, pid):
    return _result(f"server-state log={log} pid={pid}\n")


LAUNCHED = _result()
LOST = _result("HTTPConnectionError('')\n", -1)
NOTHING_ON_DISK = _state("missing", "none")
ALIVE = _state("present", "alive")
DEAD = _state("present", "dead")


class ServerScript:
    def __init__(self, launch=(LAUNCHED,), state=(ALIVE,), tail="INFO Loading weights\n", ready=False):
        self.launch = list(launch)
        self.state = {h: list(s) for h, s in state.items()} if isinstance(state, dict) else list(state)
        self.tail = tail
        self.ready = ready
        self.events = []
        self.launches = []

    def __call__(self, cmd, hosts=None, detailed=False, **kwargs):
        host = hosts[0] if hosts else HEAD
        if "nohup" in cmd:
            self.launches.append(cmd)
            kind, result = "launch", self._next(self.launch)
        elif "server-state" in cmd:
            kind, result = "probe", self._next(self.state[host] if isinstance(self.state, dict) else self.state)
        elif cmd.startswith("tail "):
            kind, result = "tail", self.tail
        elif cmd.startswith("grep -qiE "):
            kind, result = "ready_grep", _result("", 0 if self.ready else 1)
        else:
            kind, result = "other", _result() if detailed else ""
        self.events.append(kind)
        return {host: result}

    @staticmethod
    def _next(queue):
        return queue.pop(0) if len(queue) > 1 else queue[0]


class FakeClock:
    """Stands in for ``time.sleep`` and records each sleep.

    ``now`` is the wall time: each sleep moves it, and ``advance`` moves it for
    the time a command takes.
    """

    def __init__(self):
        self.now = 0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds):
        self.now += seconds

    def patch(self):
        return mock.patch("cvs.lib.inference.vllm_job.time.sleep", side_effect=self.sleep)


def _variant(log_dir, pp, timeouts=None):
    cell = f"ISL=1024,OSL=1024,TP=8,PP={pp},CONC=16"
    return VllmConfig(
        enforce_thresholds=False,
        threshold_json="threshold.json",
        paths={"shared_fs": log_dir, "models_dir": "/models", "log_dir": log_dir, "hf_token_file": FAKE_CRED_FILE},
        container={"name": "test", "image": "test"},
        server_params={"model": "/models/test-model", "tp": 8, "pp": pp, "port": 8000},
        benchmark_params={"num_prompts": 64},
        sweeps={cell: {}},
        runs=[cell],
        vllm={"distributed_executor_backend": "mp"},
        timeouts=timeouts or {},
    )


def _job(responder=None, hosts=(HEAD,), log_dir="/logs", timeouts=None):
    orch = FakeOrch(hosts=list(hosts))
    orch.exec = responder or ServerScript()
    return VllmJob(
        orch=orch,
        variant=_variant(log_dir, pp=len(hosts), timeouts=timeouts),
        isl="1024",
        osl="1024",
        concurrency="16",
        num_prompts="64",
    )


@mock.patch("cvs.lib.inference.vllm_job.time.sleep")
class TestStartServer(unittest.TestCase):
    def test_each_rank_clears_and_writes_its_own_pid_file(self, sleep):
        setup = []
        _job(lambda cmd, hosts=None, **kwargs: setup.append(cmd), hosts=(HEAD, WORKER)).build_server_cmd()
        script = ServerScript()
        _job(script, hosts=(HEAD, WORKER)).start_server()
        self.assertEqual(len(script.launches), 2)
        for rank, cmd in enumerate(script.launches):
            rank_dir = f"/logs/vllm/out-node{rank}/isl1024_osl1024_conc16"
            self.assertIn(f"mkdir -p {rank_dir} && rm -f {rank_dir}/vllm_serve_server.log {rank_dir}/server.pid", setup)
            self.assertIn(f"echo $! > {rank_dir}/server.pid;", cmd)

    def test_lost_launch_that_left_no_files_is_retried_after_a_settle_wait(self, sleep):
        script = ServerScript(launch=[LOST, LAUNCHED], state=[NOTHING_ON_DISK])
        sleep.side_effect = lambda seconds: script.events.append("sleep")
        job = _job(script)
        job.start_server()
        self.assertEqual(script.events, ["launch", "sleep", "probe", "launch"])
        sleep.assert_called_once_with(job._launch_retry_wait)

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

    def test_log_without_pid_file_raises_not_started(self, sleep):
        with self.assertRaisesRegex(RuntimeError, "vllm server not started .*no PID file"):
            _job(ServerScript(state=[_state("present", "none")], ready=True)).wait_ready()

    def test_dead_worker_is_reported_with_its_host_and_rank(self, sleep):
        script = ServerScript(state={HEAD: [ALIVE], WORKER: [DEAD]})
        with self.assertRaisesRegex(RuntimeError, f"exited during startup on {WORKER} \\(rank 1\\)"):
            _job(script, hosts=(HEAD, WORKER)).wait_ready()


def _timed_server(clock, ready_from=None, state=(ALIVE,), poll_seconds=0):
    """A ServerScript that notes when each probe and readiness grep runs.

    It reports ready from the ``ready_from``-th readiness grep on, and each grep
    takes ``poll_seconds`` of the clock.
    """
    script = ServerScript(state=state)
    script.timeline = []

    def respond(cmd, hosts=None, detailed=False, **kwargs):
        if cmd.startswith("grep -qiE "):
            grep_count = sum(kind == "ready_grep" for kind, _ in script.timeline) + 1
            script.ready = ready_from is not None and grep_count >= ready_from
        result = script(cmd, hosts=hosts, detailed=detailed, **kwargs)
        if script.events[-1] in ("probe", "ready_grep"):
            script.timeline.append((script.events[-1], clock.now))
        if script.events[-1] == "ready_grep":
            clock.advance(poll_seconds)
        return result

    return script, respond


def _polls(script):
    return [now for kind, now in script.timeline if kind == "ready_grep"]


class TestWaitReadyBudget(unittest.TestCase):
    def test_ready_on_an_early_poll_returns_there_after_the_30_s_precheck(self):
        clock = FakeClock()
        script, respond = _timed_server(clock, ready_from=2)
        with clock.patch():
            _job(respond, timeouts={"server_ready_s": 3630}).wait_ready()
        self.assertEqual(clock.sleeps, [30, 10, 10])
        self.assertEqual(script.timeline, [("probe", 30), ("ready_grep", 40), ("probe", 40), ("ready_grep", 50)])

    def test_never_ready_raises_once_the_sleeps_after_the_precheck_add_up_to_the_budget_with_a_last_poll_there(self):
        clock = FakeClock()
        script, respond = _timed_server(clock)
        with clock.patch(), self.assertRaisesRegex(RuntimeError, "^vllm server did not become ready before timeout$"):
            _job(respond, timeouts={"server_ready_s": 25}).wait_ready()
        self.assertEqual((clock.sleeps, _polls(script)), ([30, 10, 10, 5], [40, 50, 55]))
        self.assertEqual(sum(clock.sleeps[1:]), 25)

    def test_the_precheck_sleeps_30_s_even_when_the_budget_is_shorter(self):
        clock = FakeClock()
        script, respond = _timed_server(clock)
        with clock.patch(), self.assertRaises(RuntimeError):
            _job(respond, timeouts={"server_ready_s": 5}).wait_ready()
        self.assertEqual((clock.sleeps, _polls(script)), ([30, 5], [35]))

    def test_time_spent_polling_comes_on_top_of_the_budget(self):
        clock = FakeClock()
        script, respond = _timed_server(clock, poll_seconds=4)
        with clock.patch(), self.assertRaises(RuntimeError):
            _job(respond, timeouts={"server_ready_s": 25}).wait_ready()
        self.assertEqual((clock.sleeps, _polls(script)), ([30, 10, 10, 5], [40, 54, 63]))
        self.assertEqual(clock.now, 30 + 25 + 3 * 4)

    def test_a_server_that_dies_while_polling_raises_on_that_poll(self):
        clock = FakeClock()
        script, respond = _timed_server(clock, state=[ALIVE, ALIVE, DEAD])
        with clock.patch(), self.assertRaisesRegex(RuntimeError, "vllm server exited during startup"):
            _job(respond, timeouts={"server_ready_s": 3630}).wait_ready()
        self.assertEqual(_polls(script), [40, 50])

    def test_without_a_server_ready_s_the_budget_is_the_old_default_total(self):
        clock = FakeClock()
        script, respond = _timed_server(clock)
        with clock.patch(), self.assertRaises(RuntimeError):
            _job(respond, timeouts={"server_ready_s": None}).wait_ready()
        self.assertEqual(sum(clock.sleeps[1:]), 330 + 60 * 60)
        self.assertEqual(len(_polls(script)), (330 + 60 * 60) // 10)
        self.assertEqual(max(clock.sleeps[1:]), 10)


COMPLETE = "Successful requests: 64\nServing Benchmark Result\n"


def _timed_client(clock, logs, tail_seconds=0):
    """exec_on_head giving the next of ``logs`` for each client-log tail, the last one repeating.

    Returns the clock times of the tails, the commands, and the responder;
    each tail takes ``tail_seconds`` of the clock.
    """
    tails, commands = [], []

    def respond(cmd, **kwargs):
        commands.append(cmd)
        if cmd.startswith("tail "):
            tails.append(clock.now)
            clock.advance(tail_seconds)
        return {HEAD: logs[min(len(tails), len(logs)) - 1]}

    return tails, commands, respond


def _client_job(respond, timeouts):
    job = _job(timeouts=timeouts)
    job.orch.exec_on_head = respond
    return job


class TestWaitClientBudget(unittest.TestCase):
    def test_a_client_that_completes_returns_on_that_poll_with_the_first_poll_after_one_interval(self):
        clock = FakeClock()
        tails, commands, respond = _timed_client(clock, ["", "INFO running", COMPLETE])
        with clock.patch():
            _client_job(respond, {"client_s": 5410}).wait_client_complete()
        self.assertEqual((tails, clock.sleeps), ([10, 20, 30], [10, 10, 10]))
        self.assertTrue(commands[-1].startswith("cat "))

    def test_a_client_that_never_completes_raises_once_the_sleeps_add_up_to_the_budget_with_a_last_poll_there(self):
        clock = FakeClock()
        tails, commands, respond = _timed_client(clock, ["INFO running"])
        with clock.patch(), self.assertRaisesRegex(RuntimeError, "^client did not complete before poll cap$"):
            _client_job(respond, {"client_s": 25}).wait_client_complete()
        self.assertEqual((tails, clock.sleeps), ([10, 20, 25], [10, 10, 5]))
        self.assertTrue(commands[-1].startswith("cat "))

    def test_time_spent_polling_comes_on_top_of_the_budget(self):
        clock = FakeClock()
        tails, _, respond = _timed_client(clock, ["INFO running"], tail_seconds=4)
        with clock.patch(), self.assertRaises(RuntimeError):
            _client_job(respond, {"client_s": 25}).wait_client_complete()
        self.assertEqual((tails, clock.sleeps, clock.now), ([10, 24, 33], [10, 10, 5], 25 + 3 * 4))

    def test_a_crash_or_too_many_failed_requests_raises_on_the_poll_that_sees_it(self):
        for log, message in (
            ("Traceback (most recent call last):\n  KeyError: x\n", "client failed: .*Traceback"),
            ("Successful requests: 98\nFailed requests: 2\n", r"client failed: .*Failed requests: 2/100 \(2\.0%\)"),
        ):
            with self.subTest(log=log):
                clock = FakeClock()
                tails, _, respond = _timed_client(clock, ["INFO running", log])
                with clock.patch(), self.assertRaisesRegex(RuntimeError, message):
                    _client_job(respond, {"client_s": 5410}).wait_client_complete()
                self.assertEqual(tails, [10, 20])

    def test_failed_requests_within_one_percent_keep_polling_until_completion(self):
        clock = FakeClock()
        tolerated = "Successful requests: 199\nFailed requests: 1\n"
        tails, _, respond = _timed_client(clock, [tolerated, tolerated + "Serving Benchmark Result\n"])
        with clock.patch():
            _client_job(respond, {"client_s": 5410}).wait_client_complete()
        self.assertEqual(tails, [10, 20])

    def test_without_a_client_s_the_budget_is_the_old_default_total(self):
        clock = FakeClock()
        tails, _, respond = _timed_client(clock, ["INFO running"])
        with clock.patch(), self.assertRaises(RuntimeError):
            _client_job(respond, {"client_s": None}).wait_client_complete()
        self.assertEqual(sum(clock.sleeps), 120 + 20 * 60)
        self.assertEqual((len(tails), max(clock.sleeps)), ((120 + 20 * 60) // 10, 10))


@unittest.skipUnless(sys.platform.startswith("linux"), "the server probe reads /proc")
class TestServerCommandsInBash(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.env = dict(os.environ)
        self.env_script = None
        self.job = _job(responder=self._bash, log_dir=self.tmp)
        os.makedirs(self.job.out_dir)

    def _bash(self, cmd, hosts=None, detailed=False, **kwargs):
        if self.env_script:
            cmd = cmd.replace(f"{TMP}/server_env_script.sh", self.env_script)
        proc = run_trusted(["bash", "-c", cmd], capture_output=True, text=True, check=False, env=self.env)
        output = proc.stdout + proc.stderr
        return {HEAD: {"output": output, "exit_code": proc.returncode} if detailed else output}

    def _write(self, path, text):
        with open(path, "w") as f:
            f.write(text)

    def _state_of_exited(self, reap):
        proc = subprocess.Popen(["true"])  # nosec B603 B607 - fixed argv; the test needs a live, unreaped child
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

    def test_server_env_reads_the_token_file_in_the_shell(self):
        token_file = os.path.join(self.tmp, "hf_token")
        self.job.hf_token_file = token_file
        self.env_script = os.path.join(self.tmp, "server_env_script.sh")
        sent = []
        self.job.orch.exec = lambda cmd, hosts=None, **kwargs: (sent.append(cmd), self._bash(cmd, hosts, **kwargs))[1]
        for content, expected in ((f"{FAKE_CRED}\n", FAKE_CRED), (None, "")):
            with self.subTest(file_present=content is not None):
                if content is None:
                    os.remove(token_file)
                else:
                    self._write(token_file, content)
                self.job.build_server_cmd()
                exported = run_trusted(
                    ["bash", "-c", f'. {self.env_script}; printf %s "$HF_TOKEN"'], capture_output=True, text=True
                )
                self.assertEqual(exported.stdout, expected)
                self.assertFalse([cmd for cmd in sent if FAKE_CRED in cmd])

    def test_nothing_on_disk(self):
        self.assertEqual(self.job._server_state(0, HEAD), ("missing", "none"))

    def test_earlier_runs_files_do_not_stop_a_lost_launch_from_retrying(self):
        # Cell dirs are keyed only by isl/osl/conc, so an earlier run's files can be waiting.
        self._write(self.job.server_log, "INFO Application startup complete.\n")
        self._write(self.job.server_pid_file, "1\n")
        self.env_script = os.path.join(self.tmp, "server_env_script.sh")
        self.job.build_server_cmd()

        def lose_launches(cmd, hosts=None, **kwargs):
            return {HEAD: LOST} if "nohup" in cmd else self._bash(cmd, hosts, **kwargs)

        self.job.orch.exec = lose_launches
        with mock.patch("cvs.lib.inference.vllm_job.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "vllm server not started"):
                self.job.start_server()

    def test_launch_records_a_live_server_pid(self):
        bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(bin_dir)
        self._write(os.path.join(bin_dir, "vllm"), "#!/bin/sh\nexec sleep 300\n")
        os.chmod(os.path.join(bin_dir, "vllm"), 0o700)
        self.env["PATH"] = f"{bin_dir}:{self.env['PATH']}"
        self.env_script = os.path.join(self.tmp, "server_env_script.sh")
        self._write(self.env_script, "")

        self.job.start_server()

        with open(self.job.server_pid_file) as f:
            pid = int(f.read())
        self.addCleanup(os.kill, pid, signal.SIGKILL)
        self.assertEqual(self.job._server_state(0, HEAD)[1], "alive")


if __name__ == "__main__":
    unittest.main()
