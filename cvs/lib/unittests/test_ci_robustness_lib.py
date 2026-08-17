# cvs/lib/unittests/test_ci_robustness_lib.py
"""Unit tests for retry + GPU-cleanup helpers (cvs.lib.ci_robustness_lib)."""

import unittest

import cvs.lib.ci_robustness_lib as rb


class TestClassifyFailure(unittest.TestCase):
    def test_retriable_patterns(self):
        self.assertTrue(rb.classify_failure("NCCL ERROR: remote process exited"))
        self.assertTrue(rb.classify_failure("MPI_Init: PML add procs failed"))
        self.assertTrue(rb.classify_failure("Connection timed out"))
        self.assertTrue(rb.classify_failure("no result rows returned"))

    def test_non_retriable_wins(self):
        # Even though it also contains a retriable-ish word, corruption must not retry.
        self.assertFalse(rb.classify_failure("SEVERE DATA CORRUPTION: NCCL ERROR"))
        self.assertFalse(rb.classify_failure("RCCL Test float schema validation failed: ..."))
        self.assertFalse(rb.classify_failure("rccl-tests reported '#wrong' = 5"))

    def test_unknown_defaults_retriable(self):
        self.assertTrue(rb.classify_failure("some unfamiliar transient blip"))

    def test_accepts_exception_objects(self):
        self.assertFalse(rb.classify_failure(RuntimeError("SEVERE DATA CORRUPTION")))
        self.assertTrue(rb.classify_failure(RuntimeError("socket closed")))


class TestRunWithRetries(unittest.TestCase):
    def test_success_first_attempt(self):
        calls = []
        out = rb.run_with_retries(lambda: (calls.append(1), "ok")[1], max_retries=3)
        self.assertEqual(out, "ok")
        self.assertEqual(len(calls), 1)

    def test_success_after_retries(self):
        state = {"n": 0}

        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise RuntimeError("NCCL ERROR transient")
            return "recovered"

        sleeps = []
        retries = []
        out = rb.run_with_retries(
            flaky, max_retries=3, backoff_sec=5,
            sleep_fn=lambda s: sleeps.append(s),
            on_before_retry=lambda n: retries.append(n),
        )
        self.assertEqual(out, "recovered")
        self.assertEqual(state["n"], 3)               # failed twice, succeeded on 3rd
        self.assertEqual(retries, [2, 3])             # hook ran before attempts 2 and 3
        self.assertEqual(sleeps, [5, 10])             # linear backoff 5*1, 5*2

    def test_exhausts_and_raises(self):
        state = {"n": 0}

        def always_fail():
            state["n"] += 1
            raise RuntimeError("timeout")

        with self.assertRaises(RuntimeError):
            rb.run_with_retries(always_fail, max_retries=2, sleep_fn=lambda s: None)
        self.assertEqual(state["n"], 3)               # 1 + 2 retries

    def test_non_retriable_raises_immediately(self):
        state = {"n": 0}

        def corrupt():
            state["n"] += 1
            raise RuntimeError("SEVERE DATA CORRUPTION")

        with self.assertRaises(RuntimeError):
            rb.run_with_retries(corrupt, max_retries=5, sleep_fn=lambda s: None)
        self.assertEqual(state["n"], 1)               # never retried


class TestParseGpuPids(unittest.TestCase):
    def test_parse_typical_output(self):
        out = """
========================= ROCm System Management Interface =========================
================================== KFD Processes ===================================
PID      PROCESS NAME        GPU(s)   VRAM USED
12345    all_reduce_perf     8        1234567
12346    all_reduce_perf     8        1234567
0        systemd             0        0
====================================================================================
"""
        pids = rb.parse_gpu_pids(out)
        self.assertEqual(pids, [12345, 12346])        # header text + pid 0 excluded

    def test_empty(self):
        self.assertEqual(rb.parse_gpu_pids(""), [])
        self.assertEqual(rb.parse_gpu_pids(None), [])


class TestBuildCleanupScript(unittest.TestCase):
    def test_self_safe_patterns_and_pkill(self):
        script = rb.build_gpu_cleanup_script(process_patterns=["all_reduce_perf", "orted"])
        # Bracketized (self-match-safe) forms present...
        self.assertIn("[a]ll_reduce_perf", script)
        self.assertIn("[o]rted", script)
        # ...and the bare forms are NOT used as a standalone pkill target.
        self.assertNotIn("pkill -9 -f -- all_reduce_perf", script)
        self.assertIn("pkill -9 -f", script)

    def test_gpu_pids_block_toggle(self):
        with_pids = rb.build_gpu_cleanup_script(kill_gpu_pids=True)
        without = rb.build_gpu_cleanup_script(kill_gpu_pids=False)
        self.assertIn("rocm-smi --showpids", with_pids)
        self.assertNotIn("rocm-smi --showpids", without)

    def test_container_block_toggle(self):
        with_c = rb.build_gpu_cleanup_script(kill_containers=True)
        without = rb.build_gpu_cleanup_script(kill_containers=False)
        self.assertIn("docker kill", with_c)
        self.assertNotIn("docker kill", without)

    def test_sudo_prefix(self):
        s = rb.build_gpu_cleanup_script(process_patterns=["orted"], use_sudo=True)
        self.assertIn("sudo pkill", s)


if __name__ == "__main__":
    unittest.main()
