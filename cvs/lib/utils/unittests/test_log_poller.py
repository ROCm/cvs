'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/utils/log_poller.py::LogPoller.
'''

import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from cvs.lib.utils.log_poller import LogPoller

_MOD = "cvs.lib.utils.log_poller"


def _orch(hosts):
    o = MagicMock()
    o.hosts = list(hosts)
    o.exec_cmd_list = MagicMock(return_value={})
    return o


class DrainTests(unittest.TestCase):
    def test_drain_maps_hosts_and_advances_cursor(self):
        o = _orch(["h0", "h1"])
        o.exec_cmd_list.return_value = {"h0": "a\nb\n", "h1": "c\n"}
        p = LogPoller(o, ["/l0", "/l1"], complete_pattern="done")
        self.assertEqual(p.drain(), {0: "a\nb\n", 1: "c\n"})
        self.assertEqual(p._cursor, [2, 1])
        # exec_cmd_list is called quietly (print_console=False)
        self.assertFalse(o.exec_cmd_list.call_args.kwargs.get("print_console", True))

    def test_second_drain_tails_from_cursor(self):
        o = _orch(["h0"])
        o.exec_cmd_list.return_value = {"h0": "a\nb\n"}
        p = LogPoller(o, ["/l0"], complete_pattern="done")
        p.drain()
        o.exec_cmd_list.return_value = {"h0": "c\n"}
        p.drain()
        self.assertIn("tail -n +3", o.exec_cmd_list.call_args.args[0][0])  # cursor 2 -> +3

    def test_log_paths_length_must_match_hosts(self):
        with self.assertRaises(ValueError):
            LogPoller(_orch(["h0", "h1"]), ["/only-one"], complete_pattern="done")


class _LocalTailOrch:
    """Run the poller's ``tail -n +K`` commands against a real file with GNU tail."""

    def __init__(self, path):
        self.hosts = ["h0"]
        self.path = path

    def exec_cmd_list(self, cmd_list, print_console=False):
        # Keep \r as \r so this matches GNU tail's on-disk records (text=True would
        # turn \r into \n and reintroduce the cursor skew these tests catch).
        out = subprocess.check_output(["bash", "-c", cmd_list[0]])
        return {"h0": out.decode("utf-8", errors="surrogateescape")}


class DrainCursorVsTailTests(unittest.TestCase):
    """Cursor must match GNU tail -n records, not str.splitlines()."""

    def _poller(self, path):
        return LogPoller(_LocalTailOrch(path), [path], complete_pattern="done")

    def test_incomplete_last_line_is_reread_when_completed(self):
        """File 'hello\\nwor' then append 'ld\\n' must still surface 'world'."""
        fd, path = tempfile.mkstemp(prefix="logpoller-partial-")
        os.close(fd)
        try:
            with open(path, "wb") as fh:
                fh.write(b"hello\nwor")
            p = self._poller(path)
            first = p.drain()
            self.assertEqual(first[0], "hello\n")
            self.assertEqual(p._cursor[0], 1)

            with open(path, "ab") as fh:
                fh.write(b"ld\n")
            second = p.drain()
            self.assertIn("world", second[0])
            self.assertEqual(p._cursor[0], 2)
        finally:
            os.unlink(path)

    def test_carriage_return_does_not_skip_a_later_line(self):
        """GNU tail treats 'a\\rb\\n' as one record; a following line must still drain."""
        fd, path = tempfile.mkstemp(prefix="logpoller-cr-")
        os.close(fd)
        try:
            with open(path, "wb") as fh:
                fh.write(b"step 1\rstep 2\n")
            p = self._poller(path)
            p.drain()
            self.assertEqual(p._cursor[0], 1)

            with open(path, "ab") as fh:
                fh.write(b"line3\n")
            second = p.drain()
            self.assertIn("line3", second[0])
        finally:
            os.unlink(path)


class IsCompleteTests(unittest.TestCase):
    def _poller(self, counts, policy="all"):
        o = _orch(list(counts))
        o.exec_cmd_list.return_value = {h: str(c) for h, c in counts.items()}
        return LogPoller(o, [f"/{h}" for h in counts], complete_pattern="done", complete_policy=policy)

    def test_all_true(self):
        self.assertTrue(self._poller({"h0": 1, "h1": 1}).is_complete())

    def test_all_false_when_one_missing(self):
        self.assertFalse(self._poller({"h0": 1, "h1": 0}).is_complete())

    def test_any_true(self):
        self.assertTrue(self._poller({"h0": 0, "h1": 1}, policy="any").is_complete())

    def test_node0_true(self):
        self.assertTrue(self._poller({"h0": 1, "h1": 0}, policy="node0").is_complete())

    def test_callable_override(self):
        cb = MagicMock(return_value=True)
        p = LogPoller(_orch(["h0"]), ["/l0"], is_complete=cb)
        self.assertTrue(p.is_complete())
        cb.assert_called_once()


class ScanTests(unittest.TestCase):
    def _poller(self, **kw):
        return LogPoller(_orch(["h0"]), ["/l0"], complete_pattern="done", **kw)

    def test_error_pattern_raises(self):
        with self.assertRaises(RuntimeError):
            self._poller(error_patterns={"oom": "RESOURCE_EXHAUSTED"}).scan("h0", 0, "RESOURCE_EXHAUSTED: x")

    def test_case_insensitive_match(self):
        with self.assertRaises(RuntimeError):
            self._poller(error_patterns={"nan": r"nan"}).scan("h0", 0, "loss: NaN")

    def test_first_match_wins_in_insertion_order(self):
        # The dict order controls precedence; the first matching key is reported.
        p = self._poller(error_patterns={"nan": r"nan", "oom": r"RESOURCE_EXHAUSTED"})
        with self.assertRaises(RuntimeError) as ctx:
            p.scan("h0", 0, "loss: NaN and RESOURCE_EXHAUSTED")
        self.assertIn("'nan'", str(ctx.exception))

    def test_benign_text_does_not_raise(self):
        self._poller(error_patterns={"oom": "RESOURCE_EXHAUSTED"}).scan("h0", 0, "step done, ok")

    def test_scan_callable_override(self):
        cb = MagicMock()
        self._poller(scan_chunk=cb).scan("h0", 0, "anything")
        cb.assert_called_once_with("h0", 0, "anything")

    def test_ignore_pattern_suppresses_match(self):
        # The only offending line also matches an ignore pattern -> no raise.
        p = self._poller(
            error_patterns={"nccl": r"NCCL ERROR"},
            ignore_error_patterns={"benign": r"NCCL ERROR: graceful shutdown"},
        )
        p.scan("h0", 0, "NCCL ERROR: graceful shutdown\n")

    def test_ignore_does_not_mask_real_error_on_other_line(self):
        # One ignored line plus a genuine one in the same chunk -> still raises.
        p = self._poller(
            error_patterns={"nccl": r"NCCL ERROR"},
            ignore_error_patterns={"benign": r"graceful shutdown"},
        )
        with self.assertRaises(RuntimeError):
            p.scan("h0", 0, "NCCL ERROR: graceful shutdown\nNCCL ERROR: unhandled cuda\n")

    def test_no_ignore_patterns_matches_current_behavior(self):
        with self.assertRaises(RuntimeError):
            self._poller(error_patterns={"nccl": r"NCCL ERROR"}).scan("h0", 0, "NCCL ERROR: boom")


class SilentPollTests(unittest.TestCase):
    def test_silent_by_default(self):
        o = _orch(["h0"])
        o.exec_cmd_list.return_value = {"h0": "a\n"}
        p = LogPoller(o, ["/l0"], complete_pattern="done")
        p.drain()
        self.assertIs(o.exec_cmd_list.call_args.kwargs.get("print_console"), False)

    def test_verbose_prints_node_commands(self):
        o = _orch(["h0"])
        o.exec_cmd_list.return_value = {"h0": "1"}
        p = LogPoller(o, ["/l0"], complete_pattern="done", silent_poll=False)
        p.drain()
        self.assertIs(o.exec_cmd_list.call_args.kwargs.get("print_console"), True)
        p.is_complete()
        self.assertIs(o.exec_cmd_list.call_args.kwargs.get("print_console"), True)


class PollTests(unittest.TestCase):
    @patch(f"{_MOD}.ConsoleSpinner")
    @patch(f"{_MOD}.time.sleep")
    def test_completion_returns(self, _sleep, _spin):
        p = LogPoller(_orch(["h0"]), ["/l0"], is_complete=MagicMock(return_value=True), timeout_s=100)
        p.drain = MagicMock(side_effect=[{}, {0: "done\n"}])
        p.poll()  # should not raise

    @patch(f"{_MOD}.ConsoleSpinner")
    @patch(f"{_MOD}.time.sleep")
    def test_error_chunk_raises(self, _sleep, _spin):
        p = LogPoller(_orch(["h0"]), ["/l0"], complete_pattern="done", error_patterns={"nan": r"nan"}, timeout_s=100)
        p.is_complete = MagicMock(return_value=False)
        p.drain = MagicMock(return_value={0: "loss NaN"})
        with self.assertRaises(RuntimeError):
            p.poll()

    @patch(f"{_MOD}.ConsoleSpinner")
    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.time.monotonic", side_effect=[0.0, 100.0])
    def test_timeout_raises(self, _mono, _sleep, _spin):
        p = LogPoller(_orch(["h0"]), ["/l0"], complete_pattern="done", timeout_s=10)
        p.is_complete = MagicMock(return_value=False)
        p.drain = MagicMock(return_value={})
        with self.assertRaises(RuntimeError):
            p.poll()

    @patch(f"{_MOD}.ConsoleSpinner")
    @patch(f"{_MOD}.time.sleep")
    def test_spins_between_drains(self, _sleep, spin_cls):
        spinner = spin_cls.return_value
        p = LogPoller(_orch(["h0"]), ["/l0"], is_complete=MagicMock(side_effect=[False, True]), timeout_s=100)
        p.drain = MagicMock(side_effect=[{}, {}, {0: "done\n"}])
        p.poll()
        spinner.spin_until.assert_called_once()
        spinner.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
