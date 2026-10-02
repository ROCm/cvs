'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unit tests for cvs/lib/utils/console_spinner.py::ConsoleSpinner.
'''

import unittest
from unittest.mock import MagicMock, patch

from cvs.lib.utils.console_spinner import ConsoleSpinner

_MOD = "cvs.lib.utils.console_spinner"


class _FakeStream:
    def __init__(self, tty):
        self._tty = tty
        self.writes = []
        self.closed = False

    def write(self, text):
        self.writes.append(text)

    def flush(self):
        pass

    def close(self):
        self.closed = True

    def isatty(self):
        return self._tty


def _spinner(tty, owns=False, log=None):
    stream = _FakeStream(tty) if tty else None
    with patch.object(ConsoleSpinner, "_open_console", MagicMock(return_value=(stream, owns))):
        sp = ConsoleSpinner("Working", log=log or MagicMock())
    return sp, stream


class ConsoleSpinnerTests(unittest.TestCase):
    def test_draw_animates_and_advances_on_tty(self):
        sp, stream = _spinner(tty=True)
        sp._draw()
        sp._draw()
        joined = "".join(stream.writes)
        self.assertIn("\rWorking [ | ]", joined)
        self.assertIn("\rWorking [ / ]", joined)  # frame advanced

    def test_clear_and_stop_on_tty(self):
        sp, stream = _spinner(tty=True)
        sp._draw()
        sp.clear()
        sp.stop()
        joined = "".join(stream.writes)
        self.assertIn("\r\033[K", joined)  # clear sequence
        self.assertTrue(joined.endswith("\n"))  # stop drops to a fresh line

    def test_stop_closes_owned_stream(self):
        sp, stream = _spinner(tty=True, owns=True)
        sp.stop()
        self.assertTrue(stream.closed)

    def test_stop_is_idempotent(self):
        # A repeat stop() (e.g. after LogPoller.poll()'s finally already stopped)
        # must be a genuine no-op: the closed stream is not touched again.
        sp, stream = _spinner(tty=True, owns=True)
        sp.stop()
        writes_after_first = len(stream.writes)
        sp.stop()  # must not raise, must not write to the closed handle
        self.assertEqual(len(stream.writes), writes_after_first)
        self.assertFalse(sp._tty)

    def test_no_stream_writes_off_tty(self):
        sp, _ = _spinner(tty=False)
        sp._draw()
        sp.clear()
        sp.stop()  # all no-ops; nothing to assert beyond "does not raise"

    @patch(f"{_MOD}.time.sleep")
    @patch(f"{_MOD}.time.monotonic")
    def test_off_tty_emits_heartbeat(self, mono, _sleep):
        # start=0; loop check 0<10 True -> now=100 (>= heartbeat) -> logs -> next check 100<10 False
        mono.side_effect = [0.0, 0.0, 100.0, 100.0, 100.0]
        log = MagicMock()
        sp, _ = _spinner(tty=False, log=log)
        sp.spin_until(10.0)
        self.assertTrue(log.info.called)
        self.assertTrue(any("Working" in str(a) for c in log.info.call_args_list for a in c.args))


if __name__ == "__main__":
    unittest.main()
