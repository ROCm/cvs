'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Reusable in-place console spinner for long-running CVS operations (training,
inference, benchmarks, ...). See cvs/lib/utils/README.md.
'''

from __future__ import annotations

import sys
import time

from cvs.lib import globals


class ConsoleSpinner:
    """One-line in-place busy indicator, e.g. ``Training In Progress [ | ]``.

    Animates on the controlling terminal (``/dev/tty``) so it shows even under
    pytest's fd-level output capture and is never written to a ``--log-file``
    (that is driven by the logger). Off a TTY (CI / nohup) it does not animate;
    it emits an occasional heartbeat through the logger so batch logs still show
    progress. The caller wipes the spinner line with :meth:`clear` before it logs
    real output, and calls :meth:`stop` when done (also releases ``/dev/tty``).
    """

    _FRAMES = "|/-\\"

    def __init__(self, label, *, tick_s=0.25, heartbeat_s=60, log=None):
        self._label = label
        self._tick_s = tick_s
        self._heartbeat_s = heartbeat_s
        self._log = log or globals.log
        self._stream, self._owns_stream = self._open_console()
        self._tty = self._stream is not None
        self._i = 0
        self._active = False
        self._start = time.monotonic()
        self._last_heartbeat = 0.0

    @staticmethod
    def _open_console():
        """Return ``(stream, owns_stream)`` for the live terminal, else ``(None, False)``.

        Prefer the controlling terminal (``/dev/tty``) so the spinner shows even
        under pytest's fd-level capture (which points ``sys.__stdout__`` at a temp
        file); fall back to ``sys.__stdout__`` when it is itself a TTY.
        """
        try:
            return open("/dev/tty", "w", buffering=1), True  # closed in stop()
        except OSError:
            pass
        stream = sys.__stdout__
        if stream is not None and hasattr(stream, "isatty") and stream.isatty():
            return stream, False
        return None, False

    def _write(self, text):
        try:
            self._stream.write(text)
            self._stream.flush()
        except (OSError, ValueError):
            self._tty = False

    def clear(self):
        """Erase the on-screen spinner line so the next log output starts clean."""
        if self._tty and self._active:
            self._write("\r\033[K")
            self._active = False

    def _draw(self):
        """Write one spinner frame in place (TTY only)."""
        if not self._tty:
            return
        self._write(f"\r{self._label} [ {self._FRAMES[self._i % len(self._FRAMES)]} ]")
        self._active = True
        self._i += 1

    def spin_until(self, deadline):
        """Animate (TTY) or sleep with a periodic heartbeat (non-TTY) until deadline."""
        while time.monotonic() < deadline:
            if self._tty:
                self._draw()
                time.sleep(self._tick_s)
            else:
                now = time.monotonic() - self._start
                if now - self._last_heartbeat >= self._heartbeat_s:
                    self._log.info("%s (%.0fs elapsed)", self._label, now)
                    self._last_heartbeat = now
                time.sleep(min(self._tick_s * 8, max(0.0, deadline - time.monotonic())))

    def stop(self):
        """Clear the spinner line, drop to a fresh line (TTY only), and release /dev/tty.

        Idempotent: after the first call the stream is released and the TTY state
        is cleared, so a repeat call (e.g. a caller stopping a spinner that
        ``LogPoller.poll()``'s ``finally`` already stopped) is a genuine no-op
        rather than a swallowed write to the closed handle.
        """
        self.clear()
        if self._tty:
            self._write("\n")
        if self._owns_stream and self._stream is not None:
            try:
                self._stream.close()
            except OSError:
                pass
        self._owns_stream = False
        self._tty = False
        self._stream = None
