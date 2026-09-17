'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Reusable per-node log-tailing poll loop for long-running CVS workloads
(training, inference, benchmarks). See cvs/lib/utils/README.md.
'''

from __future__ import annotations

import re
import shlex
import time

from cvs.lib import globals
from cvs.lib.utils.console_spinner import ConsoleSpinner


def _result_text(result):
    """Normalize an orchestrator result value (str or ``{'output': str}``) to text."""
    if isinstance(result, str):
        return result
    return (result or {}).get("output", "") or ""


class LogPoller:
    """Poll per-node log files until a completion marker appears.

    Every ``drain_interval_s`` the poller: (1) fetches the NEW lines of each
    node's log (``tail`` from a per-node cursor), (2) streams ``stream_node``'s
    new lines through the logger, (3) scans every node's new chunk for error
    signatures (raising on the first match), and (4) checks for completion
    (``grep`` for ``complete_pattern``). Between drains a one-line
    :class:`ConsoleSpinner` shows the workload is alive. :meth:`poll` returns on
    completion and raises ``RuntimeError`` on timeout or an error signature.

    ``orch`` must expose ``hosts`` (ordered) and
    ``exec_cmd_list(cmd_list, print_console=False)`` where ``cmd_list[i]`` runs on
    ``hosts[i]`` (both the baremetal and container orchestrators satisfy this).

    Completion and error scanning are "batteries included" (pattern-driven) but
    can be fully overridden with the ``is_complete`` / ``scan_chunk`` callables
    for a workload with bespoke logic.
    """

    def __init__(
        self,
        orch,
        log_paths,
        *,
        # completion
        complete_pattern=None,
        complete_policy="all",
        is_complete=None,
        # per-chunk error scanning
        error_patterns=None,
        ignore_error_patterns=None,
        scan_chunk=None,
        # streaming / UX
        stream_node=0,
        error_label="Run",
        label="In Progress",
        # timing
        timeout_s=None,
        drain_interval_s=10,
        silent_poll=True,
        tick_s=0.330,
        heartbeat_s=60,
        log=None,
    ):
        self.orch = orch
        self.hosts = list(orch.hosts)
        self.num_nodes = len(self.hosts)
        self._log_paths = [log_paths(i) for i in range(self.num_nodes)] if callable(log_paths) else list(log_paths)
        if len(self._log_paths) != self.num_nodes:
            raise ValueError(f"log_paths ({len(self._log_paths)}) must match orch.hosts ({self.num_nodes})")
        if is_complete is None and not complete_pattern:
            raise ValueError("LogPoller needs either complete_pattern or an is_complete callable")
        if complete_policy not in ("all", "any", "node0"):
            raise ValueError(f"complete_policy must be all|any|node0, got {complete_policy!r}")

        self._complete_pattern = complete_pattern
        self._complete_policy = complete_policy
        self._is_complete_cb = is_complete
        self._error_patterns = dict(error_patterns or {})
        # Lines matching any ignore pattern are treated as NOT a real error, even
        # if they also match an error pattern (e.g. benign shutdown warnings).
        self._ignore_res = [re.compile(p, re.IGNORECASE) for p in (ignore_error_patterns or {}).values() if p]
        self._scan_cb = scan_chunk
        self._stream_node = stream_node
        self._error_label = error_label
        self._label = label
        self._timeout_s = timeout_s
        self._drain_interval_s = drain_interval_s
        # silent_poll=True (default) keeps the tail/grep node commands off the
        # console; pass silent_poll=False to print them (debugging the polling).
        self._print_console = not silent_poll
        self._tick_s = tick_s
        self._heartbeat_s = heartbeat_s
        self._log = log or globals.log
        self._cursor = [0] * self.num_nodes

    # ---------- draining ----------
    def drain(self):
        """Return ``{node_idx: new_text}`` for lines written since the last drain.

        Runs quietly by default (``silent_poll``) so the orchestrator does not
        re-echo the bulk output; the caller streams/scans it. Advances each node's
        cursor by the number of complete newline-terminated records so the next
        ``tail -n +`` starts on the same record GNU tail would.
        """
        cmd_list = [
            f"tail -n +{self._cursor[i] + 1} {shlex.quote(self._log_paths[i])} 2>/dev/null || true"
            for i in range(self.num_nodes)
        ]
        out = self.orch.exec_cmd_list(cmd_list, print_console=self._print_console)
        node_of = {h: i for i, h in enumerate(self.hosts)}
        new_by_node = {}
        for host, result in (out or {}).items():
            text = _result_text(result)
            i = node_of.get(host)
            if i is None or not text:
                continue
            # GNU tail -n +K counts newline-terminated records. splitlines() also
            # splits on \r and counts a last line with no \n, which would advance
            # the cursor past that record so the next tail skips it. Keep only
            # complete \n records; a trailing fragment is re-read next drain.
            if not text.endswith('\n'):
                last_nl = text.rfind('\n')
                if last_nl == -1:
                    continue
                text = text[: last_nl + 1]
            self._cursor[i] += text.count('\n')
            new_by_node[i] = text
        return new_by_node

    # ---------- completion ----------
    def is_complete(self):
        """True when the run has finished per ``complete_policy`` (or the callable)."""
        if self._is_complete_cb is not None:
            return bool(self._is_complete_cb())
        cmd_list = [
            f"grep -cE {shlex.quote(self._complete_pattern)} {shlex.quote(self._log_paths[i])} 2>/dev/null || true"
            for i in range(self.num_nodes)
        ]
        out = self.orch.exec_cmd_list(cmd_list, print_console=self._print_console)
        if not out:
            return False
        node_of = {h: i for i, h in enumerate(self.hosts)}
        matched = {}
        for host, result in out.items():
            i = node_of.get(host)
            if i is None:
                continue
            text = _result_text(result).strip()
            matched[i] = bool(text) and text != "0"
        if self._complete_policy == "any":
            return any(matched.values())
        if self._complete_policy == "node0":
            return matched.get(0, False)
        return len(matched) == self.num_nodes and all(matched.values())

    # ---------- per-chunk error scanning ----------
    def scan(self, host, node_idx, text):
        """Raise ``RuntimeError`` on the first ``error_patterns`` match in ``text``.

        Patterns are scanned in insertion order (case-insensitive); the caller
        controls precedence by ordering the dict (e.g. NaN/Inf and fatal
        signatures before broader ones). A match whose offending line also matches
        an ``ignore_error_patterns`` entry is skipped (not a real error). The FULL
        offending chunk is logged before raising, since the message is truncated.
        """
        if self._scan_cb is not None:
            self._scan_cb(host, node_idx, text)
            return
        if not text:
            return
        for name, pattern in self._error_patterns.items():
            if not pattern:
                continue
            for match in re.finditer(pattern, text, re.IGNORECASE):
                line = self._offending_line(text, match.start())
                if self._ignore_res and any(ig.search(line) for ig in self._ignore_res):
                    continue  # offending line is explicitly ignored -> not a real error
                self._log.error(
                    "%s FAILURE chunk (node %s / %s):\n%s", self._error_label, node_idx, host, text.rstrip()
                )
                raise RuntimeError(f"{self._error_label} error '{name}' on {host} (node {node_idx}): {text[-500:]}")

    @staticmethod
    def _offending_line(text, pos):
        """The single log line containing offset ``pos`` (for ignore-pattern checks)."""
        start = text.rfind("\n", 0, pos) + 1
        end = text.find("\n", pos)
        return text[start:] if end == -1 else text[start:end]

    def _stream(self, spinner, new_by_node):
        """Log the streamed node's new lines (after clearing the spinner line)."""
        streamed = (new_by_node.get(self._stream_node) or "").rstrip()
        if streamed:
            spinner.clear()
            self._log.info("%s", streamed)

    # ---------- main loop ----------
    def poll(self, timeout_s=None):
        """Poll until completion (returns) or timeout / error signature (raises)."""
        timeout_s = timeout_s if timeout_s is not None else self._timeout_s
        if timeout_s is None:
            raise ValueError("timeout_s is required (constructor or poll())")

        start = time.monotonic()
        deadline = start + timeout_s
        spinner = ConsoleSpinner(self._label, tick_s=self._tick_s, heartbeat_s=self._heartbeat_s, log=self._log)
        # State the followed node ONCE (the streamed lines below carry no per-line
        # prefix); which node is being tailed is otherwise not obvious.
        self._log.info("streaming node %d log; polling every %ds", self._stream_node, self._drain_interval_s)
        try:
            while True:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"{self._error_label} did not complete within {timeout_s:.0f}s")

                new_by_node = self.drain()
                # Stream the followed node FIRST so the chunk reaches the log even
                # if the scan below raises on this same chunk (e.g. a traceback).
                self._stream(spinner, new_by_node)
                for i, text in new_by_node.items():
                    self.scan(self.hosts[i], i, text)

                if self.is_complete():
                    # Flush the tail written between the drain above and this check
                    # (final marker + anything after it). Scan EVERY node so a
                    # worker-only failure in this window is not missed.
                    tail = self.drain()
                    self._stream(spinner, tail)
                    for i, text in tail.items():
                        self.scan(self.hosts[i], i, text)
                    spinner.clear()
                    self._log.info("%s complete (%.0fs elapsed)", self._label, time.monotonic() - start)
                    return

                spinner.spin_until(min(time.monotonic() + self._drain_interval_s, deadline))
        finally:
            spinner.stop()
