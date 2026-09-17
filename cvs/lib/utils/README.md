# cvs/lib/utils — shared library utilities

Framework-agnostic helpers usable by any CVS suite (training, inference,
benchmarks). This page covers the log-polling utilities:

- **`log_poller.LogPoller`** — poll per-node log files until a completion marker
  appears, streaming new lines and scanning for error signatures, with a live
  console spinner between polls.
- **`console_spinner.ConsoleSpinner`** — a one-line in-place busy indicator used
  by `LogPoller` (also usable standalone).

## Why

Every suite that launches a long-running workload in a container and tails
per-node logs needs the same loop: fetch the new log lines every few seconds,
show them, watch for failure signatures, detect completion, and keep the console
alive in between. `LogPoller` is that loop, so suites don't re-implement it.

## Requirements on the orchestrator

`LogPoller` talks only to the orchestrator (the `orch` fixture). The orchestrator
must expose:

- `orch.hosts` — ordered list of hosts.
- `orch.exec_cmd_list(cmd_list, print_console=False)` — run `cmd_list[i]` on
  `hosts[i]` in parallel and return `{host: output}` (output may be a `str` or a
  `{"output": str, ...}` dict; both are handled).

Both the baremetal and container orchestrators satisfy this. `print_console=False`
keeps the poller quiet (no command echo, host banners, or bulk output); the
poller surfaces exactly what it chooses.

## Quick start (convenience form)

Give it the per-node log paths, a completion regex, and your error patterns:

```python
from cvs.lib.utils.log_poller import LogPoller

LogPoller(
    orch,
    [f"{out_dir}/out-node{i}/train.log" for i in range(len(orch.hosts))],
    complete_pattern=r"completed step:\s*99,",   # grep'd on each node
    error_patterns={                              # {name: regex}, first match (case-insensitive) raises
        "NaN/Inf": r"loss:\s*nan",                # put the most specific/fatal patterns first
        "NCCL ERROR": r"NCCL ERROR|NCCL timeout",
        "OOM": r"RESOURCE_EXHAUSTED: Out of memory",
    },
    ignore_error_patterns=None,                   # optional {name: regex} allowlist for benign matches
    silent_poll=True,                             # False -> echo the per-node tail/grep commands (debug)
    stream_node=0,                                # which node's new lines to log
    label="Training In Progress",                 # spinner label
    error_label="Training",                       # noun in raised/timeout messages
    timeout_s=3600,
    drain_interval_s=10,                          # fetch new lines every 10s
).poll()                                          # returns on completion; raises on timeout/error
```

`error_patterns` is a single ordered dict — put NaN/Inf and other fatal
signatures first, since the **first** match wins. There is no separate
`nan_pattern` / `always_on_patterns`; the caller composes one dict (e.g. merge
a base set with per-run overrides).

## Constructor parameters

| Param | Default | Purpose |
|---|---|---|
| `orch`, `log_paths` | — | orchestrator + per-node log paths (`list` or `callable(i)->path`) |
| `complete_pattern` | `None` | regex grep'd per node to detect completion |
| `complete_policy` | `"all"` | `"all"` (every node), `"any"`, or `"node0"` |
| `is_complete` | `None` | callable `()->bool`; **overrides** the pattern/policy |
| `error_patterns` | `{}` | `{name: regex}` scanned per chunk in order, case-insensitive; first match raises |
| `ignore_error_patterns` | `None` | `{name: regex}`; an error match whose offending line also matches one of these is skipped (benign noise) |
| `scan_chunk` | `None` | callable `(host, node_idx, text)` that raises; **overrides** the pattern scan |
| `stream_node` | `0` | which node's new lines to log each drain (named once at start) |
| `error_label` | `"Run"` | word used in raised/timeout messages (e.g. `"Training"`) |
| `label` | `"In Progress"` | spinner label |
| `timeout_s` | `None` | total budget (required at construct or `poll()` time) |
| `drain_interval_s` | `10` | seconds between log drains |
| `silent_poll` | `True` | keep the tail/grep node commands off the console; `False` prints them (`print_console=True`) for debugging the poll loop |
| `tick_s`, `heartbeat_s` | `0.330`, `60` | spinner tick / off-TTY heartbeat cadence (passed through to `ConsoleSpinner`, whose own standalone default `tick_s` is `0.25`) |
| `log` | `globals.log` | logger for streamed lines, failures, heartbeat |

Completion/error are "batteries included" via the patterns, but the
`is_complete` and `scan_chunk` callables let a suite plug in bespoke logic.

## Behavior

- **Draining:** `tail -n +<cursor>` per node with a per-node cursor, so each log
  line reaches the console/`--log-file` exactly once (no repeated `tail` window).
- **Streaming:** `stream_node` is named once at poll start (e.g. `"streaming node
  0 log; polling every 10s"`); thereafter its new lines are logged **raw** (no
  per-line prefix), first each cycle so a chunk is captured even if the scan
  below raises on it, then every node's chunk is scanned.
- **Completion:** on the completion check it drains + scans once more (the final
  marker and anything after it — e.g. a shutdown traceback or last-step NaN),
  scanning **every** node so a worker-only failure isn't missed.
- **Errors:** `scan()` raises `RuntimeError` on the first `error_patterns` match
  (insertion order, case-insensitive), logging the full offending chunk first.
  A match is skipped when its offending line also matches an
  `ignore_error_patterns` entry (e.g. a benign shutdown warning that overlaps a
  broad signature), so scanning continues to any real error in the chunk.
- **Verbosity:** the poller runs quietly (`silent_poll=True`); pass
  `silent_poll=False` to echo the per-node `tail`/`grep` commands and their
  output (`print_console=True`) when debugging the poll loop itself.
- **Timeout:** raises `RuntimeError` when `timeout_s` elapses before completion.

## ConsoleSpinner

`LogPoller` runs a `ConsoleSpinner` between drains. It animates
`"<label> [ | / - \\ ]"` in place on the controlling terminal (`/dev/tty`) so it
shows even under pytest's fd-level output capture and is **never** written to a
`--log-file`. Off a TTY (CI / `nohup`) it doesn't animate — it emits a periodic
heartbeat through the logger. Standalone use:

```python
from cvs.lib.utils.console_spinner import ConsoleSpinner
import time

sp = ConsoleSpinner("Warming up")
try:
    while not done():
        sp.spin_until(time.monotonic() + 5)  # animate ~5s
        sp.clear()                            # wipe the line before logging
        log.info("...progress...")
finally:
    sp.stop()                                 # newline + release /dev/tty
```

## Overriding completion / scanning with callables

When a grep pattern or a flat `error_patterns` dict isn't enough, pass callables
(they take precedence over `complete_pattern` / `error_patterns`):

```python
# is_complete: bespoke completion (e.g. a marker file, an HTTP health check,
# or a rank-specific condition). Called each poll cycle; return True when done.
def served():
    out = orch.exec_cmd_list(
        ["test -f /workspace/DONE && echo 1 || true"] * len(orch.hosts),
        print_console=False,
    )
    return all(str(v).strip() == "1" for v in out.values())

# scan_chunk: bespoke error logic (e.g. parse JSON lines, or a stateful check).
# Raise to fail the run; the poller does not catch it.
def scan_chunk(host, node_idx, text):
    if "CUDA error" in text or '"status": "failed"' in text:
        raise RuntimeError(f"inference failure on {host} (node {node_idx}): {text[-500:]}")

LogPoller(
    orch,
    [f"/logs/node{i}.log" for i in range(len(orch.hosts))],
    is_complete=served,        # overrides complete_pattern / complete_policy
    scan_chunk=scan_chunk,     # overrides error_patterns
    stream_node=0,
    label="vLLM serving",
    error_label="vLLM",
    timeout_s=1800,
).poll()
```

## Adapting a suite

1. Ensure logs are written per node to known paths.
2. Pick a `complete_pattern` (or pass an `is_complete` callable).
3. Provide an ordered `error_patterns` dict (fatal/specific first), or a
   `scan_chunk` callable for bespoke logic.
4. Construct a `LogPoller` and call `poll()`; wrap failures with your suite's
   `fail_test(...)` if desired.

See `cvs/lib/training/jaxmaxtext/jaxmaxtext_training_lib.py::MaxTextTrainingJob.poll_for_completion`
for a complete real-world caller (it merges NaN/Inf + always-on + config patterns
into one `error_patterns` dict). Unit tests live in `cvs/lib/utils/unittests/`.
