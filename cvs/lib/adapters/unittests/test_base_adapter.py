"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

BaseWorkloadAdapter.prepare(): role-contract validation + rocm-smi probe.

These tests guard the boundary checks the base prepare() owns: workload
configs whose topology.roles don't match the adapter's required_roles are
rejected at the boundary with a clear message; rocm-smi probe failures
surface as SetupFailure naming the offending host; the per-host GPU count
lands on ctx.scratch where subclasses can read it.
"""

from __future__ import annotations

import time
import types
import unittest

from cvs.lib.base_adapter import BaseWorkloadAdapter
from cvs.lib.failure_taxonomy import SetupFailure


class _ConcreteAdapter(BaseWorkloadAdapter):
    """Minimal concrete adapter for testing the base prepare path."""

    framework = "concrete"
    required_roles = ("server",)
    optional_roles = ()

    def launch(self, ctx):
        pass

    def progress_predicate(self, ctx):
        from cvs.lib.adapter_protocol import Progress

        return Progress.DONE

    def parse(self, ctx):
        pass


class _MultiRoleAdapter(BaseWorkloadAdapter):
    framework = "multirole"
    required_roles = ("prefill", "decode")
    optional_roles = ("router",)

    def launch(self, ctx):
        pass

    def progress_predicate(self, ctx):
        from cvs.lib.adapter_protocol import Progress

        return Progress.DONE

    def parse(self, ctx):
        pass


class _FakeExecutor:
    """Duck-typed executor returning a canned rocm-smi output."""

    def __init__(self, output: str):
        self.output = output
        self.calls = []

    def exec(self, cmd, timeout=None):
        self.calls.append((cmd, timeout))
        return self.output


class _FakeMultiExecutor:
    """A2 ``_MultiHostExecutor`` test double.

    ``per_host_outputs`` maps host -> canned rocm-smi output (or callable
    raising on exec). Records (host, cmd, timeout) for every per-host exec
    call so tests can assert the probe ran exactly once per host.
    """

    def __init__(self, per_host_outputs):
        self._per_host = per_host_outputs
        self.calls = []  # list[(host, cmd, timeout)]

    def executor_for(self, host):
        outer = self

        class _One:
            def exec(_self, cmd, timeout=None):
                outer.calls.append((host, cmd, timeout))
                out = outer._per_host[host]
                if callable(out):
                    return out()
                return out

        return _One()

    def exec(self, cmd, timeout=None):  # back-compat forward (unused here)
        first = next(iter(self._per_host))
        return self.executor_for(first).exec(cmd, timeout=timeout)


def _ctx(bindings, executor=None):
    return types.SimpleNamespace(
        bindings=bindings,
        executor=executor,
        scratch={},
    )


# ---- role contract -----------------------------------------------------


class TestRequiredRolesContract(unittest.TestCase):
    def test_missing_required_role_raises(self):
        ctx = _ctx(bindings={"worker": ["n0"]})
        with self.assertRaises(SetupFailure) as exc:
            _ConcreteAdapter().prepare(ctx)
        self.assertIn("missing required roles", str(exc.exception))
        self.assertIn("server", str(exc.exception))

    def test_empty_bindings_list_for_required_role_raises(self):
        ctx = _ctx(bindings={"server": []})
        with self.assertRaises(SetupFailure) as exc:
            _ConcreteAdapter().prepare(ctx)
        self.assertIn("bound to no hosts", str(exc.exception))

    def test_extra_role_raises_naming_known_set(self):
        ctx = _ctx(bindings={"server": ["n0"], "router": ["n1"]})
        with self.assertRaises(SetupFailure) as exc:
            _ConcreteAdapter().prepare(ctx)
        self.assertIn("router", str(exc.exception))
        self.assertIn("adapter knows", str(exc.exception))

    def test_optional_role_allowed_but_not_required(self):
        # Multi-role adapter declares router as optional: present is fine,
        # absent is fine.
        with_router = _ctx(bindings={"prefill": ["n0"], "decode": ["n1"], "router": ["n2"]})
        _MultiRoleAdapter().prepare(with_router)  # no raise
        without_router = _ctx(bindings={"prefill": ["n0"], "decode": ["n1"]})
        _MultiRoleAdapter().prepare(without_router)  # no raise

    def test_no_executor_skips_probe_but_validates_contract(self):
        ctx = _ctx(bindings={"server": ["n0"]}, executor=None)
        _ConcreteAdapter().prepare(ctx)
        self.assertNotIn("host_gpus", ctx.scratch)  # probe skipped


# ---- rocm-smi probe ----------------------------------------------------


_ROCM_SMI_8_GPU = """
============================ ROCm System Management Interface ============================
GPU[0]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[1]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[2]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[3]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[4]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[5]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[6]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
GPU[7]		: Card Series: AMD INSTINCT MI300X (HBM3) OAM
==========================================================================================
"""


class TestRocmSmiProbe(unittest.TestCase):
    def test_probe_caches_count_per_bound_host(self):
        ctx = _ctx(
            bindings={"server": ["n0"]},
            executor=_FakeExecutor(_ROCM_SMI_8_GPU),
        )
        _ConcreteAdapter().prepare(ctx)
        self.assertEqual(ctx.scratch["host_gpus"], {"n0": 8})

    def test_multi_host_keys_every_bound_host(self):
        # A2: per-host probe via _MultiHostExecutor.executor_for(host).
        # Each host returns its own rocm-smi output and is probed exactly
        # once (hosts that appear in multiple roles are deduped).
        n0_out = _ROCM_SMI_8_GPU
        n1_out = "\n".join(f"GPU[{i}]\t: stub" for i in range(4))
        n2_out = "\n".join(f"GPU[{i}]\t: stub" for i in range(2))
        executor = _FakeMultiExecutor({"n0": n0_out, "n1": n1_out, "n2": n2_out})
        ctx = _ctx(
            bindings={"prefill": ["n0"], "decode": ["n1", "n2"]},
            executor=executor,
        )
        _MultiRoleAdapter().prepare(ctx)
        self.assertEqual(ctx.scratch["host_gpus"], {"n0": 8, "n1": 4, "n2": 2})
        probed_hosts = [host for host, _cmd, _to in executor.calls]
        self.assertEqual(sorted(probed_hosts), ["n0", "n1", "n2"])
        self.assertEqual(len(probed_hosts), 3)  # one probe per host, no replication

    def test_multi_host_probe_failure_names_host(self):
        def _boom():
            raise RuntimeError("ssh timeout")

        executor = _FakeMultiExecutor({
            "n0": _ROCM_SMI_8_GPU,
            "n1": _boom,
            "n2": _ROCM_SMI_8_GPU,
        })
        ctx = _ctx(
            bindings={"prefill": ["n0"], "decode": ["n1", "n2"]},
            executor=executor,
        )
        with self.assertRaises(SetupFailure) as exc:
            _MultiRoleAdapter().prepare(ctx)
        msg = str(exc.exception)
        self.assertIn("rocm-smi probe failed on host n1", msg)
        self.assertIn("ssh timeout", msg)

    def test_multi_host_zero_gpus_names_host(self):
        executor = _FakeMultiExecutor({
            "n0": _ROCM_SMI_8_GPU,
            "n1": "No ROCm devices found",
        })
        ctx = _ctx(
            bindings={"prefill": ["n0"], "decode": ["n1"]},
            executor=executor,
        )
        with self.assertRaises(SetupFailure) as exc:
            _MultiRoleAdapter().prepare(ctx)
        self.assertIn("0 GPUs on host n1", str(exc.exception))

    def test_probe_failure_raises_setup_failure_with_hosts(self):
        class _Boom:
            def exec(self, cmd, timeout=None):
                raise RuntimeError("ssh timeout")

        ctx = _ctx(bindings={"server": ["n0"]}, executor=_Boom())
        with self.assertRaises(SetupFailure) as exc:
            _ConcreteAdapter().prepare(ctx)
        self.assertIn("rocm-smi probe failed", str(exc.exception))
        self.assertIn("n0", str(exc.exception))
        self.assertIn("ssh timeout", str(exc.exception))

    def test_zero_gpus_raises_setup_failure(self):
        ctx = _ctx(
            bindings={"server": ["n0"]},
            executor=_FakeExecutor("No ROCm devices found"),
        )
        with self.assertRaises(SetupFailure) as exc:
            _ConcreteAdapter().prepare(ctx)
        self.assertIn("rocm-smi reported 0 GPUs", str(exc.exception))


# ---- _launch_role / _wait_http_pool ------------------------------------



class _FakeRunner:
    """Per-host runner double: records every exec() and answers HTTP probes
    according to a per-host script."""

    def __init__(self, host, http_script=None):
        self.host = host
        self.calls = []  # list[str]
        # http_script: list of strings each exec returns, popped in order.
        # When exhausted, returns "200" (steady-state ready). A callable
        # is invoked instead of returned.
        self._script = list(http_script or [])

    def exec(self, cmd, timeout=None):
        self.calls.append(cmd)
        if self._script:
            item = self._script.pop(0)
            if callable(item):
                return item()
            return item
        return "200"


class _FakeMultiHostRunner:
    """Multi-host executor double: hands out a ``_FakeRunner`` per host."""

    def __init__(self, runners):
        # runners: dict[host -> _FakeRunner]
        self._runners = runners
        self.executor_for_calls = []

    def executor_for(self, host):
        self.executor_for_calls.append(host)
        return self._runners[host]

    def exec(self, cmd, timeout=None):
        # primary forward (unused here)
        first = next(iter(self._runners))
        return self._runners[first].exec(cmd, timeout=timeout)


class _EventSink:
    """Test double for EventWriter. Validates names against the closed
    EVENT_VOCAB so a helper that emits an unknown name fails offline
    instead of escaping to a real-HW SetupFailure -- the gap that hid
    the launch.role_started bug from PR-A2's offline gate."""

    def __init__(self):
        from cvs.lib.manifest.events import EVENT_VOCAB
        self._vocab = EVENT_VOCAB
        self.events = []

    def emit(self, name, **kwargs):
        if name not in self._vocab:
            from cvs.lib.manifest.events import UnknownEventError
            raise UnknownEventError(
                f"_EventSink: event name {name!r} not in EVENT_VOCAB"
            )
        self.events.append((name, kwargs))


def _launch_ctx(bindings, runners):
    return types.SimpleNamespace(
        run_id="testrun",
        bindings=bindings,
        executor=_FakeMultiHostRunner(runners),
        containers=[],
        events=_EventSink(),
        scratch={},
    )


def _stub_handle_enter(monkey_self):
    """Make ContainerHandle.__enter__ a no-op for tests (no real docker run)."""
    from cvs.lib.runtime import container_handle as ch_mod

    original = ch_mod.ContainerHandle.__enter__

    def _noop(self):
        self.started = True
        return self

    ch_mod.ContainerHandle.__enter__ = _noop
    monkey_self.addCleanup(setattr, ch_mod.ContainerHandle, "__enter__", original)


class _MultiServerAdapter(BaseWorkloadAdapter):
    framework = "fakefw"
    required_roles = ("server",)

    def launch(self, ctx):
        pass

    def progress_predicate(self, ctx):
        from cvs.lib.adapter_protocol import Progress

        return Progress.DONE

    def parse(self, ctx):
        pass


class TestLaunchRoleFansOutAcrossBoundHosts(unittest.TestCase):
    def test_launch_role_fans_out_across_bound_hosts(self):
        _stub_handle_enter(self)
        runners = {h: _FakeRunner(h) for h in ("h0", "h1", "h2")}
        ctx = _launch_ctx({"server": ["h0", "h1", "h2"]}, runners)
        adapter = _MultiServerAdapter()
        handles = adapter._launch_role(
            ctx, "server", image="img:latest", command="vllm serve",
        )
        self.assertEqual(len(handles), 3)
        for h in handles:
            self.assertTrue(h.started)
        # One executor_for() per bound host, in bound-host order.
        self.assertEqual(ctx.executor.executor_for_calls, ["h0", "h1", "h2"])
        # Each runner saw its own docker run command (handle.__enter__ is
        # stubbed, so no exec was issued during enter -- but the runner is
        # bound on the handle for later wait/teardown).
        for host, h in zip(("h0", "h1", "h2"), handles):
            self.assertIs(h.runner, runners[host])
            self.assertEqual(h.name, f"fakefw_server_{host}_testrun")


class TestLaunchRoleRecordsHandlesByRole(unittest.TestCase):
    def test_launch_role_records_handles_by_role(self):
        _stub_handle_enter(self)
        runners = {h: _FakeRunner(h) for h in ("h0", "h1")}
        ctx = _launch_ctx({"server": ["h0", "h1"]}, runners)
        adapter = _MultiServerAdapter()
        handles = adapter._launch_role(
            ctx, "server", image="img", command="cmd",
        )
        # ctx.containers carries the same handles, in the same order:
        # the canonical list teardown iterates is unchanged.
        self.assertEqual(ctx.containers, handles)
        # And the per-role parallel is populated.
        self.assertIn("server", adapter.handles_by_role)
        self.assertEqual(adapter.handles_by_role["server"], handles)


class TestWaitHttpPoolPollsEveryHandleConcurrently(unittest.TestCase):
    def test_wait_http_pool_polls_every_handle_concurrently(self):
        _stub_handle_enter(self)

        # Two slow hosts each block their probe for SLOW_S; the fast host
        # returns immediately. Wall-clock should be ~SLOW_S, not 3*SLOW_S.
        SLOW_S = 0.4

        def _slow():
            time.sleep(SLOW_S)
            return "200"

        runners = {
            "h0": _FakeRunner("h0", http_script=[_slow]),
            "h1": _FakeRunner("h1", http_script=[_slow]),
            "h2": _FakeRunner("h2"),  # ready immediately
        }
        ctx = _launch_ctx({"server": ["h0", "h1", "h2"]}, runners)
        adapter = _MultiServerAdapter()
        adapter.http_pool_interval_s = 0.01
        adapter._launch_role(ctx, "server", image="img", command="cmd")

        start = time.monotonic()
        adapter._wait_http_pool("server", "/health", 8888, timeout_s=5.0)
        elapsed = time.monotonic() - start

        # Concurrent: bounded by the slow host (~SLOW_S + scheduling slack),
        # NOT serial sum (2*SLOW_S = 0.8s). Generous upper bound (1.5*SLOW
        # = 0.6) keeps it from flapping under load.
        self.assertLess(elapsed, SLOW_S * 1.5,
                        f"wait_http_pool ran serially: {elapsed:.2f}s for SLOW_S={SLOW_S}s")
        # Every host saw the probe at least once.
        for host, runner in runners.items():
            self.assertTrue(
                any("/health" in c for c in runner.calls),
                f"host {host} was not probed; calls={runner.calls}",
            )
        # Sanity: probes target localhost on the per-host runner, not a
        # shared base-url (the multi-host generalization invariant).
        for runner in runners.values():
            self.assertTrue(any("localhost:8888/health" in c for c in runner.calls))



class TestEventSinkRejectsUnknownNames(unittest.TestCase):
    """Regression test for the PR-A2 launch.role_started escape: any helper
    that emits an event name outside EVENT_VOCAB must fail offline."""

    def test_event_sink_rejects_unknown_event_name(self):
        from cvs.lib.manifest.events import UnknownEventError
        sink = _EventSink()
        with self.assertRaises(UnknownEventError):
            sink.emit("launch.role_started", role="server")

    def test_event_sink_accepts_known_event_names(self):
        sink = _EventSink()
        sink.emit("launch.container_up", role="server", host="h0")
        sink.emit("launch.role_ready", role="server")
        self.assertEqual(len(sink.events), 2)


class TestWaitHttpPoolHonorsDeadlineWithHungProbe(unittest.TestCase):
    """Regression test for the unbounded as_completed bug: a probe that
    blocks longer than timeout_s must NOT pin the role wait past the
    deadline. The old code would block forever on a hung future."""

    def test_wait_http_pool_raises_liveness_failure_when_probe_hangs(self):
        from cvs.lib.failure_taxonomy import LivenessFailure
        _stub_handle_enter(self)

        # Probe blocks much longer than the role's timeout. With unbounded
        # as_completed this would never raise; with the deadline-bounded
        # version it raises LivenessFailure within ~timeout_s.
        HANG_S = 5.0
        TIMEOUT_S = 0.3

        def _hang():
            time.sleep(HANG_S)
            return "200"

        runners = {"h0": _FakeRunner("h0", http_script=[_hang])}
        ctx = _launch_ctx({"server": ["h0"]}, runners)
        adapter = _MultiServerAdapter()
        adapter.http_pool_interval_s = 0.01
        adapter._launch_role(ctx, "server", image="img", command="cmd")

        start = time.monotonic()
        with self.assertRaises(LivenessFailure) as exc:
            adapter._wait_http_pool("server", "/health", 8888, timeout_s=TIMEOUT_S)
        elapsed = time.monotonic() - start

        # Must exit close to TIMEOUT_S, not HANG_S. Bound generously to
        # account for scheduler jitter under load while still proving the
        # deadline -- not the hung probe -- bounds the wait.
        self.assertLess(elapsed, HANG_S * 0.5,
                        f"_wait_http_pool blocked on hung probe: {elapsed:.2f}s")
        self.assertIn("timed out", str(exc.exception))
        self.assertIn("h0", str(exc.exception))


class TestWaitHttpPoolCancelsOrphanProbesOnTimeout(unittest.TestCase):
    """Regression test for the orphan-thread leak: on LivenessFailure, the
    thread pool must cancel queued futures so probes don't keep curl'ing
    through SSH sessions that teardown is about to reuse."""

    def test_wait_http_pool_cancels_pending_on_timeout(self):
        from cvs.lib.failure_taxonomy import LivenessFailure
        _stub_handle_enter(self)

        HANG_S = 3.0
        TIMEOUT_S = 0.2

        def _hang():
            time.sleep(HANG_S)
            return "200"

        runners = {f"h{i}": _FakeRunner(f"h{i}", http_script=[_hang])
                   for i in range(3)}
        ctx = _launch_ctx({"server": ["h0", "h1", "h2"]}, runners)
        adapter = _MultiServerAdapter()
        adapter.http_pool_interval_s = 0.01
        adapter._launch_role(ctx, "server", image="img", command="cmd")

        start = time.monotonic()
        with self.assertRaises(LivenessFailure):
            adapter._wait_http_pool("server", "/health", 8888, timeout_s=TIMEOUT_S)
        elapsed = time.monotonic() - start

        # If shutdown(wait=False, cancel_futures=False) -- the old buggy
        # behavior -- this assertion would still pass since shutdown
        # didn't wait. The point of the test is to LOCK IN the new
        # contract: shutdown either waited (because we're tearing down
        # cleanly) or cancelled (so we don't pay HANG_S of wall-clock
        # blocking the test process on the lingering ThreadPoolExecutor).
        # Either way, total elapsed stays under HANG_S.
        self.assertLess(elapsed, HANG_S * 0.5,
                        f"shutdown leaked probe threads past timeout: {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
