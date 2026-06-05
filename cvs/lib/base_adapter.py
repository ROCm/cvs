"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import abc
import concurrent.futures
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from cvs.lib.adapter_protocol import Progress
from cvs.lib.config.thresholds import ThresholdVerdict
from cvs.lib.failure_taxonomy import LivenessFailure, SafetyViolation, SetupFailure, VerificationFailure
from cvs.lib.runtime.container_handle import ContainerHandle


# rocm-smi line marker pattern (matches "GPU[0]", "GPU[1]", ...). One match per
# discovered device. Used by BaseWorkloadAdapter._discover_gpu_count.
_GPU_LINE_RE = re.compile(r"^GPU\[(\d+)\]", re.MULTILINE)


class BaseWorkloadAdapter(abc.ABC):
    """Concrete defaults most adapters inherit.

    Subclasses must implement the workload-specific steps (``launch``,
    ``progress_predicate``, ``parse``). The polling ``await_completion``, the
    threshold-driven ``verify``, the role-contract + rocm-smi-probing
    ``prepare``, and the forensics-capturing ``teardown`` are provided here
    so an adapter does not re-implement them.

    Role contract: subclasses declare which role names they expect via
    ``required_roles`` (default: single-role ``("server",)`` for inference).
    The base ``prepare`` validates ``ctx.bindings`` against this contract
    BEFORE any framework-specific work, raising ``SetupFailure`` with the
    expected vs. actual set on mismatch.

    Runtime GPU discovery: the base ``prepare`` probes ``rocm-smi`` on each
    bound host (one host at a time via the shared executor) and caches the
    per-host GPU count into ``ctx.scratch["host_gpus"]: Dict[str, int]``.
    Subclasses can read this to validate workload-specific GPU requirements
    (e.g. VllmAdapter checks ``tp <= min(host_gpus.values())``); failures
    raise ``SetupFailure`` naming the offending host.

    Multi-role launching: subclasses use ``_launch_role`` to fan out one
    ``ContainerHandle`` per host bound to a role (via
    ``ctx.executor.executor_for(host)``) and ``_wait_http_pool`` to
    concurrently gate readiness across every handle of a role. Both
    populate ``self.handles_by_role`` -- a role-indexed parallel of
    ``ctx.containers`` -- so role-specific teardown / wait stays cheap
    without changing the canonical container list teardown iterates.

    The ``ctx`` parameter is the ``RunContext`` introduced by G5b; it is
    typed as ``Any`` here for back-compat with adapters that pre-date the
    typed seam.
    """

    framework: str = "base"

    # Role contract -- subclasses override to declare what bindings shape
    # they expect. Default: single-role inference under the literal role
    # name "server" (matches the Topology shorthand ``nnodes: N`` expansion).
    required_roles: Tuple[str, ...] = ("server",)
    optional_roles: Tuple[str, ...] = ()

    # Tunable by subclasses or via config; await_completion honors these.
    poll_interval_s: float = 5.0
    completion_timeout_s: float = 3600.0

    # rocm-smi probe shape (overridable for sites with custom paths).
    gpu_probe_cmd: str = "rocm-smi --showproductname 2>/dev/null"
    gpu_probe_timeout_s: float = 30.0

    # _wait_http_pool tuning.
    http_pool_interval_s: float = 5.0

    def __init__(self) -> None:
        # Role-indexed parallel of ``ctx.containers``: populated by
        # ``_register`` / ``_launch_role`` so subclasses can teardown or
        # wait per role without scanning the global list. Stays empty for
        # adapters that bypass the helpers (e.g. unit tests that only
        # exercise ``prepare``).
        self.handles_by_role: Dict[str, List[ContainerHandle]] = {}

    # ---- prepare -------------------------------------------------------

    def prepare(self, ctx: Any) -> None:
        """Validate role contract; probe rocm-smi for per-host GPU count.

        Two boundary checks before any framework work:

        1. Role-contract: ``ctx.bindings`` keys must match
           ``required_roles ∪ optional_roles`` -- missing required keys
           raise ``SetupFailure``; unexpected keys raise ``SetupFailure``
           naming what the adapter knows about. This catches workload
           YAMLs whose ``topology.roles`` shape doesn't match the
           adapter (e.g. an sglang-disagg config dispatched to the
           vLLM adapter).

        2. GPU discovery: for each bound host across every role, ``rocm-smi
           --showproductname`` is executed via ``ctx.executor`` (a hostname-
           specific Pssh-backed executor in the dtni conftest). The number
           of ``GPU[N]`` lines is parsed; the per-host count is cached on
           ``ctx.scratch["host_gpus"]``. Subclass adapters consume this
           (e.g. TP validation) without re-probing.

        Skipping the probe is intentional when ``ctx.executor is None``
        (dry-run / unit-test mode); subclass checks should treat the
        absence of ``host_gpus`` as "discovery was skipped, trust the
        config" and not as zero GPUs.
        """
        self._validate_role_contract(ctx)
        if ctx.executor is None:
            return
        ctx.scratch["host_gpus"] = self._discover_gpu_count(ctx)

    def _validate_role_contract(self, ctx: Any) -> None:
        bindings = dict(ctx.bindings or {})
        declared = set(bindings)
        required = set(self.required_roles)
        optional = set(self.optional_roles)
        missing = sorted(required - declared)
        if missing:
            raise SetupFailure(
                f"{self.framework}: workload config missing required roles {missing}; "
                f"declared {sorted(declared)} (adapter requires {sorted(required)})"
            )
        # Empty bindings lists for a required role are also a failure
        # (binder skipped the cell but the run was still dispatched).
        empty_required = sorted(r for r in required if not bindings.get(r))
        if empty_required:
            raise SetupFailure(
                f"{self.framework}: required roles {empty_required} bound to no hosts "
                f"(bindings={bindings}); check the cluster file has enough nodes"
            )
        extra = sorted(declared - required - optional)
        if extra:
            raise SetupFailure(
                f"{self.framework}: workload config declared unknown roles {extra}; "
                f"adapter knows {sorted(required | optional)}"
            )

    def _discover_gpu_count(self, ctx: Any) -> Dict[str, int]:
        """Probe each bound host once; return ``{hostname: n_gpus}``.

        If ``ctx.executor`` exposes ``executor_for(host)`` (the A2
        ``_MultiHostExecutor``), probe each bound host through its own
        per-host executor and report the real per-host count. A failure on
        any single host raises ``SetupFailure`` naming the host; a 0-GPU
        result on any host likewise raises naming the host. Hosts that
        appear in multiple roles are probed once.

        Fallback (executor has no ``executor_for``): the old single-shot
        probe whose result is replicated across every bound host. Kept so
        unit tests can pass a bare duck-typed executor.
        """
        all_hosts: list[str] = []
        seen: set[str] = set()
        for hosts in ctx.bindings.values():
            for h in hosts:
                if h not in seen:
                    seen.add(h)
                    all_hosts.append(h)
        if not all_hosts:
            return {}

        per_host_exec = getattr(ctx.executor, "executor_for", None)
        if per_host_exec is None:
            try:
                out = ctx.executor.exec(self.gpu_probe_cmd, timeout=self.gpu_probe_timeout_s)
            except Exception as exc:  # noqa: BLE001 - boundary classification
                raise SetupFailure(
                    f"{self.framework}: rocm-smi probe failed on bound host(s) {all_hosts}: {exc}"
                ) from exc
            text = out if isinstance(out, str) else "\n".join(str(v) for v in out.values())
            count = len(_GPU_LINE_RE.findall(text))
            if count == 0:
                raise SetupFailure(
                    f"{self.framework}: rocm-smi reported 0 GPUs on bound host(s) {all_hosts}; raw output:\n{text[:500]}"
                )
            return {host: count for host in all_hosts}

        result: Dict[str, int] = {}
        for host in all_hosts:
            host_exec = per_host_exec(host)
            try:
                out = host_exec.exec(self.gpu_probe_cmd, timeout=self.gpu_probe_timeout_s)
            except Exception as exc:  # noqa: BLE001 - boundary classification
                raise SetupFailure(
                    f"{self.framework}: rocm-smi probe failed on host {host}: {exc}"
                ) from exc
            text = out if isinstance(out, str) else "\n".join(str(v) for v in out.values())
            count = len(_GPU_LINE_RE.findall(text))
            if count == 0:
                raise SetupFailure(
                    f"{self.framework}: rocm-smi reported 0 GPUs on host {host}; raw output:\n{text[:500]}"
                )
            result[host] = count
        return result

    # ---- multi-role launch helpers -------------------------------------

    def _register(self, role: str, handle: ContainerHandle, ctx: Any) -> None:
        """Append ``handle`` to both ``ctx.containers`` (the canonical list
        ``teardown`` iterates) and ``self.handles_by_role[role]`` (a role-
        indexed parallel for role-aware waits / per-role teardown).
        """
        ctx.containers.append(handle)
        self.handles_by_role.setdefault(role, []).append(handle)

    def _launch_role(
        self,
        ctx: Any,
        role: str,
        *,
        image: str,
        env: Optional[Dict[str, str]] = None,
        command: Optional[str] = None,
        per_host_kwargs_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        **spec_kwargs: Any,
    ) -> List[ContainerHandle]:
        """Fan out one ``ContainerHandle`` per host bound to ``role``.

        For each host in ``ctx.bindings[role]``:

        - Build a ``ContainerHandle`` whose ``runner`` is scoped to that
          host via ``ctx.executor.executor_for(host)`` (PR-1's
          ``_MultiHostExecutor``). Without ``executor_for``, falls back
          to the shared ``ctx.executor`` -- preserves single-host shape
          for fakes / single-host clusters where the same executor is
          the right runner for the one bound host.
        - Apply ``per_host_kwargs_fn(host)`` overrides on top of the
          shared ``spec_kwargs`` (used by future disagg adapters to
          differ ports / env per host).
        - Name the container ``{framework}_{role}_{host}_{run_id}`` so
          forensics from multiple containers on the same node don't
          collide.
        - Enter the handle (``docker run -d``) and register it.

        Returns the list of started handles in bound-host order so the
        caller can wire endpoints (router config etc).

        Caller composes ``env`` / ``command`` / ``volumes`` / ``ports`` /
        ``network`` etc. Helper only owns the per-host loop, the per-host
        executor binding, and the registration -- it does NOT make
        framework-specific decisions about the env-merge precedence or
        the command shape.
        """
        per_host_exec = getattr(ctx.executor, "executor_for", None)
        hosts: List[str] = list(ctx.bindings.get(role) or [])
        if not hosts:
            raise SetupFailure(
                f"{self.framework}: _launch_role({role!r}) called with empty bindings; "
                f"prepare() should have caught this"
            )
        handles: List[ContainerHandle] = []
        for host in hosts:
            runner = per_host_exec(host) if per_host_exec is not None else ctx.executor
            kwargs: Dict[str, Any] = dict(spec_kwargs)
            host_env: Dict[str, str] = dict(env or {})
            if per_host_kwargs_fn is not None:
                overrides = per_host_kwargs_fn(host) or {}
                # ``env`` from per-host overrides merges INTO the shared env
                # (host-specific keys win); other kwargs replace wholesale
                # so the override can swap a port mapping etc.
                host_env.update(overrides.pop("env", {}) or {})
                kwargs.update(overrides)
            kwargs["env"] = host_env
            handle = ContainerHandle(
                image=image,
                run_id=ctx.run_id,
                runner=runner,
                name=f"{self.framework}_{role}_{host}_{ctx.run_id}",
                command=command,
                **kwargs,
            )
            handle.__enter__()
            self._register(role, handle, ctx)
            ctx.events.emit(
                "launch.container_up",
                run_id=ctx.run_id,
                container=handle.name,
                role=role,
                host=host,
            )
            handles.append(handle)
        return handles

    def _wait_http_pool(self, role: str, path: str, port: int, timeout_s: float) -> None:
        """Concurrently poll HTTP ``{path}`` on every handle of ``role``.

        Each handle's runner issues
        ``curl -s -o /dev/null -w '%{http_code}' http://localhost:{port}{path}``
        in its own thread so a slow host doesn't serialize the rest of the
        pool -- wall-clock is bounded by the slowest single host's
        readiness time, not the sum. On ``timeout_s`` elapsing without
        every handle returning HTTP 200, raises ``LivenessFailure`` naming
        the host(s) that did not come ready.

        Used by single-role adapters too (vLLM's server is one handle in
        a pool of one); the migration to this helper is the load-bearing
        bit of PR-A2.
        """
        handles = list(self.handles_by_role.get(role) or [])
        if not handles:
            raise LivenessFailure(
                f"{self.framework}: _wait_http_pool({role!r}) called with no registered handles"
            )
        url = f"http://localhost:{port}{path}"
        probe_cmd = f"curl -s -o /dev/null -w '%{{http_code}}' {url}"

        def _ready(handle: ContainerHandle) -> bool:
            try:
                out = handle.runner.exec(probe_cmd)
            except Exception:  # noqa: BLE001 - probe failure handled by the poller
                return False
            text = out if isinstance(out, str) else "\n".join(str(v) for v in out.values())
            return "200" in text

        deadline = time.monotonic() + timeout_s
        ready: set = set()
        # max_workers caps at the pool size so each handle gets its own
        # thread (no head-of-line blocking on a small fixed worker count).
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(handles)))
        try:
            while True:
                pending = [h for h in handles if id(h) not in ready]
                if not pending:
                    return
                futures = {executor.submit(_ready, h): h for h in pending}
                for fut in concurrent.futures.as_completed(futures):
                    if fut.result():
                        ready.add(id(futures[fut]))
                if all(id(h) in ready for h in handles):
                    return
                if time.monotonic() >= deadline:
                    missing = [h.name for h in handles if id(h) not in ready]
                    raise LivenessFailure(
                        f"{self.framework}: _wait_http_pool({role!r}) timed out after "
                        f"{timeout_s}s; host(s) not ready: {missing}"
                    )
                time.sleep(self.http_pool_interval_s)
        finally:
            executor.shutdown(wait=False)

    # ---- lifecycle (subclasses) ----------------------------------------

    @abc.abstractmethod
    def launch(self, ctx: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def progress_predicate(self, ctx: Any) -> Progress:
        raise NotImplementedError

    @abc.abstractmethod
    def parse(self, ctx: Any) -> None:
        raise NotImplementedError

    def await_completion(self, ctx: Any) -> None:
        """Poll ``progress_predicate`` until DONE; classify BROKEN/timeout."""
        deadline = time.monotonic() + self.completion_timeout_s
        while True:
            state = self.progress_predicate(ctx)
            if state is Progress.DONE:
                return
            if state is Progress.BROKEN:
                ctx.events.emit("safety.violated", run_id=ctx.run_id)
                raise SafetyViolation("progress predicate broke mid-run")
            if time.monotonic() >= deadline:
                raise LivenessFailure(f"await_completion timed out after {self.completion_timeout_s}s")
            time.sleep(self.poll_interval_s)

    def verify(self, ctx: Any) -> List[ThresholdVerdict]:
        """Evaluate every configured threshold against ``ctx.result``.

        Verdicts are recorded on the context regardless of outcome (so the
        manifest captures passing thresholds too); a single failing threshold
        raises ``VerificationFailure`` -- classification happens at this raise
        site, never by post-hoc inspection.
        """
        verdicts = [threshold.evaluate(ctx.result) for threshold in ctx.config.thresholds]
        ctx.scratch["threshold_verdicts"] = verdicts
        failed = [v for v in verdicts if not v.passed]
        if failed:
            ctx.events.emit("verify.failed", failed=len(failed), total=len(verdicts))
            raise VerificationFailure(
                f"{len(failed)}/{len(verdicts)} thresholds failed",
                detail={"verdicts": [v.model_dump() for v in verdicts]},
            )
        ctx.events.emit("verify.passed", total=len(verdicts))
        return verdicts

    def teardown(self, ctx: Any) -> None:
        """Capture forensics from every registered container, then remove them.

        Always safe to call (best-effort): runs in the ``Job``'s ``finally`` so
        a crash never leaks containers. Captured artifacts are written under the
        run's ``logs/`` directory.
        """
        logs_dir = ctx.layout.logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        for handle in ctx.containers:
            try:
                artifacts = handle.capture()
                for name, text in artifacts.items():
                    (logs_dir / f"{handle.name}.{name}").write_text(text)
                    ctx.logs[f"{handle.name}.{name}"] = text
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
            finally:
                handle.remove()
