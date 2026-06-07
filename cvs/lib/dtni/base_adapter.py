"""DTNI v1 BaseWorkloadAdapter — trimmed from dev/dtni.

Differences vs dev/dtni:
- Single WorkloadError (no SetupFailure / LivenessFailure / SafetyViolation / VerificationFailure).
- verify() moved out (Job calls cvs.lib.dtni.verdict.evaluate_all directly).
- teardown captures into ctx.logs[name] -> str (no FS write here; artifact_writer owns FS).
- No events.emit hard dep — uses ctx.events which is a no-op by default.
"""

from __future__ import annotations

import abc
import concurrent.futures
import re
import shlex
import time
from typing import Any, Callable, Dict, List, Optional

from cvs.lib.dtni.errors import WorkloadError
from cvs.lib.dtni.container_handle import ContainerHandle

_GPU_LINE_RE = re.compile(r"^GPU\[(\d+)\]", re.MULTILINE)


class BaseWorkloadAdapter(abc.ABC):
    framework: str = "base"

    required_roles: tuple[str, ...] = ("server",)
    optional_roles: tuple[str, ...] = ()

    poll_interval_s: float = 5.0
    completion_timeout_s: float = 3600.0
    gpu_probe_cmd: str = "rocm-smi --showproductname 2>/dev/null"
    gpu_probe_timeout_s: float = 30.0
    http_pool_interval_s: float = 5.0

    def __init__(self) -> None:
        self.handles_by_role: Dict[str, List[ContainerHandle]] = {}

    # ---- prepare -------------------------------------------------------

    def prepare(self, ctx: Any) -> None:
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
            raise WorkloadError(
                f"{self.framework}: workload missing required roles {missing}; "
                f"declared {sorted(declared)}"
            )
        empty_required = sorted(r for r in required if not bindings.get(r))
        if empty_required:
            raise WorkloadError(
                f"{self.framework}: required roles {empty_required} bound to no hosts"
            )
        extra = sorted(declared - required - optional)
        if extra:
            raise WorkloadError(
                f"{self.framework}: workload declared unknown roles {extra}; "
                f"adapter knows {sorted(required | optional)}"
            )

    def _discover_gpu_count(self, ctx: Any) -> Dict[str, int]:
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
        result: Dict[str, int] = {}
        for host in all_hosts:
            host_exec = per_host_exec(host) if per_host_exec else ctx.executor
            try:
                out = host_exec.exec(self.gpu_probe_cmd, timeout=self.gpu_probe_timeout_s)
            except Exception as exc:
                raise WorkloadError(
                    f"{self.framework}: rocm-smi probe failed on host {host}: {exc}"
                ) from exc
            text = out if isinstance(out, str) else "\n".join(str(v) for v in out.values())
            count = len(_GPU_LINE_RE.findall(text))
            if count == 0:
                raise WorkloadError(
                    f"{self.framework}: rocm-smi reported 0 GPUs on {host}; output[:500]:\n{text[:500]}"
                )
            result[host] = count
        return result

    # ---- multi-role launch helpers -------------------------------------

    def _register(self, role: str, handle: ContainerHandle, ctx: Any) -> None:
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
        per_host_exec = getattr(ctx.executor, "executor_for", None)
        hosts: List[str] = list(ctx.bindings.get(role) or [])
        if not hosts:
            raise WorkloadError(f"{self.framework}: _launch_role({role!r}) called with empty bindings")
        handles: List[ContainerHandle] = []
        for host in hosts:
            runner = per_host_exec(host) if per_host_exec is not None else ctx.executor
            kwargs: Dict[str, Any] = dict(spec_kwargs)
            host_env: Dict[str, str] = dict(env or {})
            if per_host_kwargs_fn is not None:
                overrides = per_host_kwargs_fn(host) or {}
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
            handles.append(handle)
        return handles

    def _wait_http_pool(self, role: str, path: str, port: int, timeout_s: float) -> None:
        handles = list(self.handles_by_role.get(role) or [])
        if not handles:
            raise WorkloadError(f"{self.framework}: _wait_http_pool({role!r}) called with no handles")
        url = f"http://localhost:{port}{path}"
        probe_cmd = f"curl -s -o /dev/null -w '%{{http_code}}' {url}"

        def _ready(handle: ContainerHandle) -> bool:
            try:
                out = handle.runner.exec(probe_cmd)
            except Exception:
                return False
            text = out if isinstance(out, str) else "\n".join(str(v) for v in out.values())
            return "200" in text

        deadline = time.monotonic() + timeout_s
        ready: set = set()
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(handles)))
        timed_out = False
        try:
            while True:
                pending = [h for h in handles if id(h) not in ready]
                if not pending:
                    return
                futures = {ex.submit(_ready, h): h for h in pending}
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    for fut in concurrent.futures.as_completed(futures, timeout=remaining):
                        if fut.result():
                            ready.add(id(futures[fut]))
                except concurrent.futures.TimeoutError:
                    pass
                if all(id(h) in ready for h in handles):
                    return
                if time.monotonic() >= deadline:
                    missing = [h.name for h in handles if id(h) not in ready]
                    timed_out = True
                    raise WorkloadError(
                        f"{self.framework}: _wait_http_pool({role!r}) timed out after "
                        f"{timeout_s}s; not ready: {missing}"
                    )
                time.sleep(self.http_pool_interval_s)
        finally:
            ex.shutdown(wait=not timed_out, cancel_futures=True)

    # ---- lifecycle (subclasses) ----------------------------------------

    @abc.abstractmethod
    def launch(self, ctx: Any) -> None: ...

    def progress_predicate(self, ctx: Any) -> str:
        """Return 'running' | 'done' | 'broken'. Default: always done after launch."""
        return "done"

    @abc.abstractmethod
    def parse(self, ctx: Any) -> None: ...

    def await_completion(self, ctx: Any) -> None:
        deadline = time.monotonic() + self.completion_timeout_s
        while True:
            state = self.progress_predicate(ctx)
            if state == "done":
                return
            if state == "broken":
                raise WorkloadError(f"{self.framework}: progress predicate broke mid-run")
            if time.monotonic() >= deadline:
                raise WorkloadError(
                    f"{self.framework}: await_completion timed out after {self.completion_timeout_s}s"
                )
            time.sleep(self.poll_interval_s)

    def teardown(self, ctx: Any) -> None:
        for handle in ctx.containers:
            try:
                artifacts = handle.capture()
                for name, text in artifacts.items():
                    ctx.logs[f"{handle.name}.{name}"] = text
            except Exception:
                pass
            finally:
                handle.remove()
