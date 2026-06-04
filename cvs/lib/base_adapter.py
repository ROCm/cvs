"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import abc
import re
import time
from typing import Any, Dict, List, Tuple

from cvs.lib.adapter_protocol import Progress
from cvs.lib.config.thresholds import ThresholdVerdict
from cvs.lib.failure_taxonomy import LivenessFailure, SafetyViolation, SetupFailure, VerificationFailure


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

        Today's ``ctx.executor`` is bound to a single host (the first host
        in the first role); for that case the probe runs once and the result
        keys every bound host with the same count. When per-role executors
        land (A2), this loop becomes per-host and each ``exec`` targets the
        correct node.
        """
        all_hosts = [h for hosts in ctx.bindings.values() for h in hosts]
        if not all_hosts:
            return {}
        try:
            out = ctx.executor.exec(self.gpu_probe_cmd, timeout=self.gpu_probe_timeout_s)
        except Exception as exc:  # noqa: BLE001 - boundary classification
            raise SetupFailure(f"{self.framework}: rocm-smi probe failed on bound host(s) {all_hosts}: {exc}") from exc
        text = out if isinstance(out, str) else "\n".join(str(v) for v in out.values())
        count = len(_GPU_LINE_RE.findall(text))
        if count == 0:
            raise SetupFailure(
                f"{self.framework}: rocm-smi reported 0 GPUs on bound host(s) {all_hosts}; raw output:\n{text[:500]}"
            )
        return {host: count for host in all_hosts}

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
