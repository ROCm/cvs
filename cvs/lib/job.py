"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
"""

from __future__ import annotations

import getpass
import importlib.metadata
import subprocess
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional

from cvs.lib.failure_taxonomy import (
    FailureCategory,
    SetupFailure,
    WorkloadFailure,
    category_of,
)
from cvs.lib.manifest.schema import (
    ConfigInputs,
    Identity,
    Manifest,
    PatternMatch,
    PhaseTiming,
    Verdicts,
)
from cvs.lib.manifest.sidecars import write_resolved_config
from cvs.lib.run_context import RunContext


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cvs_version() -> Optional[str]:
    try:
        return importlib.metadata.version("cvs")
    except Exception:  # noqa: BLE001
        return None


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _container_env(cfg) -> dict:
    """Return ``container.env`` AS-IS for the manifest (B7 / W7 removed).

    Reads ``cfg.container.env`` directly: a missing ``container`` attribute or
    a non-dict ``env`` shape is a config-layer regression that must surface as
    a boundary failure, not be silently flattened to ``{}``. Configs without a
    container section pass ``container = ContainerSpec()`` (env={}) so the
    happy path stays cheap.
    """
    return {str(k): str(v) for k, v in cfg.container.env.items()}


class Job:
    """Mode-blind driver for the six-phase workload lifecycle.

    No ``if mode == "training"`` branching: the same driver runs every
    adapter. Failures are recorded by *category at the raise site* (B4);
    ``teardown`` always runs in ``finally``; a :class:`Manifest` is always
    produced (even on failure), making the run forensically complete. The
    :class:`Job` itself owns phase-boundary events -- ``prepare.*``,
    ``parse.done``, ``teardown.*`` (B6). Sub-phase events
    (``launch.container_up``, ``launch.role_ready``, ``step``, ``request``,
    ...) belong to adapters, never here.
    """

    def __init__(self, adapter, ctx: RunContext, scanner=None) -> None:
        self.adapter = adapter
        self.ctx = ctx
        self.scanner = scanner

    def _phase(
        self,
        phases: List[PhaseTiming],
        name: str,
        fn: Callable[[], None],
    ) -> None:
        start = _now()
        t0 = time.monotonic()
        status = "complete"
        try:
            fn()
        except BaseException:
            status = "failed"
            raise
        finally:
            phases.append(
                PhaseTiming(
                    phase=name,
                    start=start,
                    end=_now(),
                    duration_s=round(time.monotonic() - t0, 4),
                    status=status,
                )
            )

    def _prepare(self) -> None:
        self.ctx.events.emit("prepare.start", run_id=self.ctx.run_id)
        seed = getattr(self.ctx.config, "seed", None)
        self.ctx.events.emit("seed.logged", seed=seed)
        try:
            self.adapter.prepare(self.ctx)
        except WorkloadFailure:
            raise
        except Exception as exc:  # noqa: BLE001 - classify at the boundary (B4)
            raise SetupFailure(f"prepare failed: {exc}") from exc
        self.ctx.events.emit("prepare.done", run_id=self.ctx.run_id)

    def _launch(self) -> None:
        try:
            self.adapter.launch(self.ctx)
        except WorkloadFailure:
            raise
        except Exception as exc:  # noqa: BLE001 - classify at the boundary (B4)
            raise SetupFailure(f"launch failed: {exc}") from exc

    def _parse(self) -> None:
        self.adapter.parse(self.ctx)
        self.ctx.events.emit("parse.done", run_id=self.ctx.run_id)

    def run(self) -> Manifest:
        ctx = self.ctx
        ctx.layout.ensure()
        phases: List[PhaseTiming] = []
        started_at = _now()
        overall = "complete"
        failure_category: Optional[str] = None
        # True iff a WorkloadFailure was raised at its own boundary. The
        # taxonomy is classify-at-the-raise-site: a downstream fatal pattern
        # may upgrade an unclassified/generic verdict (B4 "fatal pattern
        # overrides an error verdict too") but MUST NOT clobber a
        # WorkloadFailure category.
        boundary_classified = False

        try:
            self._phase(phases, "prepare", self._prepare)
            self._phase(phases, "launch", self._launch)
            self._phase(phases, "await", lambda: self.adapter.await_completion(ctx))
            self._phase(phases, "parse", self._parse)
            self._phase(phases, "verify", lambda: self.adapter.verify(ctx))
        except WorkloadFailure as exc:
            overall = "failed"
            failure_category = exc.category.value
            boundary_classified = True
        except Exception as exc:  # noqa: BLE001 - generic boundary failure (B4)
            overall = "failed"
            failure_category = category_of(exc).value
        finally:
            ctx.events.emit("teardown.start", run_id=ctx.run_id)
            try:
                self._phase(phases, "teardown", lambda: self.adapter.teardown(ctx))
            except Exception:  # noqa: BLE001 - teardown best-effort
                pass
            ctx.events.emit("teardown.done", run_id=ctx.run_id)

        # The remaining work (pattern scan, override, resolved-config sidecar,
        # _build_manifest, manifest.write) must always close the events
        # sidecar, even when one of these steps raises -- AGENTS.md invariant
        # "the events file is still closed" applies to harness bugs too.
        try:
            pattern_matches = self._scan_patterns()
            if any(m.severity == "fatal" for m in pattern_matches) and not boundary_classified:
                failure_category = FailureCategory.FAILURE_PATTERN_MATCHED.value
                if overall == "complete":
                    overall = "failed"

            resolved_written = False
            try:
                write_resolved_config(
                    ctx.layout.resolved_config_path,
                    ctx.config.model_dump(mode="json"),
                )
                resolved_written = True
            except OSError:
                # Disk error writing the YAML sidecar. The manifest must
                # NOT advertise a path that does not exist on disk -- a
                # downstream consumer (cvs export) would follow a stale
                # pointer.
                pass

            manifest = self._build_manifest(
                phases,
                started_at,
                overall,
                failure_category,
                pattern_matches,
                resolved_written=resolved_written,
            )
            manifest.write(ctx.layout.manifest_path)
            return manifest
        finally:
            try:
                ctx.events.close()
            except Exception:  # noqa: BLE001 - close itself is best-effort
                pass

    def _scan_patterns(self) -> List[PatternMatch]:
        """Collect pattern hits from adapter-recorded matches + scanner output.

        Adapter-recorded matches (``ctx.scratch['pattern_matches']``) ride
        along unchanged. When a scanner is wired (G6a B5), every captured log
        in ``ctx.logs`` is scanned. :class:`PatternHit` carries the id but
        not the severity, so the scanner MUST expose its compiled catalog as
        ``.patterns`` (each entry has ``.id`` and ``.severity``) -- otherwise
        we cannot know whether a hit is fatal, and silently defaulting to
        "warn" would disable the B4 override. Fail closed instead.
        """
        matches: List[PatternMatch] = []
        for rec in self.ctx.scratch.get("pattern_matches", []):
            matches.append(rec if isinstance(rec, PatternMatch) else PatternMatch(**rec))
        if self.scanner is None:
            return matches
        if not hasattr(self.scanner, "patterns"):
            raise TypeError(
                f"scanner must expose .patterns (catalog with .id + .severity); got {type(self.scanner).__name__}"
            )
        severities = {p.id: p.severity for p in self.scanner.patterns}
        for stream_name, text in self.ctx.logs.items():
            source = "dmesg" if "dmesg" in stream_name else "framework_log"
            for hit in self.scanner.scan(text, source=source):
                if hit.pattern_id not in severities:
                    raise ValueError(
                        f"scanner emitted hit for pattern_id {hit.pattern_id!r} "
                        "not present in scanner.patterns; severity is unknowable"
                    )
                severity = severities[hit.pattern_id]
                self.ctx.events.emit(
                    "pattern.matched",
                    id=hit.pattern_id,
                    severity=severity,
                    source=hit.source,
                )
                matches.append(
                    PatternMatch(
                        id=hit.pattern_id,
                        severity=severity,
                        line=hit.line,
                        node=stream_name,
                        source=hit.source,
                    )
                )
        return matches

    def _build_manifest(
        self,
        phases,
        started_at,
        overall,
        failure_category,
        pattern_matches,
        *,
        resolved_written: bool = False,
    ) -> Manifest:
        ctx = self.ctx
        cfg = ctx.config
        verdicts = Verdicts(
            overall_status=overall,
            failure_category=failure_category,
            threshold_verdicts=ctx.scratch.get("threshold_verdicts", []),
            pattern_matches=pattern_matches,
            scalars={k: float(v) for k, v in ctx.result.scalars.items()},
        )
        identity = Identity(
            run_id=ctx.run_id,
            test_id=ctx.layout.test_id,
            cell_id=getattr(ctx.cell, "id", ctx.layout.cell_id),
            config_hash=cfg.config_hash() if hasattr(cfg, "config_hash") else None,
            workload_hash=cfg.workload_hash() if hasattr(cfg, "workload_hash") else None,
            verification_hash=cfg.verification_hash() if hasattr(cfg, "verification_hash") else None,
            cvs_version=_cvs_version(),
            cvs_git_sha=_git_sha(),
            started_at=started_at,
            finished_at=_now(),
            invoker=getpass.getuser(),
        )
        # B7-half (G3+G5): populate ConfigInputs.env/commands AS-IS (security
        # removed; addendum §4 B7). Adapters stage commands they want
        # recorded onto ctx.scratch['commands']. resolved_config_path is only
        # advertised when the YAML sidecar was actually written (so the
        # manifest never points at a nonexistent file).
        config_inputs = ConfigInputs(
            resolved_config_path=str(ctx.layout.resolved_config_path) if resolved_written else None,
            model=getattr(cfg, "model", None),
            env=_container_env(cfg),
            commands=list(ctx.scratch.get("commands", [])),
            seed=getattr(cfg, "seed", None),
        )
        manifest = Manifest(
            identity=identity,
            config=config_inputs,
            phases=phases,
            verdicts=verdicts,
        )
        layout = ctx.layout
        manifest.sidecars.events = str(layout.events_path) if layout.events_path.exists() else None
        manifest.sidecars.samples = str(layout.samples_path) if layout.samples_path.exists() else None
        manifest.sidecars.trajectory = str(layout.trajectory_path) if layout.trajectory_path.exists() else None
        manifest.sidecars.logs_dir = str(layout.logs_dir)
        return manifest
