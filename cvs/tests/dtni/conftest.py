"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

DTNI pytest layer -- BAREBONES (PR-X: barebones-G6a, addendum §9
"G6 re-split -- barebones-first").

Owns ONLY the B5 wiring + the ``workload_run`` fixture that drives one cell
through ``Job.run()``. Sweep expansion (``pytest_generate_tests``), marker
auto-derivation (``cvs.lib.markers``), tier deselection
(``pytest_collection_modifyitems`` + ``cvs.lib.tiers``), and
``@requires_benchmark`` land together in PR-Z (the matrix layer) once
PR-Y has a real vertical to slice. Until then this conftest runs every
collected test as a single cell.

A4 (suite-name -> directory) is owned by
``cvs/cli_plugins/list_plugin.py``; this conftest assumes it has already
resolved.
"""

from __future__ import annotations

import getpass
import os
import uuid

import pytest

from cvs.lib.run_context import RunContext
from cvs.lib.cluster.binder import bind
from cvs.lib.cluster.pool import load_cluster_file
from cvs.lib.config.loader import load_config_file
from cvs.lib.failure_pattern_scanner import FailurePatternScanner
from cvs.lib.job import Job
from cvs.lib.manifest.events import EventWriter
from cvs.lib.manifest.layout import RunLayout
from cvs.lib.manifest.schema import ConfigInputs, Identity, Manifest, Verdicts
from cvs.lib.registry import get_adapter


# ----- B5: scanner wiring (the contract the matrix-layer PR-Z will reuse) -----


def _build_job(adapter, ctx) -> Job:
    """Construct ``Job(adapter, ctx, scanner=FailurePatternScanner())`` (B5).

    Extracted so the wiring contract is unit-testable without spinning up a
    full pytest session, RunContext, or executor. If you refactor the
    fixture, this function MUST keep threading ``scanner=`` -- the
    Integration Milestone's tier-1 dmesg assertions are vacuous without it.
    """
    return Job(adapter, ctx, scanner=FailurePatternScanner())


# ----- Suite plumbing (single-cell barebones; sweep parametrize lands in PR-Z) -----


def _load_config(config):
    return load_config_file(config.getoption("config_file"))


def _suite_name(config) -> str:
    return getattr(config, "_suite_name", None) or "dtni"


class _SingleHostExecutor:
    """Minimal Pssh-backed executor for the bound server node (duck-typed .exec)."""

    def __init__(self, host: str, user: str, pkey: str) -> None:
        from cvs.lib.parallel.pssh import Pssh

        self._pssh = Pssh(None, [host], user=user, pkey=pkey)

    def exec(self, cmd, timeout=None):
        out = self._pssh.exec(cmd, timeout=timeout) if timeout is not None else self._pssh.exec(cmd)
        if isinstance(out, dict):
            return "\n".join(str(v) for v in out.values())
        return str(out)

    def download(self, remote, local):
        """SFTP-fetch ``remote`` from the bound node into ``local`` on devbox.

        Thin passthrough to ``Pssh.download_file`` for adapters that need
        to pull a small result artifact back without standing up a shared
        filesystem. ``Pssh.download_file`` suffixes the local path with
        the host name to avoid multi-host collisions; we copy that
        suffixed file into ``local`` so callers see the exact path they
        asked for.
        """
        import shutil
        from pathlib import Path

        local_p = Path(local)
        local_p.parent.mkdir(parents=True, exist_ok=True)
        result = self._pssh.download_file(str(remote), str(local_p))
        # download_file returns {host: actual_local_path}; collapse to local.
        actual = next(iter(result.values()))
        if actual != str(local_p):
            shutil.move(actual, str(local_p))
        return str(local_p)


def _build_executor(bind_result, pool):
    if os.environ.get("CVS_DTNI_DRY"):
        return None
    for hosts in bind_result.bindings.values():
        if hosts:
            node = pool.nodes[hosts[0]]
            try:
                return _SingleHostExecutor(node.ip, node.user, node.ssh_key)
            except Exception:
                return None
    return None


def _execute_single_cell(config) -> Manifest:
    """Bind once, build a single cell (cell_id="single"), run via Job."""
    cfg = _load_config(config)
    pool = load_cluster_file(config.getoption("cluster_file"))
    bind_result = bind(cfg.topology, pool)

    artifact_dir = os.environ.get("CVS_ARTIFACT_DIR", "/tmp/cvs/artifacts")
    test_id = _suite_name(config)
    run_id = f"{getpass.getuser()}-{uuid.uuid4().hex[:8]}"
    cell_id = "single"
    layout = RunLayout(artifact_dir, test_id, cell_id, cfg.workload_hash(), run_id)

    if bind_result.status == "skipped":
        layout.ensure()
        manifest = Manifest(
            identity=Identity(run_id=run_id, test_id=test_id, cell_id=cell_id),
            config=ConfigInputs(model=cfg.model, seed=cfg.seed),
            verdicts=Verdicts(overall_status="skipped", skip_reason=bind_result.reason),
        )
        manifest.write(layout.manifest_path)
        return manifest

    executor = _build_executor(bind_result, pool)
    events = EventWriter(layout.ensure().events_path)
    adapter = get_adapter(cfg.framework)()
    # Single-cell barebones: cell_params is empty (PR-Z lowers sweep -> per-cell params).
    ctx = RunContext(cfg, None, bind_result.bindings, layout, events, run_id, executor=executor)
    return _build_job(adapter, ctx).run()


def _run_cache(config):
    if not hasattr(config, "_dtni_runs"):
        config._dtni_runs = {}
    return config._dtni_runs


@pytest.fixture(scope="session")
def workload_run(request):
    """Run one cell through ``Job(... , scanner=FailurePatternScanner()).run()``.

    Cached on the pytest ``config`` so logistics/inference/frameworks tier
    tests in the same session share one real run. Skipped cells skip every
    tier test with the binder's reason.
    """
    config = request.config
    cache = _run_cache(config)
    if "single" not in cache:
        cache["single"] = _execute_single_cell(config)
    manifest = cache["single"]
    if manifest.verdicts.overall_status == "skipped":
        pytest.skip(f"cell skipped: {manifest.verdicts.skip_reason}")
    return manifest


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
    runs = getattr(config, "_dtni_runs", {})
    if not runs:
        return
    terminalreporter.write_sep("-", "DTNI workload runs")
    for cell_id, manifest in runs.items():
        verdicts = manifest.verdicts
        line = f"  {cell_id}: {verdicts.overall_status}"
        if verdicts.failure_category:
            line += f" ({verdicts.failure_category})"
        if verdicts.skip_reason:
            line += f" ({verdicts.skip_reason})"
        terminalreporter.write_line(line)
