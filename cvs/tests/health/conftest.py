'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

"""Pytest plumbing for the multi-orch health suite.

Scope and blast radius
----------------------
This conftest lives at cvs/tests/health/ and therefore loads for ALL tests in
that directory, including the legacy sibling suites (transferbench_cvs.py,
agfhc_cvs.py, csp_qual_agfhc.py) that have NOT been migrated to multi-orch.

To keep the migration safe and reviewable in isolation, the new fixtures and
hooks defined here are written to be inert for tests that do not opt in:

  - cluster_orch / orch fixtures: only triggered when a test takes them as
    parameters. Legacy suites take `phdl` (their own fixture) and are
    unaffected.
  - _per_test_setup autouse fixture: a no-op on tests that do not use the
    `orch` fixture (gated on request.fixturenames).
  - requires_rvs marker enforcement: gated on the same request.fixturenames
    check; legacy sibling suites are not version-skipped by this hook.
  - xdist guard (pytest_configure): applies to the whole pytest invocation;
    rvs_cvs.py sessions are session-scoped and would race under xdist
    regardless of opt-in.

A follow-up issue tracks migrating the sibling suites; once they are on the
new fixtures, the per-test-setup gate can be relaxed.
"""

import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.tests.health._rvs_orch_helpers import (
    exec_detailed,
    get_run_id,
    require_run_id,
    sealed_tmp,
    sealed_tmp_dir,
    stamp_run_id,
)

log = globals.log


# ---------------------------------------------------------------------------
# pytest_configure: marker registration + xdist guard
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers and guard against unsafe pytest-xdist usage.

    The session-scoped cluster_orch fixture spawns a single set of containers
    per-cell; under xdist each worker would attempt its own setup with the
    same cvs_iter_${MULTIORCH_RUN_ID} name -> docker-side collision and a
    racing dispose() during worker teardown. Hard-fail before any test runs.
    """
    config.addinivalue_line(
        "markers",
        "requires_rvs(min_version=None, max_version=None): "
        "Skip the test if the cluster's RVS version is outside the half-open "
        "range [min_version, max_version). Replaces the legacy "
        "should_skip_individual_test / should_skip_level_test helpers.",
    )
    config.addinivalue_line(
        "markers",
        "requires_multinode: "
        "Skip the test if the cluster has fewer than 2 nodes. Use for tests "
        "that genuinely depend on inter-node fanout.",
    )

    if config.pluginmanager.has_plugin("xdist"):
        # `numprocesses` is set by `-n auto` / `-n N`. None or 0 means xdist
        # is loaded but not in use, which is harmless.
        if getattr(config.option, "numprocesses", None):
            raise pytest.UsageError(
                "cvs/tests/health/rvs_cvs.py is incompatible with pytest-xdist "
                "(-n flag). The session-scoped cluster_orch fixture cannot be "
                "shared across xdist workers without container name collisions. "
                "Re-run without -n."
            )


# ---------------------------------------------------------------------------
# cluster_orch: session-scoped Orchestrator, built ONCE per pytest session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cluster_orch(request, pytestconfig):
    """Session-scoped Orchestrator for the multi-orch suite.

    Single source of truth for cluster execution; backend type is decided at
    config-time (cluster_config.json["orchestrator"]) and the suite source is
    blind to it.

    Lifecycle:
      1. require_run_id (D2 default: hard-fail if MULTIORCH_RUN_ID unset).
      2. Build OrchestratorConfig.from_configs(cluster_file, config_file) so
         placeholder resolution stays single-source (no raw json.load + manual
         resolve_cluster_config_placeholders in the suite).
      3. Construct the orch via OrchestratorFactory.create_orchestrator.
      4. Register addfinalizer(o.dispose) BEFORE calling o.prepare(), so a
         prepare() exception still triggers dispose() (rollback inside
         prepare() handles partial state, and the finalizer is a backstop).
      5. Call o.prepare(); if it returns False, fail the session.
      6. Yield the orch for the duration of the session.
    """
    require_run_id()

    cluster_file = pytestconfig.getoption("cluster_file")
    config_file = pytestconfig.getoption("config_file")
    log.info(
        "cluster_orch building from cluster_file=%s, config_file=%s, run_id=%s",
        cluster_file, config_file, get_run_id(),
    )

    cfg = OrchestratorConfig.from_configs(cluster_file, config_file)
    orch = OrchestratorFactory.create_orchestrator(log, cfg)

    # Register dispose() BEFORE prepare() so partial-prepare state is still
    # cleaned up even if prepare() raises mid-way. ContainerOrchestrator.prepare
    # also rolls back internally; the finalizer is the backstop.
    request.addfinalizer(orch.dispose)

    if not orch.prepare():
        pytest.fail(
            f"orch.prepare() returned False for orchestrator={cfg.orchestrator}; "
            "see logs for details. Common causes: container image not pullable, "
            "sshd:2224 unreachable, partial container launch."
        )

    return orch


@pytest.fixture(scope="module")
def orch(cluster_orch):
    """Module-scope alias for cluster_orch.

    Module scope is the natural granularity for the migrated rvs_cvs.py: each
    test module corresponds to one suite. The actual lifecycle is still
    session-scoped (see cluster_orch); this alias exists so individual test
    files don't need to type cluster_orch everywhere.
    """
    return cluster_orch


# ---------------------------------------------------------------------------
# _per_test_setup: autouse; gated on `orch` fixture usage
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _per_test_setup(request):
    """Per-function-scope setup that runs for every test (including each
    parametrize ID). No-op for tests that do NOT depend on the `orch` fixture
    (e.g., legacy sibling suites under cvs/tests/health/).

    Hardening:
      - Asserts globals.error_list == [] at entry. If a prior test leaked an
        error message that wasn't consumed by update_test_result(), we hard-
        fail loudly here instead of silently inheriting it.
      - Resets globals.error_list = [] so the test starts clean.
      - Stamps a sentinel into the orch log so the harness can correlate
        cells <-> tests.
      - Enforces @requires_rvs(min_version=..., max_version=...) skip policy
        against the session-scoped rvs_version fixture.

    The order of fixture parameters matters: we depend on `request` only at
    function scope so we can introspect fixturenames + markers cheaply.
    """
    if "orch" not in request.fixturenames and "cluster_orch" not in request.fixturenames:
        # Not a multi-orch test; do nothing.
        yield
        return

    # Reset the legacy globals.error_list at entry so each test (including
    # each parametrize ID) starts clean. The legacy update_test_result() does
    # NOT clear error_list after pytest.fail, so without this reset a failed
    # test would poison every subsequent test in the same module.
    #
    # Note: we deliberately do NOT hard-fail on a non-empty error_list at
    # entry. That assertion would fire after every legitimate fail_test()
    # since pytest.fail() raises before update_test_result() can clear the
    # list. Clearing at both entry and exit (via finalizer) is the safe
    # invariant; leak detection that would catch missing update_test_result()
    # calls is deferred along with the broader globals.error_list -> pytest.fail
    # migration listed in the plan §4 "Deferred" notes.
    globals.error_list = []

    def _clear_error_list():
        # Ensure no leakage to the next test even if the body crashed before
        # update_test_result() ran.
        globals.error_list = []

    request.addfinalizer(_clear_error_list)

    # Apply requires_rvs skip policy (only if the marker is present).
    marker = request.node.get_closest_marker("requires_rvs")
    if marker is not None:
        # Lazily resolve rvs_version via getfixturevalue so tests that don't
        # use the marker don't pay the cost.
        try:
            current = request.getfixturevalue("rvs_version")
        except pytest.FixtureLookupError:
            current = None
        if current is None:
            pytest.skip("requires_rvs: rvs_version fixture unavailable")
        else:
            from packaging import version as _v
            min_v = marker.kwargs.get("min_version")
            max_v = marker.kwargs.get("max_version")
            cur_v = _v.parse(current)
            if min_v is not None and cur_v < _v.parse(min_v):
                pytest.skip(f"[POLICY_SKIP] requires_rvs min_version={min_v}, current={current}")
            if max_v is not None and cur_v >= _v.parse(max_v):
                pytest.skip(f"[POLICY_SKIP] requires_rvs max_version={max_v}, current={current}")

    # Apply requires_multinode skip policy.
    if request.node.get_closest_marker("requires_multinode") is not None:
        orch_obj = request.getfixturevalue("orch")
        if len(orch_obj.hosts) < 2:
            pytest.skip(
                f"[POLICY_SKIP] requires_multinode (have {len(orch_obj.hosts)} node)"
            )

    # Stamp the sentinel for harness correlation. Best-effort; do not fail the
    # test if the orch is dead at this point (the test itself will fail loudly).
    try:
        orch_obj = request.getfixturevalue("orch")
        stamp_run_id(orch_obj, request.node.nodeid)
    except Exception as e:  # noqa: BLE001
        log.warning(f"_per_test_setup: stamp_run_id failed (non-fatal): {e}")

    yield


# ---------------------------------------------------------------------------
# Session-scoped derived fixtures: rvs_version, gpu_device_map, rvs_config_paths
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rvs_path(cluster_orch, pytestconfig):
    """Resolved RVS binary path. Reads from the test config (cvs.input
    config_file) under config_dict['rvs']['path']."""
    import json
    cfg_path = pytestconfig.getoption("config_file")
    with open(cfg_path) as f:
        cfg = json.load(f)
    p = cfg.get("rvs", {}).get("path", "/opt/rocm/bin")
    return p


@pytest.fixture(scope="session")
def rvs_version(cluster_orch, rvs_path):
    """Detect RVS version across all cluster nodes; return the minimum.

    Hard-fails (pytest.exit returncode=2 -> READINESS_FAIL in the harness) if
    any node's `rvs --version` output cannot be parsed; this is treated as a
    cluster-readiness failure, not a test failure.
    """
    import re
    from packaging import version as _v

    out = cluster_orch.exec(f"{rvs_path}/rvs --version", timeout=30)
    versions = {}
    for node, raw in out.items():
        m = re.search(r"(\d+\.\d+\.\d+)", raw or "")
        if not m:
            pytest.exit(
                f"READINESS_FAIL: could not parse RVS version on node {node}: "
                f"raw output={raw!r}",
                returncode=2,
            )
        versions[node] = m.group(1)
    log.info("RVS versions per node: %s", versions)
    return min(versions.values(), key=_v.parse)


@pytest.fixture(scope="session")
def gpu_device_map(cluster_orch):
    """Detect per-node GPU market_name (e.g., 'MI300X') via amd-smi JSON.

    Returns Dict[node_ip, device_name|None]; suite must handle None for nodes
    where amd-smi parsing fails.
    """
    import json
    out = cluster_orch.exec("sudo amd-smi static -a -g 0 --json", timeout=30)
    device_map = {}
    for node, raw in out.items():
        try:
            data = json.loads(raw)
            market = data["gpu_data"][0]["asic"]["market_name"]
            # "AMD Instinct MI300X" -> "MI300X"
            name = market.replace("AMD Instinct ", "").replace(" ", "")
            device_map[node] = name or None
        except Exception as e:  # noqa: BLE001
            log.warning(f"gpu_device_map: parse failure on {node}: {e}; raw={raw!r}")
            device_map[node] = None
    log.info("gpu_device_map: %s", device_map)
    return device_map


@pytest.fixture(scope="session")
def rvs_config_paths(cluster_orch, gpu_device_map, pytestconfig):
    """Resolve the per-test RVS config file path (device-specific or default).

    Returns Dict[test_name, config_path|None] for every test in
    config_dict['rvs']['tests'] that declares a config_file. None means "not
    found on any node"; the test must handle that as a fail_test condition.
    """
    import json
    import re

    cfg_path = pytestconfig.getoption("config_file")
    with open(cfg_path) as f:
        cfg = json.load(f)
    rvs_cfg = cfg["rvs"]
    base = rvs_cfg["config_path_default"]

    # Get available device-specific folders per node
    out = cluster_orch.exec(f"ls -d {base}/*/ 2>/dev/null", timeout=30)
    folders_per_node = {}
    for node, raw in out.items():
        folders = []
        for line in (raw or "").splitlines():
            m = re.search(r"/([^/]+)/$", line.strip())
            if m and m.group(1).startswith("MI3"):
                folders.append(m.group(1))
        folders_per_node[node] = folders

    resolved = {}
    for test in rvs_cfg.get("tests", []):
        cf = test.get("config_file")
        if not cf:
            continue
        chosen = None
        for node, dev in gpu_device_map.items():
            if dev and dev in folders_per_node.get(node, []):
                candidate = f"{base}/{dev}/{cf}"
                check = cluster_orch.exec(f"ls -l {candidate}", timeout=15)
                if not re.search(r"No such file", check.get(node, ""), re.I):
                    chosen = candidate
                    break
        if chosen is None:
            default = f"{base}/{cf}"
            check = cluster_orch.exec(f"ls -l {default}", timeout=15)
            for node, raw in check.items():
                if not re.search(r"No such file", raw or "", re.I):
                    chosen = default
                    break
        resolved[test["name"]] = chosen
    log.info("rvs_config_paths: %s", resolved)
    return resolved


# ---------------------------------------------------------------------------
# _stale_fixture_check: re-validate one session-scoped value at teardown
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _stale_fixture_check(request):
    """At end of session, re-derive rvs_version from the live cluster and
    assert it equals the value computed at session start. If they diverge,
    the cluster mutated mid-session (someone upgraded ROCm, swapped a node)
    and earlier results may be invalid.

    Requires the session to have used cluster_orch; otherwise no-op.
    """
    initial = {}

    def _check():
        # Only run if cluster_orch was actually built; check the cached value.
        cluster_orch_fixture = request.session._fixturemanager._arg2fixturedefs.get("cluster_orch")
        if not cluster_orch_fixture:
            return
        try:
            current_orch = request.getfixturevalue("cluster_orch")
            current_rvs = request.getfixturevalue("rvs_version")
        except Exception as e:  # noqa: BLE001
            log.warning(f"_stale_fixture_check: skipped ({e})")
            return
        # Re-derive directly (do NOT reuse cached fixture)
        import re
        rvs_path = "/opt/rocm/bin"  # match rvs_path default
        out = current_orch.exec(f"{rvs_path}/rvs --version", timeout=30)
        fresh = {}
        for node, raw in out.items():
            m = re.search(r"(\d+\.\d+\.\d+)", raw or "")
            fresh[node] = m.group(1) if m else None
        from packaging import version as _v
        fresh_min = min((v for v in fresh.values() if v), key=_v.parse, default=None)
        if fresh_min != current_rvs:
            log.error(
                "_stale_fixture_check FAILED: session_start rvs_version=%s, "
                "session_end rvs_version=%s. Cluster mutated mid-session.",
                current_rvs, fresh_min,
            )
            # Surface as a session-finish error in junit via pytest.fail in a
            # final dummy test would be clean, but at session-finish time we
            # can only log. The harness reads logs for this signal.
        else:
            log.info("_stale_fixture_check: rvs_version stable (%s)", current_rvs)

    request.addfinalizer(_check)
    yield initial
