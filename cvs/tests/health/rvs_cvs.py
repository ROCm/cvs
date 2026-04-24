'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent
publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

# ROCm Validation Suite (RVS) tests, multi-orch backend.
#
# Backend-blind by construction: this module never reads orch.orchestrator_type
# and never branches on backend identity. Routing is intent-based:
#   - RVS workload (rvs --version, rvs -g, rvs -c <conf>, rvs -r <level>) and
#     workload-adjacent commands (amd-smi, rocm_agent_enumerator, ls of RVS
#     config dirs, copying/sedding RVS config files into sealed scratch) all
#     go through orch.exec, which routes wherever the orchestrator runs.
#   - Host-namespace commands (lsmod, dmesg, etc.) would go through
#     host_only(orch).exec; this suite currently has none.
#
# All sudo prefixes come from orch.privileged_prefix() via the sudo() helper.
# All temporary files live under sealed_tmp(...) which is per-MULTIORCH_RUN_ID
# to keep parallel cells in the matrix isolated.

import json
import re

import pytest
from packaging import version

from cvs.lib.utils_lib import (
    fail_test,
    print_test_output,
    scan_test_results,
    update_test_result,
)
from cvs.lib import globals
from cvs.tests.health._rvs_orch_helpers import (
    ensure_sealed_tmp_dir,
    sealed_tmp,
    sudo,
)

log = globals.log


# Test names that collapse into the test_rvs_individual parametrize. Order is
# preserved for readability; pytest collection follows this order, which keeps
# cross-cell parity diffs deterministic.
RVS_INDIVIDUAL_TESTS = [
    "mem_test",
    "gst_single",
    "iet_stress",
    "pebb_single",
    "pbqt_single",
    "babel_stream",
]


# ---------------------------------------------------------------------------
# Module-scope config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config_dict(pytestconfig):
    """Load the RVS section of the test config and resolve placeholders.

    Note: cluster placeholders (e.g. {user-id}) live in cluster_file and are
    resolved by OrchestratorConfig.from_configs (which the conftest's
    cluster_orch fixture uses). For test-config placeholders we still need
    the legacy resolve_test_config_placeholders helper since OrchestratorConfig
    does not own that surface.
    """
    from cvs.lib.utils_lib import (
        resolve_cluster_config_placeholders,
        resolve_test_config_placeholders,
    )

    cluster_file = pytestconfig.getoption("cluster_file")
    config_file = pytestconfig.getoption("config_file")

    with open(cluster_file) as f:
        cluster_dict = json.load(f)
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)

    with open(config_file) as f:
        cfg_full = json.load(f)
    cfg = cfg_full["rvs"]
    cfg = resolve_test_config_placeholders(cfg, cluster_dict)
    log.info("config_dict (rvs section): %s", cfg)
    return cfg


@pytest.fixture(scope="module")
def rvs_test_level(config_dict):
    """RVS test level (0-5). 0 = run individual tests; 1-5 = run level_config."""
    raw = config_dict.get("rvs_test_level", 4)
    try:
        v = int(raw)
        if 0 <= v <= 5:
            return v
        log.warning("Invalid rvs_test_level=%r; using default 4", raw)
        return 4
    except (ValueError, TypeError):
        log.warning("Invalid rvs_test_level format=%r; using default 4", raw)
        return 4


# ---------------------------------------------------------------------------
# Helpers (module-private; thin wrappers around orch.exec for readability)
# ---------------------------------------------------------------------------


def _parse_rvs_test_results(test_config, out_dict):
    """Generic regex-based pass/fail parser. Mirrors the legacy semantics
    (failure = regex match in stdout). Per plan §11 D1 default we keep this
    regex-only; exit-code-aware checks are tracked as a follow-up."""
    test_name = test_config.get("name", "unknown")
    fail_pattern = test_config.get("fail_regex_pattern", r"\[ERROR\s*\]")
    for node, raw in out_dict.items():
        if re.search(fail_pattern, raw or "", re.I):
            fail_test(f"RVS {test_name} test failed on node {node}")
        else:
            log.info(f"RVS {test_name} test passed on node {node}")


def _parse_rvs_level_results(test_config, out_dict, level):
    """Multi-pattern parser used by the LEVEL-config test."""
    fail_patterns = test_config.get("fail_regex_patterns", [])
    if not fail_patterns:
        log.warning("No fail_regex_patterns for RVS LEVEL %s; treating as pass-only", level)
        return
    for node, raw in out_dict.items():
        hits = [p for p in fail_patterns if re.search(p, raw or "", re.I)]
        if hits:
            fail_test(f"RVS LEVEL-{level} test failed on node {node}. Failure patterns matched: {', '.join(hits)}")
        else:
            log.info(f"RVS LEVEL-{level} test passed on node {node}")


def _prepare_gst_single_temp_fix(orch, source_path):
    """GST TEMP-FIX, rewritten for idempotence.

    Always materializes a fresh copy of the source RVS gst_single config into
    sealed_tmp(...) (per-MULTIORCH_RUN_ID, so reruns get a clean copy) and
    applies the fp64 compute_type sed there. Never modifies the source file.
    Safe to call multiple times in the same iteration; safe across iterations.
    """
    if source_path is None:
        return None
    if "MI355X" not in source_path and "MI350X" not in source_path:
        return source_path  # no fix needed for this device

    ensure_sealed_tmp_dir(orch)
    temp_config = sealed_tmp("gst_single.conf")

    # Step 1: copy fresh from pristine source into sealed scratch.
    copy_cmd = f"cp {source_path} {temp_config}"
    copy_out = orch.exec(copy_cmd, timeout=30)
    for node, raw in copy_out.items():
        if (raw or "").strip() and "No such file" in raw:
            fail_test(f"GST TEMP-FIX: copy failed on {node}: {raw}")
            return None

    # Step 2: sed the sealed copy (NEVER the source). Idempotent across reruns
    # because the sealed copy is a fresh cp from source on every iteration.
    sed_cmds = [
        (
            r"sed -i '/^- name: gst-Tflops-8K-trig-fp64$/,/^- name:/{ "
            r"/^  data_type: fp64_r$/a\\\n  compute_type: fp64_r\n"
            r"}' "
            f"{temp_config}"
        ),
        (
            r"sed -i '/^- name: gst-Tflops-8K-rand-fp64$/,/^- name:/{ "
            r"/^  data_type: fp64_r$/a\\\n  compute_type: fp64_r\n"
            r"}' "
            f"{temp_config}"
        ),
    ]
    for cmd in sed_cmds:
        sed_out = orch.exec(cmd, timeout=30)
        for node, raw in sed_out.items():
            if (raw or "").strip():
                log.warning(f"GST TEMP-FIX sed on {node} produced output: {raw}")

    log.info(f"GST TEMP-FIX applied to {temp_config} (source untouched)")
    return temp_config


def _execute_rvs_test(orch, config_dict, rvs_config_paths, test_name):
    """Run a single named RVS test.

    Mirrors legacy execute_rvs_test semantics but routes through orch and
    uses sealed_tmp for any scratch writes.
    """
    test_config = next((t for t in config_dict["tests"] if t["name"] == test_name), None)
    if not test_config:
        fail_test(f"Test configuration for {test_name} not found")
        update_test_result()
        return

    log.info(f"Testcase Run RVS {test_config.get('description', test_name)}")
    rvs_path = config_dict["path"]
    timeout = test_config.get("timeout", 9000)

    config_path = rvs_config_paths.get(test_name)

    if test_name == "gst_single":
        config_path = _prepare_gst_single_temp_fix(orch, config_path)

    if config_path is None:
        fail_test(f"Configuration file for {test_name} not found on any/some node.")
        update_test_result()
        return

    # PEQT requires elevated permissions; everything else uses the privileged
    # prefix only when the config marks it. Today only PEQT does in legacy.
    if test_name == "peqt_single":
        rvs_cmd = f"{sudo(orch, '')}{rvs_path}/rvs -c {config_path}".strip()
    else:
        rvs_cmd = f"{rvs_path}/rvs -c {config_path}"

    out = orch.exec(rvs_cmd, timeout=timeout)
    print_test_output(log, out)
    scan_test_results(out)
    _parse_rvs_test_results(test_config, out)
    update_test_result()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rvs_gpu_enumeration(orch, config_dict):
    """Sanity: rvs -g must enumerate at least one supported GPU on every node.

    Always runs (no version gate); any node returning 'No supported GPUs
    available' fails the test.
    """
    log.info("Testcase Run RVS GPU Enumeration Test")
    rvs_path = config_dict["path"]
    out = orch.exec(f"{rvs_path}/rvs -g", timeout=60)
    print_test_output(log, out)
    scan_test_results(out)
    for node, raw in out.items():
        if re.search(r"No supported GPUs available", raw or "", re.I):
            fail_test(f"No GPUs detected in RVS enumeration on node {node}")
    update_test_result()


@pytest.mark.requires_rvs(min_version="1.3.0")
def test_rvs_level_config(orch, config_dict, rvs_version, rvs_test_level):
    """Run RVS with -r <level>; aggregate-of-modules style.

    Skipped when:
      - RVS < 1.3.0 (handled by @requires_rvs marker -> POLICY_SKIP)
      - rvs_test_level == 0 (user opted into individual tests instead)
    """
    if rvs_test_level == 0:
        pytest.skip("[POLICY_SKIP] rvs_test_level=0: Running individual tests instead")

    log.info(f"Testcase Run RVS LEVEL-{rvs_test_level} (RVS {rvs_version})")
    test_config = next((t for t in config_dict["tests"] if t["name"] == "level_config"), None)
    if not test_config:
        log.warning("level_config not in config_dict['tests']; using default settings")
        test_config = {
            "name": "level_config",
            "description": f"RVS LEVEL-{rvs_test_level} Comprehensive Test",
            "timeout": 7200,
            "fail_regex_patterns": [],
            "expected_pass_patterns": [],
        }

    rvs_path = config_dict["path"]
    timeout = test_config.get("timeout", 7200)

    rvs_cmd = f"{sudo(orch, '')}{rvs_path}/rvs -r {rvs_test_level}".strip()
    log.info(f"Executing: {rvs_cmd}")
    out = orch.exec(rvs_cmd, timeout=timeout)
    print_test_output(log, out)
    scan_test_results(out)
    _parse_rvs_level_results(test_config, out, rvs_test_level)
    update_test_result()


@pytest.mark.parametrize("test_name", RVS_INDIVIDUAL_TESTS)
def test_rvs_individual(orch, config_dict, rvs_config_paths, rvs_version, rvs_test_level, test_name):
    """Per-module RVS test (mem, gst, iet, pebb, pbqt, babel).

    Collapses the 6 legacy test functions into one parametrize. Skip policy
    matches the legacy should_skip_individual_test:
      - Skip if RVS >= 1.3.0 AND rvs_test_level != 0 (level_config replaces
        individuals on newer RVS).
      - Skip if the test_name is not declared in config_dict['tests'] (the
        iter config is a subset; missing entries are NOT failures).
      - Otherwise run.
    """
    if rvs_test_level != 0 and version.parse(rvs_version) >= version.parse("1.3.0"):
        pytest.skip(
            f"[POLICY_SKIP] RVS {rvs_version} >= 1.3.0 and rvs_test_level="
            f"{rvs_test_level}: Running LEVEL-{rvs_test_level} test instead"
        )

    if not any(t.get("name") == test_name for t in config_dict.get("tests", [])):
        pytest.skip(f"[POLICY_SKIP] {test_name} not declared in config['tests'] (iter config subset)")

    _execute_rvs_test(orch, config_dict, rvs_config_paths, test_name)
