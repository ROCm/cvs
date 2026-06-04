'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
'''

"""
Paired A/B RCCL regression test.

For each (collective, data type, NCCL env-combo) this runs a reference build (A)
and a candidate build (B) back-to-back, interleaved, for a configurable number of
repeats. Because both sides run on the same nodes within the same allocation,
environmental noise is largely common-mode and cancels in the paired comparison.

All pass/fail logic lives in ``cvs.lib.regression_lib`` (pure, unit-tested). This
module is only the cluster orchestration: it produces the A and B result samples
and hands them to the detector.

Modes (config: rccl.ab_regression):
  - control_mode=true : both sides use the reference build (A=B). Used to measure
    the on-hardware run-to-run noise floor, derive thresholds, and prove the
    detector reports zero regressions on an identical build.
  - control_mode=false: reference build vs candidate build (real detection).

See cvs/input/config_file/rccl/rccl_ab_config.json.sample for the config shape.
"""

import os
import json
import itertools
import copy

import pytest

from cvs.lib import rccl_lib
from cvs.lib import regression_lib
from cvs.lib import ci_robustness_lib
from cvs.lib.parallel_ssh_lib import *
from cvs.lib.utils_lib import *
from cvs.lib.verify_lib import *
from cvs.lib import globals

log = globals.log


def _gpu_cleanup_cfg(config_dict):
    """Return the gpu_cleanup config block with sane defaults."""
    cfg = dict(config_dict.get('gpu_cleanup', {}))
    cfg.setdefault('enabled', True)
    cfg.setdefault('kill_gpu_pids', True)
    cfg.setdefault('kill_containers', False)
    cfg.setdefault('use_sudo', False)
    cfg.setdefault('process_patterns', None)  # None -> library defaults
    return cfg


def _do_gpu_cleanup(phdl, config_dict, reason=""):
    """Run stale-GPU cleanup across all nodes if enabled in config."""
    cfg = _gpu_cleanup_cfg(config_dict)
    if not cfg.get('enabled', True):
        return
    log.info("Running stale-GPU cleanup%s", f" ({reason})" if reason else "")
    rccl_lib.cleanup_gpus_on_nodes(
        phdl,
        process_patterns=cfg.get('process_patterns'),
        kill_gpu_pids=cfg.get('kill_gpu_pids', True),
        kill_containers=cfg.get('kill_containers', False),
        use_sudo=cfg.get('use_sudo', False),
    )


# Accumulates per-(collective, dtype, env-combo) A and B run samples across the
# parametrized test invocations, consumed by the final analysis test.
#   ab_runs[group_key] = {"a": [run, run, ...], "b": [run, run, ...]}
ab_runs = {}


# --------------------------------------------------------------------------- #
# Fixtures (mirror cvs/tests/rccl/rccl_regression.py)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    return pytestconfig.getoption("cluster_file")


@pytest.fixture(scope="module")
def config_file(pytestconfig):
    return pytestconfig.getoption("config_file")


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file) as json_file:
        cluster_dict = json.load(json_file)
    cluster_dict = resolve_cluster_config_placeholders(cluster_dict)
    log.info("%s", cluster_dict)
    return cluster_dict


@pytest.fixture(scope="module")
def config_dict(config_file, cluster_dict):
    with open(config_file) as json_file:
        config_dict_t = json.load(json_file)
    config_dict = config_dict_t['rccl']
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


@pytest.fixture(scope="module")
def phdl(cluster_dict):
    env_vars = cluster_dict.get("env_vars")
    node_list = list(cluster_dict['node_dict'].keys())
    return Pssh(log, node_list, user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)


@pytest.fixture(scope="module")
def shdl(cluster_dict):
    node_list = list(cluster_dict['node_dict'].keys())
    env_vars = cluster_dict.get("env_vars")
    head_node = node_list[0]
    return Pssh(log, [head_node], user=cluster_dict['username'], pkey=cluster_dict['priv_key_file'], env_vars=env_vars)


# --------------------------------------------------------------------------- #
# Parametrization: collectives x data types x NCCL env-combos
# --------------------------------------------------------------------------- #
def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.exists(config_file):
        log.warning(f'Warning: Missing or invalid config file {config_file}')
        return

    with open(config_file) as fp:
        cfg = json.load(fp)
    rccl = cfg.get("rccl", {})
    regression = dict(rccl.get("regression", {}))
    if not regression:
        log.error("No regression object found in config - required for A/B parametrization")
        return

    # Paired channel handling (min/max kept paired, not Cartesian) - identical to
    # the single-sided regression test.
    has_min = "NCCL_MIN_NCHANNELS" in regression
    has_max = "NCCL_MAX_NCHANNELS" in regression
    if has_min != has_max:
        raise ValueError("NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS must be both present or both absent")
    paired_channels = None
    if has_min and has_max:
        min_vals = regression["NCCL_MIN_NCHANNELS"]
        max_vals = regression["NCCL_MAX_NCHANNELS"]
        if len(min_vals) != len(max_vals):
            raise ValueError("NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS must have equal length")
        paired_channels = list(zip(min_vals, max_vals))
        del regression["NCCL_MIN_NCHANNELS"]
        del regression["NCCL_MAX_NCHANNELS"]

    env_axes = []
    for key in sorted(regression.keys()):
        value = regression[key]
        if isinstance(value, list) and value:
            env_axes.append((key, value))

    if env_axes and "rccl_collective" in metafunc.fixturenames:
        rccl_collective_list = rccl.get("rccl_collective", ["all_reduce_perf"])
        env_fixture_names = [name for name, _ in env_axes]
        env_domains = [dict(env_axes)[name] for name in env_fixture_names]
        env_params, env_ids = [], []

        channel_fixture_names = []
        if paired_channels is not None:
            channel_fixture_names = ["NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"]
            env_domains.append(paired_channels)

        for env_combo in itertools.product(*env_domains):
            env_dict = dict(zip(env_fixture_names + channel_fixture_names, env_combo))
            if paired_channels is not None:
                min_ch, max_ch = env_dict.pop("NCCL_MIN_NCHANNELS")
                env_dict["NCCL_MIN_NCHANNELS"] = min_ch
                env_dict["NCCL_MAX_NCHANNELS"] = max_ch
            env_params.append(env_dict)
            env_ids.append("|".join(f"{k}={v}" for k, v in env_dict.items()))

        metafunc.parametrize("rccl_collective", rccl_collective_list)
        metafunc.parametrize("regression_params", env_params, ids=env_ids)

        # Optional data-type axis. Each data type runs as its own sweep (rccl-tests -d).
        if "data_type" in metafunc.fixturenames:
            data_type_list = rccl.get("data_types", ["float"])
            metafunc.parametrize("data_type", data_type_list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _side_params(base_rccl_test_params, side_cfg, data_type=None):
    """Build a per-side copy of rccl_test_params with the build's tests dir / lib path."""
    params = copy.deepcopy(base_rccl_test_params)
    if side_cfg.get("rccl_tests_dir"):
        params["rccl_tests_dir"] = side_cfg["rccl_tests_dir"]
    if side_cfg.get("ld_library_path"):
        params["ld_library_path"] = side_cfg["ld_library_path"]
    if data_type:
        params["data_type"] = data_type
    return params


def _run_one_side_once(phdl, shdl, cluster_dict, config_dict, side_cfg, collective, env_overrides, repeat_idx,
                       data_type=None):
    """
    Run a single sweep for one build side and return the parsed result rows.

    Raises on failure so the retry wrapper can act: a sweep "fails" if
    rccl_regression raises, if it recorded errors via fail_test (error_list), or
    if it produced no result rows.
    """
    node_list = list(cluster_dict['node_dict'].keys())
    vpc_node_list = [cluster_dict['node_dict'][n]['vpc_ip'] for n in node_list]

    rccl_test_params = _side_params(config_dict['rccl_test_params'], side_cfg, data_type)

    # Per-side, per-repeat result file so concurrent/sequential runs never clash.
    base_cvs = copy.deepcopy(config_dict['cvs_params'])
    base_file = base_cvs.get('rccl_result_file', '/tmp/rccl_result_output.json')
    stem, ext = os.path.splitext(base_file)
    label = side_cfg.get("label", "side")
    dt = data_type or "na"
    base_cvs['rccl_result_file'] = f'{stem}_{label}_{dt}_r{repeat_idx}{ext}'
    # A/B does its own comparison; disable the single-sided verification entirely.
    base_cvs['verify_bus_bw'] = 'False'
    base_cvs['verify_bw_dip'] = 'False'
    base_cvs['verify_lat_dip'] = 'False'

    # Clean, reproducible per-run log (MPI launch command + rccl-tests output),
    # separate from the verbose parallel-ssh logging. Defaults to a single
    # appended file under the A/B output dir; honour an explicit override.
    if not base_cvs.get('rccl_command_log'):
        ab_cfg = config_dict.get('ab_regression', {})
        out_dir = ab_cfg.get('output_dir') or os.getenv('CVS_OUTPUT_BASE_DIR') or '/tmp'
        base_cvs['rccl_command_log'] = os.path.join(out_dir, 'rccl_runs.log')

    env_overrides = {k: str(v) for k, v in env_overrides.items()}

    # Reset the global failure accumulator so we can detect this sweep's own
    # failures (rccl_regression records via fail_test rather than always raising).
    globals.error_list = []
    rows = rccl_lib.rccl_regression(
        phdl,
        shdl,
        collective,
        config_dict.get('env_source_script', '/dev/null'),
        config_dict['mpi_params'],
        rccl_test_params,
        base_cvs,
        node_list,
        vpc_node_list,
        env_overrides,
    )
    if globals.error_list:
        errs = list(globals.error_list)
        globals.error_list = []
        raise RuntimeError(f"sweep reported failures: {errs}")
    if not rows:
        raise RuntimeError("sweep returned no result rows")
    return rows


def _run_one_side(phdl, shdl, cluster_dict, config_dict, side_cfg, collective, env_overrides, repeat_idx,
                  data_type=None):
    """Run one sweep with retry on transient failures; clean stale GPU state between attempts."""
    retry_cfg = config_dict.get('retry', {})
    max_retries = int(retry_cfg.get('max_retries', 2))
    backoff_sec = float(retry_cfg.get('backoff_sec', 15))

    def attempt():
        return _run_one_side_once(
            phdl, shdl, cluster_dict, config_dict, side_cfg, collective, env_overrides, repeat_idx, data_type
        )

    def on_before_retry(next_attempt):
        # A flaky run can leave orphaned ranks/orted holding GPUs; clear them first.
        _do_gpu_cleanup(phdl, config_dict, reason=f"before retry {next_attempt} of {side_cfg.get('label')}")

    result = ci_robustness_lib.run_with_retries(
        attempt,
        max_retries=max_retries,
        on_before_retry=on_before_retry,
        backoff_sec=backoff_sec,
        log=log,
        label=f"{collective}/{side_cfg.get('label')}/d={data_type}/r{repeat_idx}",
    )
    # Ensure no stale errors leak into the test's final pass/fail check.
    globals.error_list = []
    return result


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_00_cleanup_stale_gpu_state(phdl, config_dict):
    """
    Kill stale RCCL/MPI processes (and optionally GPU PIDs / containers) left on
    the allocated nodes by prior or cancelled jobs, before any benchmark runs.
    Runs first by virtue of its position in this module. Best-effort: never fails.
    """
    globals.error_list = []
    _do_gpu_cleanup(phdl, config_dict, reason="pre-run")
    update_test_result()


def test_ab_pair(phdl, shdl, cluster_dict, config_dict, rccl_collective, regression_params, data_type):
    """Run reference (A) and candidate (B) interleaved for R repeats and stash samples."""
    globals.error_list = []

    ab_cfg = config_dict.get('ab_regression', {})
    repeats = int(ab_cfg.get('repeats', 7))
    control_mode = bool(ab_cfg.get('control_mode', False))

    reference_cfg = ab_cfg.get('reference', {"label": "ref"})
    candidate_cfg = ab_cfg.get('candidate', {"label": "cand"})
    if control_mode:
        # Both sides use the reference build to characterise noise.
        candidate_cfg = copy.deepcopy(reference_cfg)
        candidate_cfg['label'] = 'cand'
    reference_cfg = {**reference_cfg, "label": reference_cfg.get("label", "ref")}
    candidate_cfg = {**candidate_cfg, "label": candidate_cfg.get("label", "cand")}

    params_str = ' '.join(f'{k}={v}' for k, v in regression_params.items())
    group_key = f'{rccl_collective}-d={data_type}-{params_str}'
    ab_runs.setdefault(group_key, {"a": [], "b": []})

    for r in range(repeats):
        # Interleave A,B per repeat so slow drift affects both sides equally.
        a_rows = _run_one_side(
            phdl, shdl, cluster_dict, config_dict, reference_cfg, rccl_collective, regression_params, r, data_type
        )
        b_rows = _run_one_side(
            phdl, shdl, cluster_dict, config_dict, candidate_cfg, rccl_collective, regression_params, r, data_type
        )
        ab_runs[group_key]["a"].append(a_rows)
        ab_runs[group_key]["b"].append(b_rows)

    update_test_result()


def test_ab_analyze(request, config_dict):
    """Derive thresholds (control mode) and/or run the A/B detector; fail on confirmed regressions."""
    globals.error_list = []
    ab_cfg = config_dict.get('ab_regression', {})
    control_mode = bool(ab_cfg.get('control_mode', False))

    # Effective detector config from the user's ab_regression block.
    detector_overrides = {}
    for k in ("thresholds", "tier_boundaries", "separation_gate", "separation_b_percentile",
              "separation_a_percentile", "adjacency_min_run", "min_bandwidth_floor", "min_repeats",
              "metric", "higher_is_better"):
        if k in ab_cfg:
            detector_overrides[k] = ab_cfg[k]

    out_dir = ab_cfg.get('output_dir') or os.getenv('CVS_OUTPUT_BASE_DIR') or '/tmp'
    os.makedirs(out_dir, exist_ok=True)

    overall_has_regression = False
    all_reports = {}

    # Control mode: derive thresholds from the combined A+B (same build) data.
    if control_mode:
        control_runs = []
        for g in ab_runs.values():
            control_runs.extend(g["a"])
            control_runs.extend(g["b"])
        if control_runs:
            derived = regression_lib.derive_thresholds(
                control_runs,
                config=detector_overrides or None,
                safety_factor=float(ab_cfg.get('safety_factor', 2.0)),
            )
            log.info("Derived thresholds from control run: %s", derived["thresholds"])
            log.info("Measured noise: %s", derived["noise"])
            with open(os.path.join(out_dir, 'ab_derived_thresholds.json'), 'w') as fp:
                json.dump(derived, fp, indent=2)
            # Apply derived thresholds for the (sanity) detection below.
            detector_overrides = {**detector_overrides, "thresholds": derived["thresholds"]}

    for group_key, runs in ab_runs.items():
        report = regression_lib.detect_regressions(runs["a"], runs["b"], config=detector_overrides or None)
        all_reports[group_key] = report
        log.info("[%s]\n%s", group_key, regression_lib.format_report(report))
        if report["summary"]["has_regression"]:
            overall_has_regression = True

    with open(os.path.join(out_dir, 'ab_regression_report.json'), 'w') as fp:
        json.dump({"control_mode": control_mode, "reports": all_reports}, fp, indent=2, default=str)

    if control_mode:
        # In control mode a confirmed regression means the detector is NOT stable
        # on this hardware (identical build flagged) - that must fail loudly.
        if overall_has_regression:
            fail_test("Control run (A=B) reported a regression - detector/thresholds are not stable on this hardware")
    else:
        if overall_has_regression:
            regressions = sum(r["summary"]["regressions"] for r in all_reports.values())
            fail_test(f"A/B regression detected: {regressions} confirmed regression(s) - see ab_regression_report.json")

    update_test_result()
