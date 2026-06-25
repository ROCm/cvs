'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Fixtures and hooks for ``sglang_disagg_distributed`` (multi-node PSSH + Docker).

``--config_file`` must be JSON with top-level ``"config"`` and ``"benchmark_params"``.
Use ``active_benchmark`` (or ``SGLANG_BENCHMARK_KEY``) when multiple models are defined.

Each ``benchmark_params`` variant may set ``theshold_file`` (or ``threshold_file``) to a
JSON file beside the config; that file supplies pass/fail thresholds for performance
and lm-eval benchmarks.
'''

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from cvs.lib import globals
from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.inference import sglang_disagg_lib
from cvs.lib.utils_lib import (
    get_model_from_rocm_smi_output,
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
)

from cvs.tests.inference.sglang._shared import SGLANG_DISAGG_TEST_ORDER, resolve_benchmark_variant_key

log = globals.log


def _threshold_file_path(bp_dict: Mapping[str, Any]) -> str | None:
    """Return the threshold file path from a benchmark_params variant (typo-tolerant)."""
    for key in ("threshold_file", "threshold_file"):
        path = bp_dict.get(key)
        if path:
            return str(path).strip()
    return None

def _resolve_threshold_path(threshold_path: str) -> Path:
    """Resolve an absolute or repo-relative threshold path from config."""
    path = Path(threshold_path)
    if path.is_absolute():
        return path
    return path.resolve()


def _load_thresholds_file(path: Path) -> dict[str, Any]:
    """Load threshold JSON and drop comment keys (e.g. ``_comment``)."""
    try:
        with open(path, encoding="utf-8") as fp:
            raw = json.load(fp)
    except FileNotFoundError:
        pytest.fail(f"threshold file not found: {path}")
    except OSError as e:
        pytest.fail(f"cannot read threshold file {path}: {e}")
    except json.JSONDecodeError as e:
        pytest.fail(f"invalid JSON in threshold file {path}: {e}")

    if not isinstance(raw, dict):
        pytest.fail(f"threshold file must be a JSON object: {path}")

    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def perf_cell_key(bp_dict: Mapping[str, Any]) -> str:
    """Build the performance threshold cell key, e.g. ``ISL=1024,OSL=1024,TP=8,CONC=25``."""
    bench = (bp_dict.get("inference_tests") or {}).get("bench_serv_random") or {}
    return (
        f"ISL={bench.get('input_length', '-')},"
        f"OSL={bench.get('output_length', '-')},"
        f"TP={bp_dict.get('tensor_parallelism', '-')},"
        f"CONC={bp_dict.get('max_concurrency', '-')}"
    )


def bench_cell_key(bench_name: str) -> str:
    """Build an lm-eval threshold cell key, e.g. ``BENCH=lm_eval_hellaswag``."""
    return f"BENCH={bench_name}"


def flat_expected_from_specs(specs: Mapping[str, Any]) -> dict[str, float]:
    """Convert ``{metric: {kind, value}}`` threshold specs to flat floats for legacy checks."""
    out: dict[str, float] = {}
    for metric, spec in specs.items():
        if isinstance(spec, dict) and "value" in spec:
            out[metric] = float(spec["value"])
        else:
            out[metric] = float(spec)
    return out


def _inject_thresholds_into_bp_dict(bp_dict: dict[str, Any], thresholds: Mapping[str, Any]) -> None:
    """Merge external thresholds into ``inference_tests.*.expected_results`` in-place."""
    inference_tests = bp_dict.setdefault("inference_tests", {})

    perf_key = perf_cell_key(bp_dict)
    perf_specs = thresholds.get(perf_key)
    if perf_specs:
        bench = inference_tests.setdefault("bench_serv_random", {})
        expected = bench.setdefault("expected_results", {})
        expected["auto"] = flat_expected_from_specs(perf_specs)
        log.info("Loaded performance thresholds from cell %r", perf_key)
    else:
        log.warning("No performance thresholds for cell %r in threshold file", perf_key)

    for bench_name in ("lm_eval_hellaswag", "lm_eval_gsm8k", "lm_eval_mmlu"):
        cell = bench_cell_key(bench_name)
        acc_specs = thresholds.get(cell)
        if not acc_specs:
            continue
        bench = inference_tests.setdefault(bench_name, {})
        expected = bench.setdefault("expected_results", {})
        task_key = bench_name.removeprefix("lm_eval_")
        expected[task_key] = flat_expected_from_specs(acc_specs)
        log.info("Loaded accuracy thresholds from cell %r", cell)


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    path = pytestconfig.getoption("cluster_file")
    if not path:
        pytest.fail("--cluster_file is required")
    return path


@pytest.fixture(scope="module")
def inference_config_file(pytestconfig):
    path = pytestconfig.getoption("config_file")
    if not path:
        pytest.fail("--config_file is required")
    return path


@pytest.fixture(scope="module")
def cluster_dict(cluster_file):
    with open(cluster_file, encoding="utf-8") as fp:
        d = json.load(fp)
    return resolve_cluster_config_placeholders(d)


@pytest.fixture(scope="module")
def inference_config_root(inference_config_file):
    with open(inference_config_file, encoding="utf-8") as fp:
        return json.load(fp)


@pytest.fixture(scope="module")
def inference_dict(inference_config_root, cluster_dict):
    if isinstance(inference_config_root, dict) and "config" in inference_config_root:
        cfg = inference_config_root["config"]
    else:
        cfg = inference_config_root
    cfg = resolve_test_config_placeholders(cfg, cluster_dict)
    volume_dict = cfg['container_config']['volume_dict']
    cfg['mount_vol'] = volume_dict.pop(
        'mount_vol',
        '/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so',
    )
    return cfg


@pytest.fixture(scope="module")
def benchmark_params_dict(inference_config_root, cluster_dict):
    bp = inference_config_root["benchmark_params"]
    return resolve_test_config_placeholders(bp, cluster_dict)


@pytest.fixture(scope="module")
def benchmark_variant(inference_config_root, inference_config_file):
    return resolve_benchmark_variant_key(inference_config_root, inference_config_file)


@pytest.fixture(scope="module")
def benchmark_params(benchmark_params_dict, benchmark_variant):
    return benchmark_params_dict[benchmark_variant]


@pytest.fixture(scope="module")
def thresholds_dict(benchmark_params, benchmark_variant):
    """Load thresholds from the path in ``threshold_file`` / ``theshold_file``."""
    threshold_path_str = _threshold_file_path(benchmark_params)
    if not threshold_path_str:
        pytest.fail(
            f"benchmark_params[{benchmark_variant!r}] missing "
            "'threshold_file' or 'theshold_file' in --config_file"
        )

    threshold_path = _resolve_threshold_path(threshold_path_str)
    thresholds = _load_thresholds_file(threshold_path)
    log.info("Loaded thresholds from %s (%d cells)", threshold_path, len(thresholds))
    return thresholds


@pytest.fixture(scope="module")
def hf_token(inference_dict):
    hf_token_file = inference_dict["hf_token_file"]
    try:
        with open(hf_token_file, encoding="utf-8") as fp:
            return fp.read().rstrip("\n")
    except FileNotFoundError:
        pytest.fail(f"hf_token file not found: {hf_token_file}")
    except OSError as e:
        pytest.fail(f"cannot read hf_token file {hf_token_file}: {e}")


@pytest.fixture(scope="module")
def p_phdl(cluster_dict, inference_dict):
    env_vars = cluster_dict.get("env_vars")
    return Pssh(
        log,
        inference_dict["prefill_node_list"],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=env_vars,
    )


@pytest.fixture(scope="module")
def d_phdl(cluster_dict, inference_dict):
    env_vars = cluster_dict.get("env_vars")
    return Pssh(
        log,
        inference_dict["decode_node_list"],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=env_vars,
    )


@pytest.fixture(scope="module")
def r_phdl(cluster_dict, inference_dict):
    env_vars = cluster_dict.get("env_vars")
    return Pssh(
        log,
        [inference_dict["proxy_router_node"]],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=env_vars,
    )


@pytest.fixture(scope="module")
def b_phdl(cluster_dict, inference_dict):
    env_vars = cluster_dict.get("env_vars")
    return Pssh(
        log,
        [inference_dict["benchmark_serv_node"]],
        user=cluster_dict["username"],
        pkey=cluster_dict["priv_key_file"],
        env_vars=env_vars,
    )


@pytest.fixture(scope="module")
def gpu_type(p_phdl, cluster_dict):
    head_node = p_phdl.host_list[0]
    smi_out_dict = p_phdl.exec("rocm-smi -a | head -30")
    smi_out = smi_out_dict[head_node]
    return get_model_from_rocm_smi_output(smi_out)


@pytest.fixture(scope="module")
def inf_res_dict():
    return {}


@pytest.fixture(scope="module")
def im_obj(
    p_phdl,
    d_phdl,
    r_phdl,
    b_phdl,
    gpu_type,
    inference_dict,
    benchmark_params,
    thresholds_dict,
    hf_token,
):
    bp_dict = dict(benchmark_params)
    _inject_thresholds_into_bp_dict(bp_dict, thresholds_dict)

    return sglang_disagg_lib.SglangDisaggPD(
        bp_dict["model"],
        inference_dict,
        bp_dict,
        hf_token,
        p_phdl,
        d_phdl,
        r_phdl,
        b_phdl,
        gpu_type,
    )


def pytest_collection_modifyitems(items):
    rank = SGLANG_DISAGG_TEST_ORDER
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))