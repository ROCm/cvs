'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Fixtures and hooks for ``sglang_disagg_distributed`` (multi-node PSSH + Docker).

``--config_file`` must be JSON with top-level ``"config"`` and ``"benchmark_params"``.
Use ``active_benchmark`` (or ``SGLANG_BENCHMARK_KEY``) when multiple models are defined.

Each ``benchmark_params`` variant may set ``threshold_file`` to a
JSON file beside the config; that file supplies pass/fail thresholds for performance
and lm-eval benchmarks.
'''

from __future__ import annotations

import json
import os
from dataclasses import dataclass
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


# ---------- threshold helpers (unchanged) ----------


def _threshold_file_path(bp_dict: Mapping[str, Any]) -> str | None:
    path = bp_dict.get("threshold_file")
    if path:
        return str(path).strip()
    return None


def _resolve_threshold_path(threshold_path: str) -> Path:
    path = Path(threshold_path)
    if path.is_absolute():
        return path
    return path.resolve()


def _load_thresholds_file(path: Path) -> dict[str, Any]:
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
    bench = (bp_dict.get("inference_tests") or {}).get("bench_serv_random") or {}
    return (
        f"ISL={bench.get('input_length', '-')},"
        f"OSL={bench.get('output_length', '-')},"
        f"TP={bp_dict.get('tensor_parallelism', '-')},"
        f"CONC={bp_dict.get('max_concurrency', '-')}"
    )


def bench_cell_key(bench_name: str) -> str:
    return f"BENCH={bench_name}"


def flat_expected_from_specs(specs: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric, spec in specs.items():
        if isinstance(spec, dict) and "value" in spec:
            out[metric] = float(spec["value"])
        else:
            out[metric] = float(spec)
    return out


def _inject_thresholds_into_bp_dict(bp_dict: dict[str, Any], thresholds: Mapping[str, Any]) -> None:
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


# ---------- variant config (SGLang analogue of vLLM's load_variant) ----------


@dataclass(frozen=True)
class SglangVariantConfig:
    """Typed bundle for one SGLang disagg benchmark variant.

    Mirrors the role of vLLM's ``VariantConfig``: single entry point for tests
    and fixtures. SGLang keeps its legacy JSON shape (``config`` + ``benchmark_params``)
    rather than ``BaseVariantConfig`` schema.
    """

    config_path: str
    variant_key: str
    inference: dict[str, Any]
    benchmark_params: dict[str, Any]
    thresholds: dict[str, Any]
    enforce_thresholds: bool = True

    @property
    def hf_token_file(self) -> str:
        return self.inference["hf_token_file"]

    @property
    def model(self) -> str:
        return self.benchmark_params["model"]

    def perf_cell_key(self) -> str:
        return perf_cell_key(self.benchmark_params)


def load_sglang_variant(config_path: str, cluster_dict: Mapping[str, Any]) -> SglangVariantConfig:
    """Load, resolve placeholders, pick variant, load thresholds, inject expected_results."""
    with open(config_path, encoding="utf-8") as fp:
        root = json.load(fp)

    variant_key = resolve_benchmark_variant_key(root, config_path)

    if isinstance(root, dict) and "config" in root:
        cfg = root["config"]
    else:
        cfg = root

    inference = resolve_test_config_placeholders(cfg, cluster_dict)
    bp_all = resolve_test_config_placeholders(root["benchmark_params"], cluster_dict)
    bp = dict(bp_all[variant_key])

    threshold_path_str = _threshold_file_path(bp)
    if not threshold_path_str:
        pytest.fail(
            f"benchmark_params[{variant_key!r}] missing "
            "'threshold_file' in --config_file"
        )

    threshold_path = _resolve_threshold_path(threshold_path_str)
    thresholds = _load_thresholds_file(threshold_path)
    log.info("Loaded thresholds from %s (%d cells)", threshold_path, len(thresholds))
    _inject_thresholds_into_bp_dict(bp, thresholds)

    return SglangVariantConfig(
        config_path=config_path,
        variant_key=variant_key,
        inference=inference,
        benchmark_params=bp,
        thresholds=thresholds,
        enforce_thresholds=bool(root.get("enforce_thresholds", True)),
    )


# ---------- lifecycle (same model as vLLM conftest) ----------


class _Lifecycle:
    """Cross-test state for the lifecycle-as-tests model."""

    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report: dict[str, list[tuple[str, float, str]]] = {}

    def record(self, nodeid: str, label: str, value: float, unit: str = "s") -> None:
        self.report.setdefault(nodeid, []).append((label, value, unit))


# ---------- fixtures ----------


@pytest.fixture(scope="module")
def cluster_dict(pytestconfig):
    cluster_file = pytestconfig.getoption("cluster_file")
    if not cluster_file:
        pytest.fail("--cluster_file is required")
    with open(cluster_file, encoding="utf-8") as fp:
        d = json.load(fp)
    return resolve_cluster_config_placeholders(d)


@pytest.fixture(scope="module")
def variant_config(pytestconfig, cluster_dict) -> SglangVariantConfig:
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    return load_sglang_variant(config_file, cluster_dict)


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


# Backward-compatible aliases (existing tests/fixtures can keep using these names)


@pytest.fixture(scope="module")
def inference_config_file(variant_config):
    return variant_config.config_path


@pytest.fixture(scope="module")
def cluster_file(pytestconfig):
    path = pytestconfig.getoption("cluster_file")
    if not path:
        pytest.fail("--cluster_file is required")
    return path


@pytest.fixture(scope="module")
def inference_config_root(inference_config_file):
    with open(inference_config_file, encoding="utf-8") as fp:
        return json.load(fp)


@pytest.fixture(scope="module")
def inference_dict(variant_config):
    return variant_config.inference


@pytest.fixture(scope="module")
def benchmark_params_dict(inference_config_root, cluster_dict):
    bp = inference_config_root["benchmark_params"]
    return resolve_test_config_placeholders(bp, cluster_dict)


@pytest.fixture(scope="module")
def benchmark_variant(variant_config):
    return variant_config.variant_key


@pytest.fixture(scope="module")
def benchmark_params(variant_config):
    return variant_config.benchmark_params


@pytest.fixture(scope="module")
def thresholds_dict(variant_config):
    return variant_config.thresholds


@pytest.fixture(scope="module")
def hf_token(variant_config):
    path = variant_config.hf_token_file
    if not os.path.isfile(path):
        pytest.skip(f"hf_token file missing: {path}")
    with open(path, encoding="utf-8") as fp:
        return fp.read().strip()


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
    variant_config,
    hf_token,
):
    return sglang_disagg_lib.SglangDisaggPD(
        variant_config.model,
        variant_config.inference,
        variant_config.benchmark_params,
        hf_token,
        p_phdl,
        d_phdl,
        r_phdl,
        b_phdl,
        gpu_type,
    )


# ---------- pytest hooks ----------


def pytest_collection_modifyitems(items):
    rank = SGLANG_DISAGG_TEST_ORDER
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    lc = item.funcargs.get("lifecycle")
    rows = getattr(lc, "report", {}).get(item.nodeid) if lc else None
    if not rows:
        return
    try:
        import pytest_html
    except ImportError:
        return
    body = "".join(
        f"<tr><td>{label}</td><td>{value:.1f}</td><td>{unit}</td></tr>"
        for label, value, unit in rows
    )
    html = f"<table><tr><th>stage</th><th>value</th><th>unit</th></tr>{body}</table>"
    extras = getattr(report, "extras", [])
    extras.append(pytest_html.extras.html(html))
    report.extras = extras


def pytest_html_results_table_header(cells):
    cells.insert(-1, "<th>Value</th>")
    cells.insert(-1, "<th>Unit</th>")


def pytest_html_results_table_row(report, cells):
    props = dict(report.user_properties)
    has = "metric_value" in props
    val = props.get("metric_value")
    unit = props.get("metric_unit", "") if has else ""
    if not has:
        shown = ""
    elif val is None:
        shown = "-"
    elif isinstance(val, float):
        shown = f"{val:.3f}"
    else:
        shown = str(val)
    cells.insert(-1, f"<td>{shown}</td>")
    cells.insert(-1, f"<td>{unit}</td>")