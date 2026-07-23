'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Fixtures and hooks for SGLang inference suites (``sglang_single`` and ``sglang_disagg_distributed``).

Both suites use ``ContainerOrchestrator`` (vLLM-style) via ``cluster_container.json`` and
``load_variant()`` from ``sglang_config_loader``.

``sglang_single.py`` — ``SglangSingle`` (unified server on ``benchmark_serv_node`` only).
``sglang_disagg_distributed.py`` — ``SglangDisaggPD`` (PD roles from inference config;
containers only on the union of prefill/decode/router/bench hosts).

Each ``benchmark_params`` variant may set ``threshold_file`` to a JSON file beside the config;
that file supplies pass/fail thresholds for performance and lm-eval benchmarks.
'''

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Mapping

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.inference.sglang.sglang_common import as_node_list
from cvs.lib.inference.sglang.sglang_config_loader import (
    SglangSingleVariantConfig,
    flat_expected_from_specs,
    load_variant,
    orchestrator_container_from_variant,
    perf_cells_from_thresholds,
)
from cvs.lib.inference.sglang.sglang_disagg_lib import SglangDisaggPD
from cvs.lib.inference.sglang.sglang_single_lib import SglangSingle
from cvs.lib.parallel_ssh_lib import Pssh
from cvs.lib.utils_lib import (
    get_model_from_rocm_smi_output,
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
    update_test_result,
)
from cvs.tests.inference.sglang._shared import (
    SGLANG_SINGLE_TEST_ORDER,
    SGLANG_TEST_ORDER,
)

log = globals.log

# Re-exported for sglang_single.py / sglang_disagg_distributed.py imports.
__all__ = ["flat_expected_from_specs"]


def _use_sglang_single(request) -> bool:
    """``sglang_single.py`` uses unified single-node ``SglangSingle``."""
    return getattr(request.module, "__name__", "").endswith("sglang_single")


def _deep_merge(base, override):
    """Recursively merge ``override`` onto ``base`` (dicts merged key-wise; scalars/lists replaced)."""
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(base[k], v) if k in base else v
    return out


def _benchmark_serv_host(inference: Mapping[str, Any]) -> str:
    """Single-node suite target host from inference ``benchmark_serv_node``."""
    raw = inference.get("benchmark_serv_node")
    if not raw:
        raise ValueError(
            "sglang_single requires 'benchmark_serv_node' in the inference config "
            "(config.json / variant inference dict)"
        )
    hosts = as_node_list(raw)
    if len(hosts) != 1:
        raise ValueError(
            f"sglang_single requires exactly one benchmark_serv_node, got {hosts!r}"
        )
    return hosts[0]


def _cluster_dict_for_single_benchmark(
    cluster_dict: Mapping[str, Any],
    bench_host: str,
) -> dict[str, Any]:
    """Restrict orchestrator SSH/container scope to ``benchmark_serv_node`` only."""
    node_dict = cluster_dict.get("node_dict") or {}
    if bench_host not in node_dict:
        raise ValueError(
            f"benchmark_serv_node {bench_host!r} is not listed in cluster node_dict "
            f"(keys: {sorted(node_dict)!r})"
        )
    scoped = dict(cluster_dict)
    scoped["node_dict"] = {bench_host: node_dict[bench_host]}
    scoped["head_node_dict"] = {"mgmt_ip": bench_host}
    return scoped


def _disagg_role_hosts(inference: Mapping[str, Any]) -> list[str]:
    """Unique hosts referenced by PD role fields in the inference config."""
    seen: list[str] = []
    for key in (
        "prefill_node_list",
        "decode_node_list",
        "proxy_router_node",
        "benchmark_serv_node",
    ):
        raw = inference.get(key)
        if raw is None:
            continue
        for host in as_node_list(raw):
            if host not in seen:
                seen.append(host)
    if not seen:
        raise ValueError(
            "sglang_disagg requires at least one of prefill_node_list, decode_node_list, "
            "proxy_router_node, or benchmark_serv_node in the inference config"
        )
    return seen


def _disagg_head_host(inference: Mapping[str, Any], role_hosts: list[str]) -> str:
    """Orchestrator head: proxy, then benchmark, then first prefill node."""
    for key in ("proxy_router_node", "benchmark_serv_node", "prefill_node_list"):
        raw = inference.get(key)
        if raw is None:
            continue
        hosts = as_node_list(raw)
        if hosts:
            return hosts[0]
    return role_hosts[0]


def _cluster_dict_for_disagg_roles(
    cluster_dict: Mapping[str, Any],
    role_hosts: list[str],
    head_host: str,
) -> dict[str, Any]:
    """Restrict orchestrator scope to PD role hosts (not every cluster.json node)."""
    node_dict = cluster_dict.get("node_dict") or {}
    missing = [h for h in role_hosts if h not in node_dict]
    if missing:
        raise ValueError(
            f"disagg role hosts not in cluster node_dict: {missing!r} "
            f"(cluster keys: {sorted(node_dict)!r})"
        )
    scoped = dict(cluster_dict)
    scoped["node_dict"] = {h: node_dict[h] for h in role_hosts}
    scoped["head_node_dict"] = {"mgmt_ip": head_host}
    return scoped


def _create_container_orchestrator(cluster_dict: Mapping[str, Any], variant_config: SglangSingleVariantConfig):
    """Build a ``ContainerOrchestrator`` for single-node or disagg SGLang suites."""
    container_block = _deep_merge(
        cluster_dict.get("container", {}),
        orchestrator_container_from_variant(variant_config),
    )
    testsuite_config = {
        "orchestrator": "container",
        "container": container_block,
    }
    cfg = OrchestratorConfig.from_configs(cluster_dict, testsuite_config)
    return OrchestratorFactory.create_orchestrator(log, cfg)


# ---------- accuracy-cell helpers (disagg long-context parametrization) ----------

_ACC_CELL_RE = re.compile(
    r"^ACC_ISL=(?P<isl>\d+),OSL=(?P<osl>\d+)$"
)


def acc_cells_from_thresholds(thresholds: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = []
    for cell_key, specs in thresholds.items():
        if str(cell_key).startswith("_"):
            continue
        m = _ACC_CELL_RE.match(str(cell_key))
        if not m:
            continue
        cells.append({
            "cell_key": cell_key,
            "isl": m.group("isl"),
            "osl": m.group("osl"),
            "specs": specs,
        })
    cells.sort(key=lambda c: int(c["isl"]))
    return cells


def _cluster_dict_for_collection(metafunc) -> dict[str, Any]:
    cluster_file = metafunc.config.getoption("cluster_file")
    if not cluster_file or not os.path.isfile(cluster_file):
        return {}
    with open(cluster_file, encoding="utf-8") as fp:
        return resolve_cluster_config_placeholders(json.load(fp))


def load_acc_cells_for_collection(
    config_file: str,
    cluster_dict: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    variant = load_variant(config_file, cluster_dict or {})
    cells = acc_cells_from_thresholds(variant.thresholds)
    if not cells:
        pytest.fail(f"No ACC_ISL=... accuracy cells in thresholds for {config_file!r}")
    return cells


def pytest_generate_tests(metafunc):
    config_file = metafunc.config.getoption("config_file")
    if not config_file or not os.path.isfile(config_file):
        return

    cluster_dict = _cluster_dict_for_collection(metafunc)

    if "perf_cell" in metafunc.fixturenames:
        variant = load_variant(config_file, cluster_dict)
        cells = perf_cells_from_thresholds(variant.thresholds)
        if not cells:
            pytest.fail(f"No ISL=... performance cells in thresholds for {config_file!r}")
        ids = [f"isl{c['isl']}-osl{c['osl']}-c{c['conc']}" for c in cells]
        metafunc.parametrize("perf_cell", cells, ids=ids)

    if "acc_cell" in metafunc.fixturenames:
        cells = load_acc_cells_for_collection(config_file, cluster_dict)
        ids = [f"acc-isl{c['isl']}-osl{c['osl']}" for c in cells]
        metafunc.parametrize("acc_cell", cells, ids=ids)


# ---------- lifecycle (same model as vLLM conftest) ----------


class _Lifecycle:
    """Cross-test state for the lifecycle-as-tests model."""

    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report: dict[str, list[tuple[str, float, str]]] = {}
        self.phase_labels: dict[str, Any] = {}
        self.smoke_results: list | None = None

    def record(self, nodeid: str, label: str, value: float, unit: str = "s") -> None:
        self.report.setdefault(nodeid, []).append((label, value, unit))

    def skip_if_prior_failure(self) -> None:
        if self.failed:
            pytest.skip("a prior lifecycle stage failed")

    def complete_stage(self, request, label: str, t0: float) -> None:
        self.record(request.node.nodeid, label, time.monotonic() - t0)
        if globals.error_list:
            self.failed = True
        update_test_result()


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
def variant_config(pytestconfig, cluster_dict) -> SglangSingleVariantConfig:
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    return load_variant(config_file, cluster_dict)


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


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
    path = variant_config.paths.hf_token_file
    if not os.path.isfile(path):
        if variant_config.model.remote == 0:
            return ""
        pytest.skip(f"hf_token file missing: {path}")
    with open(path, encoding="utf-8") as fp:
        return fp.read().strip()


@pytest.fixture(scope="module")
def orch(request, cluster_dict, variant_config, lifecycle):
    """``ContainerOrchestrator`` for both single-node and disagg SGLang suites."""
    cluster = cluster_dict
    if _use_sglang_single(request):
        bench_host = _benchmark_serv_host(variant_config.inference)
        cluster = _cluster_dict_for_single_benchmark(cluster_dict, bench_host)
        log.info("sglang_single: orchestrator scoped to benchmark_serv_node=%s", bench_host)
    else:
        role_hosts = _disagg_role_hosts(variant_config.inference)
        head_host = _disagg_head_host(variant_config.inference, role_hosts)
        cluster = _cluster_dict_for_disagg_roles(cluster_dict, role_hosts, head_host)
        log.info(
            "sglang_disagg: orchestrator scoped to role hosts=%s head=%s",
            role_hosts,
            head_host,
        )
    o = _create_container_orchestrator(cluster, variant_config)

    yield o

    if not lifecycle.torn_down:
        log.info("orch leak-guard: tearing down containers")
        o.teardown_containers()
        if hasattr(o, "cleanup_log_dir"):
            o.cleanup_log_dir(variant_config.paths.log_dir)


@pytest.fixture(scope="module")
def gpu_type(request, orch, variant_config):
    if _use_sglang_single(request):
        bench_host = _benchmark_serv_host(variant_config.inference)
        smi_out_dict = orch.all.exec("rocm-smi -a | head -30")
        smi_out = smi_out_dict.get(bench_host) or next(iter(smi_out_dict.values()))
    else:
        # Disagg: probe first prefill node (may differ from cluster head).
        prefill_nodes = variant_config.inference["prefill_node_list"]
        probe_node = prefill_nodes[0] if isinstance(prefill_nodes, list) else prefill_nodes
        smi_out_dict = orch.all.exec("rocm-smi -a | head -30")
        smi_out = smi_out_dict.get(probe_node) or next(iter(smi_out_dict.values()))
    return get_model_from_rocm_smi_output(smi_out)


@pytest.fixture(scope="module")
def inf_res_dict():
    return {}


@pytest.fixture(scope="module")
def im_obj(
    request,
    orch,
    gpu_type,
    variant_config,
    hf_token,
):
    model_name = variant_config.benchmark_params["model"]

    if _use_sglang_single(request):
        return SglangSingle(
            model_name,
            variant_config.inference,
            variant_config.benchmark_params,
            hf_token,
            orch=orch,
            gpu_type=gpu_type,
        )
    return SglangDisaggPD(
        model_name,
        variant_config.inference,
        variant_config.benchmark_params,
        hf_token,
        orch=orch,
        gpu_type=gpu_type,
    )


# ---------- pytest hooks ----------


def pytest_collection_modifyitems(items):
    def rank_for(item):
        mod = getattr(item.module, "__name__", "")
        order = SGLANG_SINGLE_TEST_ORDER if mod.endswith("sglang_single") else SGLANG_TEST_ORDER
        return order.get(item.originalname or item.name.split("[")[0], 99)

    items.sort(key=rank_for)


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