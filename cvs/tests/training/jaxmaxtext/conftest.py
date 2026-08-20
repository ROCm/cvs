'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.
'''

import json
import os

import pytest

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory
from cvs.lib import globals
from cvs.lib.training.jaxmaxtext.utils.maxtext_parsing import TRAINING_METRICS
from cvs.lib.training.jaxmaxtext.utils.training_config_loader import (
    load_training_variant,
    validate_thresholds_cover_training,
)
from cvs.lib.utils_lib import resolve_cluster_config_placeholders
from cvs.tests.training.jaxmaxtext import _common

log = globals.log


def pytest_generate_tests(metafunc):
    """Parametrize per-sweep tests for BOTH suites (single + distributed):
    training_run/loss_curve over sweeps, metric over (sweep x TRAINING_METRICS)."""
    config_file = metafunc.config.getoption("config_file")
    if config_file and os.path.isfile(config_file):
        names = _common._enabled_sweep_names(config_file)
    else:
        names = ["default"]
    labels = [_common._sweep_label(n) for n in names]

    if "metric" in metafunc.fixturenames and "sweep_name" in metafunc.fixturenames:
        cases, ids = [], []
        for name, label in zip(names, labels):
            for short, _unit in TRAINING_METRICS:
                cases.append((name, short))
                ids.append(f"{label}-{short}")
        metafunc.parametrize("sweep_name,metric", cases, ids=ids)
    elif "sweep_name" in metafunc.fixturenames:
        metafunc.parametrize("sweep_name", names, ids=labels)


def _deep_merge(base, override):
    """Recursively merge `override` onto `base` (dicts merged key-wise, scalars/lists replaced)."""
    if not (isinstance(base, dict) and isinstance(override, dict)):
        return override
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(base[k], v) if k in base else v
    return out


@pytest.fixture(scope="module")
def cluster_dict(pytestconfig):
    cluster_file = pytestconfig.getoption("cluster_file")
    if not cluster_file:
        pytest.fail("--cluster_file is required")
    with open(cluster_file) as fp:
        d = json.load(fp)
    return resolve_cluster_config_placeholders(d)


@pytest.fixture(scope="module")
def variant_config(pytestconfig, cluster_dict):
    config_file = pytestconfig.getoption("config_file")
    if not config_file:
        pytest.fail("--config_file is required")
    variant = load_training_variant(config_file, cluster_dict)
    # Fail fast on a sweep-name/threshold-key mismatch: otherwise metric() would
    # silently take the "spec is None" path and emit non-gating RECORD rows, so
    # the suite would report green while gating nothing. Raises when
    # enforce_thresholds is true; warns otherwise.
    validate_thresholds_cover_training(
        expected_cells=variant.expected_cells(),
        thresholds=variant.thresholds,
        enforce_thresholds=variant.enforce_thresholds,
    )
    return variant


@pytest.fixture(scope="module", autouse=True)
def _guard_suite_matches_config(request, variant_config):
    """Fail fast when the suite and the config disagree on distributed mode.

    Both jaxmaxtext_single and jaxmaxtext_distributed share this conftest. Without
    this guard a mismatched pairing -- e.g. `cvs run jaxmaxtext_single` with a
    distributed config (skips RDMA, still launches multi-node JAX), or
    `jaxmaxtext_distributed` with a single-node config -- would start and fail
    late with confusing errors. Catch it at setup instead.
    """
    mod = (request.module.__name__ or "").rsplit(".", 1)[-1]
    distributed = variant_config.training.distributed
    if mod.endswith("_single") and distributed:
        pytest.fail(
            "suite/config mismatch: jaxmaxtext_single requires a single-node config "
            "(training.distributed=false), but this config has distributed=true. "
            "Run jaxmaxtext_distributed or point at a single-node config."
        )
    if mod.endswith("_distributed") and not distributed:
        pytest.fail(
            "suite/config mismatch: jaxmaxtext_distributed requires a distributed config "
            "(training.distributed=true), but this config has distributed=false. "
            "Run jaxmaxtext_single or point at a distributed config."
        )


class _Lifecycle:
    """Cross-test state for the lifecycle-as-tests model.

    `failed` lets a broken stage skip the rest instead of cascading;
    `torn_down` lets the explicit teardown test suppress the fixture's
    leak-guard finalizer.
    """

    def __init__(self):
        self.failed = False
        self.torn_down = False
        self.report = {}
        self.artifacts = {}

    def record(self, nodeid, label, value, unit="s"):
        self.report.setdefault(nodeid, []).append((label, value, unit))

    def add_artifact(self, nodeid, name, rel_path, abs_path):
        """Register a per-test report artifact (e.g. loss-curve PNG) for linking."""
        self.artifacts.setdefault(nodeid, []).append((name, rel_path, abs_path))


@pytest.fixture(scope="module")
def lifecycle():
    return _Lifecycle()


@pytest.fixture(scope="module")
def orch(cluster_dict, variant_config, lifecycle):
    """Construct a ContainerOrchestrator and own ONLY its teardown safety net."""
    container_block = _deep_merge(
        cluster_dict.get("container", {}),
        variant_config.container.model_dump(),
    )
    testsuite_config = {
        "orchestrator": "container",
        "container": container_block,
    }
    cfg = OrchestratorConfig.from_configs(cluster_dict, testsuite_config)
    o = OrchestratorFactory.create_orchestrator(log, cfg)
    yield o
    if not lifecycle.torn_down:
        log.info("orch fixture leak-guard: tearing down container (explicit teardown did not run)")
        o.teardown_containers()


@pytest.fixture(scope="module")
def hf_token(variant_config):
    path = variant_config.paths.hf_token_file
    if not os.path.isfile(path):
        pytest.skip(f"hf_token file missing: {path}")
    with open(path) as fp:
        return fp.read().strip()


@pytest.fixture(scope="module")
def training_res_dict():
    return {}


def pytest_collection_modifyitems(items):
    """Pin the lifecycle order explicitly."""
    rank = {
        "test_launch_container": 0,
        "test_setup_rdma": 1,
        "test_setup_tokenizer": 2,
        "test_smoke": 3,
        "test_training_run": 4,
        "test_metric": 5,
        "test_loss_curve": 6,
        "test_checkpoint_resume": 7,
        "test_print_results_table": 8,
        "test_teardown": 9,
    }
    items.sort(key=lambda it: rank.get(it.originalname or it.name.split("[")[0], 99))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Attach THIS test's recorded rows to its HTML report detail panel."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    try:
        import pytest_html
    except ImportError:
        return

    lc = item.funcargs.get("lifecycle")
    rows = getattr(lc, "report", {}).get(item.nodeid) if lc else None
    artifacts = getattr(lc, "artifacts", {}).get(item.nodeid) if lc else None

    # Each metric row gets an EXTRA "Metric Results" link to the single shared
    # metric-results HTML file (written by test_print_results_table), in addition
    # to its own per-test "Full Log" link (kept as-is). Two links per metric row.
    metric_link = None
    if (item.originalname or "") == "test_metric":
        mgr = getattr(item.config, "_html_report_manager", None)
        if mgr is not None and getattr(mgr, "is_enabled", False):
            metric_link = f"{mgr._test_html_dir}/metric_results.html"

    if not rows and not artifacts and not metric_link:
        return

    extras = getattr(report, "extras", [])

    if metric_link:
        extras.append(pytest_html.extras.url(metric_link, name="Metric Results"))

    if rows:
        body = "".join(f"<tr><td>{label}</td><td>{value:.1f}</td><td>{unit}</td></tr>" for label, value, unit in rows)
        html = f"<table><tr><th>stage</th><th>value</th><th>unit</th></tr>{body}</table>"
        extras.append(pytest_html.extras.html(html))

    for name, rel_path, abs_path in artifacts or []:
        # Primary: a clickable link to the PNG bundled next to the report.
        extras.append(pytest_html.extras.url(rel_path, name=name))
        # Best-effort inline thumbnail (base64); never break the row if it fails.
        try:
            import base64

            with open(abs_path, "rb") as fp:
                b64 = base64.b64encode(fp.read()).decode("ascii")
            extras.append(pytest_html.extras.png(b64, name=name))
        except Exception:  # noqa: BLE001
            pass

    report.extras = extras
