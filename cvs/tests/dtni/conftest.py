"""Shared pytest plumbing for all DTNI framework tests in this directory.

Every DTNI framework test (vllm_single, pytorch_single, ...) drives one
workload and emits one PASS/FAIL node per threshold metric. The
collection-time parametrization and the per-module workload fixture are
identical across frameworks, so they live here.

Each framework file in this directory only declares its `test_threshold`
body; this conftest supplies the `metric` parameter values and the
`workload_outcome` fixture they consume.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cvs.lib.dtni.runner import execute_workload


def _threshold_metrics(config_file: str | None) -> list[str]:
    """Best-effort list of metric names for collection-time parametrize.

    `cvs list <test>` invokes pytest --collect-only with `--config_file=dummy`,
    so we must not raise on a missing/unreadable path — return a sentinel
    instead so collection succeeds and the customer sees the test shape.
    The matching test then skips on the sentinel.
    """
    if not config_file or config_file == "dummy":
        return ["__no_config_file__"]
    threshold_path = Path(config_file).parent / "threshold.json"
    if not threshold_path.exists():
        return ["__no_threshold_json__"]
    try:
        metrics = list(json.loads(threshold_path.read_text()).keys())
    except (OSError, json.JSONDecodeError):
        return ["__threshold_json_unreadable__"]
    return metrics or ["__no_metrics_defined__"]


def pytest_generate_tests(metafunc):
    """Parametrize `metric` from the threshold.json sibling of --config_file."""
    if "metric" not in metafunc.fixturenames:
        return
    config_file = metafunc.config.getoption("config_file")
    metafunc.parametrize("metric", _threshold_metrics(config_file))


@pytest.fixture(scope="module")
def workload_outcome(pytestconfig):
    """Run the workload exactly once per module, shared across metric nodes.

    On phase failure (launch/await/parse), still returns the outcome — the
    per-metric tests inspect failed_phase and skip rather than fail.
    """
    cluster_file = pytestconfig.getoption("cluster_file")
    config_file = pytestconfig.getoption("config_file")
    if not cluster_file or not config_file:
        pytest.fail("DTNI tests require --cluster_file and --config_file")
    return execute_workload(
        cluster_path=cluster_file,
        workload_config_path=config_file,
    )
