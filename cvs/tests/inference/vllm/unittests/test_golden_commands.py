'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Golden commands for every shipped vLLM config.

For each config in cvs/input/config_file/inference/vllm/ (threshold files excluded), this runs
the suite's own stages against an orchestrator that only records commands, and compares the
result with golden_commands.json:

  - test_ids: the pytest test IDs of vllm_single or vllm_distributed, in run order, from the
    module's own pytest_generate_tests and the suite conftest's collection order.
  - model_fetch, smoke: the commands _common.model_fetch and _common.openai_compatible_smoke send.
  - runs.<cell>: the commands _common.vllm_inference sends for that cell. Cells run in order on
    one lifecycle, so a cell that reuses the live server sends no server commands.
  - gates.<cell>: the enforce_thresholds flag and the threshold specs _verify_cell_metrics
    passes to evaluate_metric_verdicts for that cell.
  - accuracy.<task>: the lm-eval command test_accuracy_eval sends for that task.
  - container: the container block the suite passes to the orchestrator for docker run, after
    the merge with the cluster file's. The fake cluster file's container block has one volume,
    one device and one env var, so the entry shows whether the config's lists and env replace
    or extend the cluster's.

Each command reads "<target>: <command>", or "<target> [timeout=<s>]: <command>" when the exec
passes a timeout; the target is "all", a host, or "head". Single configs get a one-host cluster
file and distributed configs a two-host one; every shipped distributed config uses
pipeline_parallel_size 2. Every shipped config must have an entry, and every entry must match.

Mocked, so nothing waits or reads results: time.sleep, VllmJob.wait_ready,
wait_client_complete, parse_results and probe_openai_endpoints, the GPU snapshot and poller,
the Prometheus scrape, and evaluate_metric_verdicts, which is recorded and returns no verdicts.
model_fetch's du gets a nonzero size. test_discover_topology is not run, so the stages read
lifecycle.ib_hcas as []; that is what discovery gives every shipped config, since single runs
skip it and every distributed config sets NCCL_IB_HCA.

Each <changeme> becomes filled-<leaf key>, or filled-<key>-<index> inside a list (filled-model,
filled-image, filled-volumes-1), so two swapped fields still show.

_load_config is the only step tied to a config loader and config format. A port's version
takes the same arguments (a shipped config, the directory for its filled copy, the fake cluster
file) and must return (config_file, config, container):
  - config_file: a file the suite's pytest_generate_tests loads, made from
    _write_filled_config's output so every filled-* token survives;
  - config: the object the suite's fixtures give its stages as variant_config;
  - container: the merged container block the suite passes to the orchestrator.
The golden file must then match unchanged.

Normalized: the temporary directory holding the filled config copies becomes <tmp>; no shipped
config renders it today. Paths come from the fake cluster's fixed username, and no rendered
command holds a timestamp or a run directory.

After an intended change, regenerate golden_commands.json from the repo root with:

    python -m cvs.tests.inference.vllm.unittests.test_golden_commands --write
'''

import contextlib
import inspect
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cvs.lib.conf_lib.loader import load_config
from cvs.lib.conf_lib.schemas.vllm import VariantConfig
from cvs.lib.inference.utils import inference_suite_lifecycle
from cvs.lib.inference.vllm_job import VllmJob
from cvs.lib.inference.vllm_topology import resolve_vllm_topology, scope_vllm_cluster
from cvs.tests.inference.vllm import _common
from cvs.tests.inference.vllm import conftest as vllm_conftest
from cvs.tests.inference.vllm import vllm_distributed, vllm_single

CONFIG_DIR = Path(__file__).resolve().parents[4] / "input" / "config_file" / "inference" / "vllm"
GOLDEN = Path(__file__).with_name("golden_commands.json")
TMP_TOKEN = "<tmp>"
MISSING = "<missing>"
CONTEXT = 40
SUITES = {"single": vllm_single, "distributed": vllm_distributed}
HOSTS = {"single": ["node0"], "distributed": ["node0", "node1"]}
CLUSTER_CONTAINER = {
    "runtime": {"args": {"volumes": ["/cluster-volume:/cluster-volume"], "devices": ["/dev/cluster-device"]}},
    "env": {"CLUSTER_ENV_VAR": "from-cluster"},
}
_PLACEHOLDER_RE = re.compile(r"<changeme(?:-[^<>]*)?>", re.IGNORECASE)


def _write_filled_config(template, out_dir):
    """Write a copy of a shipped config with each <changeme> filled with its own token.

    Comment keys are copied unchanged, and threshold_json is made absolute so the copy still
    reads the shipped threshold file. Returns the copy's path.
    """

    def fill(node, name):
        if isinstance(node, dict):
            return {key: item if key.startswith("_") else fill(item, key) for key, item in node.items()}
        if isinstance(node, list):
            return [fill(item, f"{name}-{index}") for index, item in enumerate(node)]
        if isinstance(node, str):
            return _PLACEHOLDER_RE.sub(f"filled-{name}", node)
        return node

    config = fill(json.loads(template.read_text()), "")
    config["threshold_json"] = str(template.parent / config["threshold_json"])
    out_path = Path(out_dir) / template.name
    out_path.write_text(json.dumps(config))
    return out_path


def _load_config(template, out_dir, cluster_file):
    """Load one shipped config for the suite. Returns (config_file, config, container)."""
    config_file = _write_filled_config(template, out_dir)
    loaded = load_config(VariantConfig, config_file, cluster_file)
    return config_file, loaded.config, loaded.container


def _shipped_configs():
    return sorted(path for path in CONFIG_DIR.glob("*.json") if not path.name.endswith("threshold.json"))


def _mode(template):
    return "distributed" if template.stem.endswith("_distributed") else "single"


def _cluster(mode):
    hosts = HOSTS[mode]
    return {
        "username": "cvs",
        "node_dict": {host: {"vpc_ip": host} for host in hosts},
        "head_node_dict": {"mgmt_ip": hosts[0]},
        "container": CLUSTER_CONTAINER,
    }


def _cluster_file(mode, out_dir):
    path = Path(out_dir) / f"cluster_{mode}.json"
    path.write_text(json.dumps(_cluster(mode)))
    return path


class _Captured(Exception):
    pass


class _RecordingOrch:
    def __init__(self, hosts):
        self.hosts = list(hosts)
        self.stop_on_head = False
        self.commands = []

    def _record(self, target, cmd, timeout):
        label = target if timeout is None else f"{target} [timeout={timeout}]"
        self.commands.append(f"{label}: {cmd}")

    def exec(self, cmd, hosts=None, timeout=None, **kwargs):
        self._record(",".join(hosts) if hosts else "all", cmd, timeout)
        if cmd.startswith("du -sb "):
            return {host: "1" for host in hosts or self.hosts}
        return {}

    def exec_on_head(self, cmd, timeout=None, **kwargs):
        self._record("head", cmd, timeout)
        if self.stop_on_head:
            raise _Captured()
        return {}

    def take(self):
        commands, self.commands = self.commands, []
        return commands


class _PytestConfig:
    def __init__(self, config_file):
        self.config_file = str(config_file)

    def getoption(self, name, default=None):
        return self.config_file if name == "config_file" else default


class _Metafunc:
    def __init__(self, function, config_file):
        self.function = function
        self.fixturenames = list(inspect.signature(function).parameters)
        self.config = _PytestConfig(config_file)
        self.values = None
        self.ids = None

    def parametrize(self, argnames, argvalues, ids=None, **kwargs):
        self.values = list(argvalues)
        self.ids = list(ids)


def _collect(module, config_file):
    """Return the test IDs pytest would collect from module, and each parametrized test's values."""
    items = []
    values = {}
    for name, function in vars(module).items():
        if not (name.startswith("test_") and inspect.isfunction(function)):
            continue
        metafunc = _Metafunc(function, config_file)
        module.pytest_generate_tests(metafunc)
        if metafunc.ids is None:
            items.append(SimpleNamespace(originalname=name, name=name))
            continue
        values[name] = metafunc.values
        items.extend(SimpleNamespace(originalname=name, name=f"{name}[{id_}]") for id_ in metafunc.ids)
    vllm_conftest.pytest_collection_modifyitems(_PytestConfig(config_file), items)
    return [item.name for item in items], values


def _request(nodeid):
    return SimpleNamespace(
        node=SimpleNamespace(nodeid=nodeid, user_properties=[]),
        config=SimpleNamespace(option=SimpleNamespace()),
    )


@contextlib.contextmanager
def _stage_mocks():
    """Patch out every wait and result read; yields the evaluate_metric_verdicts recorder."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch("time.sleep"))
        stack.enter_context(mock.patch.object(VllmJob, "wait_ready"))
        stack.enter_context(mock.patch.object(VllmJob, "wait_client_complete"))
        stack.enter_context(mock.patch.object(VllmJob, "probe_openai_endpoints", return_value=[]))
        stack.enter_context(mock.patch.object(VllmJob, "parse_results", return_value={"head": {}}))
        stack.enter_context(mock.patch.object(_common, "capture_gpu_metrics", return_value={}))
        stack.enter_context(mock.patch.object(_common, "start_gpu_poller"))
        stack.enter_context(mock.patch.object(_common, "stop_and_collect_gpu_poller", return_value=[]))
        stack.enter_context(mock.patch.object(_common, "scrape_vllm_metrics", return_value=None))
        yield stack.enter_context(mock.patch.object(_common, "evaluate_metric_verdicts", return_value=[]))


def _render(mode, config_file, variant, container):
    hosts = list(scope_vllm_cluster(mode, _cluster(mode))["node_dict"])
    topology = resolve_vllm_topology(mode, variant, hosts)
    variant.bind_effective_topology(topology)
    test_ids, values = _collect(SUITES[mode], config_file)

    orch = _RecordingOrch(hosts)
    lifecycle = vllm_conftest._Lifecycle()
    results = {}
    runs = {}
    gates = {}
    with _stage_mocks() as verdicts:
        _common.model_fetch(orch, variant, lifecycle, _request("test_model_fetch"))
        model_fetch = orch.take()
        _common.openai_compatible_smoke(
            orch, variant, topology.target_groups, lifecycle, _request("test_openai_compatible_smoke")
        )
        smoke = orch.take()
        for run in values.get("test_vllm_inference", []):
            verdicts.reset_mock()
            request = _request(f"test_vllm_inference[{run.cell.key}]")
            _common.vllm_inference(
                orch, variant, topology.target_groups, run, results, lifecycle, request, subtests=None
            )
            runs[run.cell.key] = orch.take()
            if verdicts.call_count != 1:
                raise AssertionError(f"{request.node.nodeid} evaluated its thresholds {verdicts.call_count} times")
            (_, thresholds), kwargs = verdicts.call_args
            gates[run.cell.key] = {"enforce_thresholds": kwargs["enforce_thresholds"], "thresholds": thresholds}

    orch.stop_on_head = True
    accuracy = {}
    for task_id in values.get("test_accuracy_eval", []):
        request = _request(f"test_accuracy_eval[{task_id}]")
        try:
            inference_suite_lifecycle.test_accuracy_eval(orch, variant, task_id, lifecycle, request)
        except _Captured:
            (accuracy[task_id],) = orch.take()
        else:
            raise AssertionError(f"test_accuracy_eval[{task_id}] sent no lm-eval command")

    return {
        "suite": SUITES[mode].__name__.rsplit(".", 1)[-1],
        "test_ids": test_ids,
        "model_fetch": model_fetch,
        "smoke": smoke,
        "runs": runs,
        "gates": gates,
        "accuracy": accuracy,
        "container": container,
    }


def _replace(value, old, new):
    if isinstance(value, dict):
        return {key: _replace(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace(item, old, new) for item in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def render_config(template, tmp_dir):
    """Render the golden entry for one shipped config, writing its filled copy under tmp_dir."""
    mode = _mode(template)
    config_file, variant, container = _load_config(template, tmp_dir, _cluster_file(mode, tmp_dir))
    return _replace(_render(mode, config_file, variant, container), str(tmp_dir), TMP_TOKEN)


def write_golden(path=GOLDEN):
    with tempfile.TemporaryDirectory() as tmp:
        golden = {template.name: render_config(template, Path(tmp)) for template in _shipped_configs()}
    path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")


def _differences(expected, actual, path=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            where = f"{path}.{key}" if path else key
            yield from _differences(expected.get(key, MISSING), actual.get(key, MISSING), where)
    elif isinstance(expected, list) and isinstance(actual, list):
        for index in range(max(len(expected), len(actual))):
            yield from _differences(
                expected[index] if index < len(expected) else MISSING,
                actual[index] if index < len(actual) else MISSING,
                f"{path}[{index}]",
            )
    elif expected != actual:
        yield path, expected, actual


def _shown(text):
    return text.replace("\n", "\\n")


def _clipped(value):
    text = _shown(value if isinstance(value, str) else json.dumps(value))
    return text if len(text) <= 2 * CONTEXT else text[: 2 * CONTEXT] + "..."


def _first_difference(expected, actual):
    if MISSING in (expected, actual) or not (isinstance(expected, str) and isinstance(actual, str)):
        return [f"golden:   {_clipped(expected)}", f"rendered: {_clipped(actual)}"]
    column = next(
        (index for index, (old, new) in enumerate(zip(expected, actual)) if old != new),
        min(len(expected), len(actual)),
    )
    start = max(column - CONTEXT, 0)
    lead = "..." if start else ""
    return [
        f"first difference at column {column}",
        f"golden:   {lead}{_shown(expected[start : column + CONTEXT])}",
        f"rendered: {lead}{_shown(actual[start : column + CONTEXT])}",
    ]


def _failure_message(config_name, differences):
    lines = [f"{config_name}: rendered entry differs from {GOLDEN.name}"]
    for path, expected, actual in differences:
        lines.append(f"  {path}")
        lines.extend(f"    {line}" for line in _first_difference(expected, actual))
    return "\n".join(lines)


class TestGoldenCommands(unittest.TestCase):
    def setUp(self):
        self.golden = json.loads(GOLDEN.read_text())
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_every_shipped_config_has_a_golden_entry(self):
        shipped = {template.name for template in _shipped_configs()}
        self.assertTrue(shipped)
        self.assertEqual(sorted(shipped - set(self.golden)), [], "shipped configs without a golden entry")
        self.assertEqual(sorted(set(self.golden) - shipped), [], "golden entries without a shipped config")

    def test_every_shipped_config_renders_its_golden_commands(self):
        for template in _shipped_configs():
            with self.subTest(config=template.name):
                if template.name not in self.golden:
                    self.fail(f"{template.name}: no entry in {GOLDEN.name}")
                differences = list(_differences(self.golden[template.name], render_config(template, self.tmp)))
                if differences:
                    self.fail(_failure_message(template.name, differences))


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        write_golden()
    else:
        unittest.main()
