'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.

Cluster hardware topology discovery suite.

Fans a per-node topology probe across every node in the cluster file, stitches
the per-node records into one cluster graph, persists the graph as JSON under
the run directory, and renders a graphical HTML view attached to the pytest
report bundle. The cluster analogue of ANC's single-node topology tree.

Verdict semantics: topology itself is not pass/fail. The test passes when every
node declared in the cluster file was reached and parsed, and fails listing the
nodes that did not respond -- so the suite doubles as a reachability check.
'''

import json
import os

import pytest

from cvs.core.run_layout import RunLayout
from cvs.lib import globals, html_lib
from cvs.lib.topology import cluster_graph, topology_probe
from cvs.lib.utils_lib import (
    fail_test,
    resolve_cluster_config_placeholders,
    resolve_test_config_placeholders,
    update_test_result,
)

log = globals.log


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
    config_dict = config_dict_t["topology"]
    config_dict = resolve_test_config_placeholders(config_dict, cluster_dict)
    log.info("%s", config_dict)
    return config_dict


def _node_label(host):
    '''A filesystem-safe per-node directory label.'''
    return host.replace("/", "_").replace(":", "_").replace(" ", "_")


def _persist_artifacts(graph, per_node_raw):
    '''Write the cluster graph JSON and each node's raw probe output to run_dir.

    Returns the path to the cluster_topology.json for logging.
    '''
    topo_dir = RunLayout.get().run_dir / "topology"
    topo_dir.mkdir(parents=True, exist_ok=True)

    graph_path = topo_dir / "cluster_topology.json"
    with open(graph_path, "w") as fp:
        json.dump(graph.to_dict(), fp, indent=2)

    for host, raw in per_node_raw.items():
        node_dir = topo_dir / _node_label(host)
        node_dir.mkdir(parents=True, exist_ok=True)
        with open(node_dir / "probe_raw.txt", "w") as fp:
            fp.write(raw if isinstance(raw, str) else str(raw))

    return graph_path


def test_discover_topology(orch, cluster_dict, config_dict, request):
    '''Probe every node, stitch a cluster topology, persist it, and render HTML.'''
    expected_nodes = set(cluster_dict.get("node_dict", {}).keys())
    if not expected_nodes:
        fail_test("cluster file has no node_dict entries to probe")
        update_test_result()
        return

    probe_timeout = int(config_dict.get("probe_timeout", topology_probe.DEFAULT_PROBE_TIMEOUT))
    cmd = topology_probe.build_probe_cmd()

    out_dict = orch.exec(cmd, timeout=probe_timeout)

    # orch.exec returns only hosts that responded; the rest are unreachable.
    returned_nodes = set(out_dict.keys())
    missing_nodes = expected_nodes - returned_nodes

    per_node = {}
    for host, raw in out_dict.items():
        record = topology_probe.parse_node_topology(raw, host=host)
        per_node[host] = record
        if not record["reachable"]:
            fail_test(f"topology probe returned no usable data from node '{host}'")
        else:
            log.info(
                "node %s: hostname=%s gpus=%d nics=%d",
                host,
                record["hostname"],
                record["gpu_count"],
                len(record["nics"]),
            )

    for host in missing_nodes:
        fail_test(f"node '{host}' did not respond to the topology probe")

    graph = cluster_graph.stitch(per_node)
    log.info("cluster topology summary: %s", graph.summary())

    graph_path = _persist_artifacts(graph, out_dict)
    log.info("cluster topology JSON written to %s", graph_path)

    html_file = f"/tmp/cluster_topology_{os.getpid()}.html"
    html_lib.build_topology_graph(html_file, graph.to_dict())
    copied_path = request.config._html_report_manager.add_html_to_report(
        html_file, link_name="Cluster Topology", request=request
    )
    if copied_path:
        log.info("Cluster topology report added to bundle: %s", copied_path)
    else:
        log.info("Cluster topology report saved under %s", html_file)

    update_test_result()
