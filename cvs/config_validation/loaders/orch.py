'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Builds a suite's orchestrator from its loaded config, on the hosts its schema
selects.
'''

from copy import deepcopy

from cvs.core.orchestrators.factory import OrchestratorConfig, OrchestratorFactory


def cluster_with_hosts(cluster, hosts):
    """The cluster with ``node_dict`` narrowed to ``hosts``, or ``cluster`` itself when ``hosts`` is None.

    ``node_dict`` keeps its form, a map keyed by host or a list of nodes keyed
    by ``mgmt_ip``, and its order, since the orchestrators take the first node
    as head. ``head_node_dict.mgmt_ip`` names that node. The input is not
    changed.
    """
    if hosts is None:
        return cluster
    nodes = cluster.get("node_dict")
    known = list(nodes) if isinstance(nodes, dict) else [node.get("mgmt_ip") for node in nodes or []]
    unknown = [host for host in hosts if host not in known]
    if unknown:
        raise ValueError(f"selected hosts not in the cluster file's node_dict: {', '.join(map(str, unknown))}")
    if not hosts:
        raise ValueError("no hosts selected from the cluster file's node_dict")
    wanted = set(hosts)
    narrowed = deepcopy(cluster)
    if isinstance(nodes, dict):
        narrowed["node_dict"] = {host: node for host, node in narrowed["node_dict"].items() if host in wanted}
        head = next(iter(narrowed["node_dict"]))
    else:
        narrowed["node_dict"] = [node for node in narrowed["node_dict"] if node.get("mgmt_ip") in wanted]
        head = narrowed["node_dict"][0]["mgmt_ip"]
    narrowed["head_node_dict"] = {**narrowed.get("head_node_dict", {}), "mgmt_ip": head}
    return narrowed


def build_orch(log, loaded, cluster):
    """An orchestrator for ``cluster`` and the ``LoadedSuiteConfig``'s orch inputs, with its config as ``suite_config``.

    ``orch.config`` keeps the merged cluster and orch inputs the orchestrators
    read; ``orch.suite_config`` is the validated suite config.
    """
    cfg = OrchestratorConfig.from_configs(cluster, loaded.orch_inputs)
    orch = OrchestratorFactory.create_orchestrator(log, cfg)
    orch.suite_config = loaded.config
    return orch
