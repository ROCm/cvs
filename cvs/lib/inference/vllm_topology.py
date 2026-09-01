'''Effective vLLM topology derived from the selected suite and orchestrator.'''

from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class EffectiveVllmTopology:
    mode: str
    hosts: tuple[str, ...]
    pipeline_parallel_size: int

    @property
    def nnodes(self) -> int:
        return len(self.hosts)

    @property
    def target_groups(self) -> tuple[tuple[str, ...], ...]:
        return (self.hosts,)


def scope_vllm_cluster(mode, cluster):
    """Return the cluster used by the selected suite.

    ``vllm_single`` is intentionally first-host-only. Distributed runs retain
    the complete cluster and use their normal single-host fallback when needed.
    """
    if mode != "single":
        return cluster

    scoped = deepcopy(cluster)
    nodes = scoped.get("node_dict")
    if isinstance(nodes, dict) and nodes:
        host, node = next(iter(nodes.items()))
        scoped["node_dict"] = {host: node}
    elif isinstance(nodes, list) and nodes:
        node = nodes[0]
        host = node.get("mgmt_ip")
        scoped["node_dict"] = [node]
    else:
        raise ValueError("vllm_single requires at least one cluster node")
    if not host:
        raise ValueError("the first vllm_single cluster node has no management address")
    scoped["head_node_dict"] = {**scoped.get("head_node_dict", {}), "mgmt_ip": host}
    return scoped


def resolve_vllm_topology(mode, variant, hosts) -> EffectiveVllmTopology:
    hosts = tuple(hosts)
    if not hosts:
        raise ValueError("vLLM requires at least one orchestrator host")

    if mode == "single":
        if int(variant.params.pipeline_parallel_size) > 1:
            raise ValueError("vllm_single requires pipeline_parallel_size=1")
        if len(hosts) != 1:
            raise ValueError("vllm_single orchestrator must be scoped to its first host")
        return EffectiveVllmTopology("single", hosts, 1)
    if mode != "distributed":
        raise ValueError(f"unknown vLLM mode: {mode!r}")
    if len(hosts) == 1:
        return EffectiveVllmTopology("single", hosts, 1)
    if len(hosts) != 2:
        raise ValueError(
            "vllm_distributed currently supports exactly two cluster hosts; "
            "add an explicit N-host recipe and thresholds before using a larger cluster"
        )

    is_ray = variant.roles.server.serve_args.get("distributed-executor-backend") == "ray"
    effective_pp = int(variant.params.pipeline_parallel_size)
    if effective_pp == 1 and not is_ray:
        raise ValueError("vllm_distributed requires pipeline_parallel_size>1 unless distributed-executor-backend=ray")
    if not variant.roles.server.ib_netdev:
        raise ValueError("vllm_distributed requires roles.server.ib_netdev on multi-host clusters")
    return EffectiveVllmTopology("distributed", hosts, effective_pp)


def build_vllm_targets(mode, variant, hosts):
    topology = resolve_vllm_topology(mode, variant, hosts)
    return topology.target_groups, topology.pipeline_parallel_size
