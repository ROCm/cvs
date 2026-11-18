'''Effective vLLM topology derived from the selected suite and orchestrator.'''

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


def _socket_netdev_configured(variant):
    container = getattr(variant, "container", None)
    env = getattr(container, "env", {}) if container is not None else {}
    return bool(env.get("NCCL_SOCKET_IFNAME"))


def resolve_vllm_topology(mode, variant, hosts) -> EffectiveVllmTopology:
    hosts = tuple(hosts)
    if not hosts:
        raise ValueError("vLLM requires at least one orchestrator host")

    if mode == "single":
        if len(hosts) != 1:
            raise ValueError("vllm_single orchestrator must be scoped to its first host")
        return EffectiveVllmTopology("single", hosts, variant.server_params.pp)
    if mode != "distributed":
        raise ValueError(f"unknown vLLM mode: {mode!r}")
    if len(hosts) == 1:
        raise ValueError("vllm_distributed needs server_params.nnodes hosts of 2 or more; the orchestrator has one")

    is_ray = variant.vllm.distributed_executor_backend == "ray"
    effective_pp = variant.server_params.pp
    if effective_pp == 1 and not is_ray:
        raise ValueError("vllm_distributed requires server_params.pp>1 unless distributed_executor_backend=ray")
    if not _socket_netdev_configured(variant):
        raise ValueError("vllm_distributed requires container.env.NCCL_SOCKET_IFNAME on multi-host clusters")
    return EffectiveVllmTopology("distributed", hosts, effective_pp)


def build_vllm_targets(mode, variant, hosts):
    topology = resolve_vllm_topology(mode, variant, hosts)
    return topology.target_groups, topology.pipeline_parallel_size
