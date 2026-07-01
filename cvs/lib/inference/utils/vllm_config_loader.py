'''
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved.

Unified config schema for the vllm suite (single-node and distributed).

Replaces inferencing_config_loader.py (single-node) and
vllm_distributed_config_loader.py (distributed) with a single schema.
Distributed params (pipeline_parallel_size, master_addr, master_port,
nnodes) default to single-node values so the same VariantConfig works for
both topologies.

cell_key format:
  Single-node (pp=1): ISL=<isl>,OSL=<osl>,TP=<tp>,CONC=<concurrency>
  Distributed (pp>1): ISL=<isl>,OSL=<osl>,TP=<tp>,PP=<pp>,CONC=<concurrency>

IB device config:
  roles.server.ib_hca_devices: list[str] | "auto" | absent
      If absent or "auto", use everything ibv_devinfo -l reports.
      If an explicit list, validate at preflight (test_discover_topology).
  roles.server.ib_netdev: str (required for distributed runs)
      Linux network interface name for NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME.
      Not derivable from HCA names. Operator sets it explicitly.
      Optional for single-node (NCCL socket selection not critical).
'''

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Literal

from cvs.lib.inference.utils.vllm_parsing import GATED_METRICS
from cvs.lib.utils.config_loader import substitute_config


class _Forbid(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Allow(BaseModel):
    model_config = ConfigDict(extra="allow")


# ---------- sub-models ----------


class ContainerConfig(_Allow):
    lifetime: str = "per_run"
    name: str = ""
    image: str = ""


class Paths(_Forbid):
    shared_fs: str
    models_dir: str
    log_dir: str
    hf_token_file: str


class ModelSpec(_Forbid):
    id: str
    remote: Literal[0, 1]


_VLLM_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}


class RoleServer(_Forbid):
    serve_args: Dict[str, Any] = {}
    env: Dict[str, str] = {}
    # IB HCA devices for NCCL_IB_HCA.
    # absent or "auto" -> use whatever ibv_devinfo -l reports.
    # explicit list -> validated at preflight against ibv_devinfo output.
    ib_hca_devices: Union[Literal["auto"], List[str], None] = None
    # Linux netdev for NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME.
    # Required when nnodes > 1. No "auto" — not reliably derivable from HCA names.
    ib_netdev: Optional[str] = None

    @field_validator("serve_args", mode="after")
    @classmethod
    def _check_log_level(cls, v):
        level = v.get("log-level")
        if level is not None and level not in _VLLM_LOG_LEVELS:
            raise ValueError(f"serve_args.log-level must be one of {sorted(_VLLM_LOG_LEVELS)}, got: {level!r}")
        return v


class Roles(_Forbid):
    server: RoleServer = Field(default_factory=RoleServer)


class GoodputSlo(_Forbid):
    ttft_ms: float
    tpot_ms: float
    e2el_ms: float


class SeqCombo(_Forbid):
    name: str
    isl: str
    osl: str
    goodput_slo: Optional[GoodputSlo] = None


class Run(_Forbid):
    combo: str
    concurrency: int


def validate_sweep_selector(combo_names, run_combo_refs):
    """Single home for the sweep-selector rule: names unique, every run.combo known.

    Called both at load time (via Sweep model_validator) and at collection time
    (pytest_generate_tests reads raw JSON before load_variant runs) so the two
    paths cannot drift.
    """
    counts = Counter(combo_names)
    dupes = sorted(name for name, count in counts.items() if count > 1)
    if dupes:
        raise ValueError(f"duplicate sequence_combination names: {dupes}")
    known = set(counts)
    unknown = sorted({r for r in run_combo_refs if r not in known})
    if unknown:
        raise ValueError(f"run.combo names no sequence_combination: {unknown} (known: {sorted(known)})")


class Sweep(_Forbid):
    sequence_combinations: List[SeqCombo]
    runs: List[Run]

    @model_validator(mode="after")
    def _check_runs_reference_known_combos(self):
        validate_sweep_selector(
            [c.name for c in self.sequence_combinations],
            [r.combo for r in self.runs],
        )
        return self


class Params(_Forbid):
    backend: str = "vllm"
    base_url: str = "http://0.0.0.0"
    port_no: str = "8888"
    dataset_name: str = "random"
    burstiness: str = "1.0"
    seed: str = "0"
    request_rate: str = "inf"
    random_range_ratio: str = "0.8"
    random_prefix_len: str = "0"
    tensor_parallelism: str = "8"
    # Distributed params. Defaults encode single-node (no PP, one node, localhost).
    pipeline_parallel_size: str = "1"
    master_addr: str = "localhost"
    master_port: str = "29501"
    nnodes: str = "1"
    tokenizer_mode: str = "auto"
    percentile_metrics: str = "ttft,tpot,itl,e2el"
    metric_percentiles: str = "50,90,95,99"
    num_prompts: str = "3200"
    client_poll_count: str = "20"


class VariantConfig(_Forbid):
    """Unified typed config for both single-node and distributed vllm runs.

    Standalone (does not extend BaseVariantConfig) so it can be constructed
    without the threshold_json field the base requires — absent from unit-test
    fixtures. Production configs always supply it via substitute_config.

    ``container`` is optional at model-level: unit-test fixtures omit it.
    The conftest ``orch`` fixture accesses ``variant_config.container.model_dump()``.
    """

    schema_version: Literal[1]
    framework: Literal["vllm"]
    gpu_arch: str
    enforce_thresholds: bool = True
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    paths: Paths
    model: ModelSpec
    roles: Roles = Field(default_factory=Roles)
    params: Params = Field(default_factory=Params)
    sweep: Sweep
    thresholds: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_distributed_consistency(self):
        nn = int(self.params.nnodes)
        pp = int(self.params.pipeline_parallel_size)
        if nn > 1 and pp == 1:
            raise ValueError(f"nnodes={nn} > 1 requires pipeline_parallel_size > 1 (got pp={pp})")
        if pp > 1 and nn == 1:
            raise ValueError(f"pipeline_parallel_size={pp} > 1 requires nnodes > 1 (got nnodes={nn})")
        if nn > 1 and not self.roles.server.ib_netdev:
            raise ValueError(
                "ib_netdev is required in roles.server when nnodes > 1. "
                "Set it to the Linux network interface name for NCCL_SOCKET_IFNAME "
                "(e.g. \"ens51f1np1\"). Cannot be auto-derived from HCA names."
            )
        return self

    @model_validator(mode="after")
    def _check_remote_not_implemented(self):
        if self.model.remote == 1:
            raise NotImplementedError("model.remote=1 (remote model download) is not implemented.")
        return self

    def cell_key(self, isl, osl, concurrency):
        """Canonical threshold key for one sweep cell.

        Emits PP= segment only for distributed runs (pp > 1), preserving
        backward-compatible single-node keys (no PP= segment).

        Single-node: ISL=<isl>,OSL=<osl>,TP=<tp>,CONC=<concurrency>
        Distributed:  ISL=<isl>,OSL=<osl>,TP=<tp>,PP=<pp>,CONC=<concurrency>
        """
        base = f"ISL={isl},OSL={osl},TP={self.params.tensor_parallelism},"
        if int(self.params.pipeline_parallel_size) > 1:
            base += f"PP={self.params.pipeline_parallel_size},"
        return base + f"CONC={concurrency}"

    def expected_cells(self):
        by_name = {c.name: c for c in self.sweep.sequence_combinations}
        return [self.cell_key(by_name[r.combo].isl, by_name[r.combo].osl, r.concurrency) for r in self.sweep.runs]

    @model_validator(mode="after")
    def _check_thresholds_cover_sweep(self):
        expected = set(self.expected_cells())
        present = set(self.thresholds.keys())
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        problems = []
        if missing:
            problems.append(f"sweep cells with no threshold entry: {missing}")
        if extra:
            problems.append(f"threshold keys matching no sweep cell (typo?): {extra}")
        gated_keys = [f"client.{m}" for m in sorted(GATED_METRICS)]
        gated_gaps = {}
        for cell in sorted(expected & present):
            specs = self.thresholds.get(cell) or {}
            absent = [k for k in gated_keys if k not in specs]
            if absent:
                gated_gaps[cell] = absent
        if gated_gaps:
            problems.append(f"cells missing gated-metric specs: {gated_gaps}")
        if problems:
            msg = "threshold.json does not match the sweep matrix; " + "; ".join(problems)
            if self.enforce_thresholds:
                raise ValueError(msg)
            warnings.warn(f"{msg} (enforce_thresholds=false -> record-only)", stacklevel=2)
        return self


# ---------- public API ----------


def load_variant(config_path, cluster_dict):
    """Load and validate a vllm variant config + its sibling threshold file.

    Strips fields unknown to VariantConfig before construction so production
    configs (which carry extra keys like threshold_json) and unit-test fixtures
    (which omit optional fields) are handled identically.
    """
    raw, thresholds = substitute_config(config_path, cluster_dict)
    known = {k: v for k, v in raw.items() if k in VariantConfig.model_fields}
    known["thresholds"] = thresholds
    return VariantConfig(**known)
