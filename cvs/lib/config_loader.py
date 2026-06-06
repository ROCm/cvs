"""Load + validate cluster, workload, threshold JSON.

Pydantic v2 for workload (catalog-driven Literal validation). Cluster is small
enough to dataclass-validate. Threshold is dict-of-dicts with kind enum.
"""

from __future__ import annotations

import getpass
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from cvs.lib.arch_detect import VALID_ARCHES
from cvs.lib.catalog import Catalog
from cvs.lib.errors import CatalogError, ConfigError
from cvs.lib.substitution import resolve_paths_block
from cvs.lib.verdict import VALID_KINDS


# ---------- cluster ----------

@dataclass
class ClusterConfig:
    username: str
    priv_key_file: str
    head_node: str
    node_dict: dict[str, dict]
    env_vars: dict[str, str]

    @property
    def hosts(self) -> list[str]:
        return list(self.node_dict.keys())


def load_cluster(path: Path) -> ClusterConfig:
    raw = _load_json(path, "cluster")
    user = raw.get("username", "{user-id}")
    if user == "{user-id}":
        user = getpass.getuser()
    head_dict = raw.get("head_node_dict") or {}
    head = head_dict.get("mgmt_ip")
    node_dict = raw.get("node_dict") or {}
    if not node_dict:
        raise ConfigError(f"{path}: node_dict is empty")
    if head not in node_dict:
        raise ConfigError(
            f"{path}: head_node_dict.mgmt_ip={head!r} not in node_dict keys {list(node_dict)}"
        )
    return ClusterConfig(
        username=user,
        priv_key_file=(raw.get("priv_key_file") or "").replace("{user-id}", user),
        head_node=head,
        node_dict=node_dict,
        env_vars=raw.get("env_vars") or {},
    )


# ---------- workload (Pydantic v2) ----------

class PathsBlock(BaseModel):
    shared_fs: str
    models_dir: str
    datasets_dir: str
    artifacts_dir: str
    images_dir: str


class ModelBlock(BaseModel):
    id: str
    remote: int = Field(ge=0, le=1)
    precision: str | None = None
    hf_token_env: str | None = None


class DatasetBlock(BaseModel):
    id: str
    remote: int = Field(ge=0, le=1)
    hf_token_env: str | None = None


class RegistryAuth(BaseModel):
    username_env: str
    password_env: str
    registry: str


class ImageBlock(BaseModel):
    tag: str
    remote: int = Field(ge=0, le=1)
    registry_auth: RegistryAuth | None = None


class TopologyRole(BaseModel):
    count: int = Field(ge=1)
    gpus_per_node: int = Field(ge=0)


class TopologyBlock(BaseModel):
    roles: dict[str, TopologyRole]


class RoleSpec(BaseModel):
    shm_size: str | None = None
    devices: list[str] = Field(default_factory=list)
    volumes: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    port: int | None = None
    health_path: str | None = None
    command: str
    seccomp_unconfined: bool = False
    extra_args: list[str] = Field(default_factory=list)
    # per-role image (disagg). If absent, top-level image used.
    image: ImageBlock | None = None


class WorkloadConfig(BaseModel):
    schema_version: Literal[1] = 1
    framework: str
    gpu_arch: str
    paths: PathsBlock

    @field_validator("gpu_arch")
    @classmethod
    def _arch_in_allowlist(cls, v: str) -> str:
        if v not in VALID_ARCHES:
            raise ValueError(f"gpu_arch={v!r} not in {sorted(VALID_ARCHES)}")
        return v
    model: ModelBlock
    dataset: DatasetBlock | None = None
    benchmarks: list[str] = Field(default_factory=list)
    benchmark_datasets_remote: int = Field(default=0, ge=0, le=1)
    image: ImageBlock | None = None
    topology: TopologyBlock
    roles: dict[str, RoleSpec]
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_roles_match_topology(self):
        topo_roles = set(self.topology.roles.keys())
        spec_roles = set(self.roles.keys())
        if topo_roles != spec_roles:
            missing = sorted(topo_roles - spec_roles)
            extra = sorted(spec_roles - topo_roles)
            raise ValueError(
                f"topology.roles {sorted(topo_roles)} != roles {sorted(spec_roles)} "
                f"(missing: {missing}, extra: {extra})"
            )
        # Either top-level image OR every role has its own image.
        if self.image is None:
            missing_img = [r for r, spec in self.roles.items() if spec.image is None]
            if missing_img:
                raise ValueError(
                    f"no top-level image; these roles also lack image: {missing_img}"
                )
        return self


def load_workload(
    path: Path,
    *,
    catalog: Catalog,
) -> WorkloadConfig:
    raw = _load_json(path, "workload")
    try:
        wl = WorkloadConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    # Catalog cross-validation (Pydantic Literal would be cleaner but we want
    # "did you mean..." messages — dynamic Literal in v2 is verbose for the
    # win it gives).
    if wl.model.id not in catalog.models:
        sugg = catalog.suggest("model", wl.model.id)
        hint = f" did you mean {sugg!r}?" if sugg else ""
        raise CatalogError(f"{path}: unknown model.id {wl.model.id!r}.{hint}")
    if wl.dataset is not None and wl.dataset.id not in catalog.datasets:
        sugg = catalog.suggest("dataset", wl.dataset.id)
        hint = f" did you mean {sugg!r}?" if sugg else ""
        raise CatalogError(f"{path}: unknown dataset.id {wl.dataset.id!r}.{hint}")
    # benchmark check deferred to whoever imports BENCHMARK_REGISTRY
    return wl


# ---------- thresholds ----------

def load_thresholds(path: Path) -> dict[str, dict]:
    raw = _load_json(path, "thresholds")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected top-level object")
    for name, spec in raw.items():
        if not isinstance(spec, dict) or "kind" not in spec:
            raise ConfigError(f"{path}: threshold {name!r} missing 'kind'")
        if spec["kind"] not in VALID_KINDS:
            raise ConfigError(
                f"{path}: threshold {name!r} kind={spec['kind']!r} "
                f"(valid: {VALID_KINDS})"
            )
        if spec["kind"] == "within":
            if "lo" not in spec or "hi" not in spec:
                raise ConfigError(f"{path}: 'within' threshold {name!r} needs lo + hi")
        else:
            if "value" not in spec:
                raise ConfigError(f"{path}: threshold {name!r} needs 'value'")
    return raw


# ---------- helpers ----------

def _load_json(path: Path, what: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"{what} file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{what} {p}: invalid JSON ({exc.msg} line {exc.lineno})") from exc


def resolve_paths(wl: WorkloadConfig, *, user_id: str | None = None) -> dict[str, str]:
    """Resolve cross-refs inside the paths block.

    paths may reference {user-id} (resolved from the cluster file), so
    we seed the resolver with that scalar before the fixed-point pass.
    """
    raw = wl.paths.model_dump()
    if user_id is not None:
        raw["user-id"] = user_id
    resolved = resolve_paths_block(raw)
    resolved.pop("user-id", None)
    return resolved
