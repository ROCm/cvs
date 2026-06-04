"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cvs.lib.config.thresholds import ThresholdUnion


def _coerce_token(value: object) -> str:
    """Fail-closed str coercion for a ``ContainerSpec`` token.

    A wrong-typed value -- ``None`` (an empty YAML mapping value), a ``bool``,
    a ``float``, a nested list/dict -- must error at load instead of silently
    becoming ``"None"`` / ``"True"`` / ``"8888.0"`` and emitting a malformed
    ``docker run`` arg. Only ``str``/``int``/``Path`` stringify: ``bool`` is
    rejected even though it subclasses ``int``, and ``float`` is rejected
    because the only tokens here are ports, device IDs, paths and env values --
    a ``float`` ``8888.0`` is a mistyped port (``-p 8888.0:...``), never intent.
    """
    if isinstance(value, bool):
        raise ValueError(f"container spec value must be a string, not a bool: {value!r}")
    if isinstance(value, (str, int, Path)):
        return str(value)
    raise ValueError(f"container spec value must be str/int/path, got {type(value).__name__}: {value!r}")


class Role(BaseModel):
    """A workload role's hardware requirement, satisfied by the binder (W5).

    Roles live in the *config* (the workload), not the cluster file (the
    hardware pool). ``selector`` is a label query against ``node.labels``.
    """

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    gpus_per_node: int = Field(default=0, ge=0)
    selector: Optional[str] = None


class Topology(BaseModel):
    """Workload-role placement requirements.

    Two YAML shapes accepted; both load to the same in-memory ``Topology``:

    1. Full form (multi-role / disagg / training): explicit
       ``roles: {name: {count, gpus_per_node, selector}}``.
    2. Shorthand (single-role inference): flat ``{nnodes, gpus_per_node,
       selector}`` expands to ``roles: {server: {count: nnodes, ...}}``.

    Shorthand keeps single-node vLLM configs out of 5-line topology blocks
    while leaving the explicit ``roles`` form mandatory for any workload that
    has more than one role (where implicit single-role would be wrong).
    Mixing the two -- providing ``roles`` AND any shorthand key -- is rejected
    at load: it's almost always a typo, never useful.
    """

    model_config = ConfigDict(extra="forbid")

    # A workload with zero roles has nothing for the binder to place; reject it
    # at load rather than producing an unschedulable config.
    roles: Dict[str, Role] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _expand_shorthand(cls, value):
        if not isinstance(value, dict):
            return value
        shorthand_keys = {"nnodes", "gpus_per_node", "selector"}
        present = shorthand_keys.intersection(value)
        if not present:
            return value
        if "roles" in value:
            raise ValueError(f"topology cannot mix explicit 'roles' with shorthand keys {sorted(present)}")
        nnodes = value.pop("nnodes", 1)
        gpus_per_node = value.pop("gpus_per_node", 0)
        selector = value.pop("selector", None)
        value["roles"] = {
            "server": {
                "count": nnodes,
                "gpus_per_node": gpus_per_node,
                "selector": selector,
            }
        }
        return value


class ContainerSpec(BaseModel):
    """Container runtime surface (A3): how the workload image is launched.

    The fields map 1:1 onto :class:`cvs.lib.runtime.ContainerHandle` kwargs.
    Every value that reaches the handle is a ``str`` -- dict keys and values
    included -- because ``ContainerHandle.build_run_command`` quotes each token
    with ``shlex.quote``, which raises ``TypeError`` on a non-``str`` rather than
    coercing. The spec therefore owns the conversion: an ``int`` port or a
    ``pathlib.Path`` host-path is stringified here (the ``mode="before"``
    validators below), so a wrong-typed value can never silently produce a
    malformed ``docker run``.

    Site-specific host paths (e.g. the model-weight cache feeding ``volumes``)
    are resolved from the cluster file at bind time (G4/G5), not baked into
    per-framework params.

    ``env`` carries container environment, including credentials such as the HF
    token on closed/internal clusters. It is deliberately *excluded* from
    ``workload_hash`` (see :meth:`BaseTestConfig.workload_hash`) so workload
    identity does not depend on the token value.
    """

    model_config = ConfigDict(extra="forbid")

    volumes: Dict[str, str] = Field(default_factory=dict)
    devices: List[str] = Field(default_factory=list)
    ports: Dict[str, str] = Field(default_factory=dict)
    env: Dict[str, str] = Field(default_factory=dict)
    shm_size: Optional[str] = None
    ipc: Optional[str] = None
    network: Optional[str] = None

    @field_validator("volumes", "ports", "env", mode="before")
    @classmethod
    def _stringify_mapping(cls, value):
        if isinstance(value, dict):
            return {_coerce_token(k): _coerce_token(v) for k, v in value.items()}
        return value

    @field_validator("devices", mode="before")
    @classmethod
    def _stringify_list(cls, value):
        if isinstance(value, list):
            return [_coerce_token(v) for v in value]
        return value

    def to_handle_kwargs(self) -> Dict[str, object]:
        """Return the ``ContainerHandle`` kwargs this spec populates.

        Only set keys are returned for the optional scalars, so handle defaults
        apply when the spec leaves them unset.
        """
        kwargs: Dict[str, object] = {
            "volumes": dict(self.volumes),
            "devices": list(self.devices),
            "ports": dict(self.ports),
            "env": dict(self.env),
        }
        if self.shm_size is not None:
            kwargs["shm_size"] = self.shm_size
        if self.ipc is not None:
            kwargs["ipc"] = self.ipc
        if self.network is not None:
            kwargs["network"] = self.network
        return kwargs


def _stable_hash(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class BaseTestConfig(BaseModel):
    """Common, framework-agnostic config surface.

    ``extra="forbid"`` is the load-time typo gate: a misspelled key fails
    ``model_validate`` immediately instead of silently defaulting and surfacing
    20 minutes into a workload run. Concrete framework configs add typed
    ``params`` and ``sweep`` fields.

    No secret/redaction layer (W7 removed): container environment, including
    tokens, rides in ``container.env`` and is recorded as-is. The hashes below
    are plain deterministic JSON serializations.
    """

    model_config = ConfigDict(extra="forbid")

    # Subclasses narrow ``framework`` to a Literal used as the dispatch key.
    WORKLOAD_KIND: ClassVar[str] = "unknown"

    schema_version: str = "2"
    framework: str
    model: str
    cluster_ref: Optional[str] = None
    seed: int = 0
    knobs: Dict[str, str] = Field(default_factory=dict)
    container: ContainerSpec = Field(default_factory=ContainerSpec)
    topology: Topology
    thresholds: List[ThresholdUnion] = Field(default_factory=list)
    benchmarks: List[str] = Field(default_factory=list)

    @property
    def workload_kind(self) -> str:
        return self.WORKLOAD_KIND

    def _dump(self) -> dict:
        # Deterministic JSON-mode serialization used by the hashes below. No
        # secret handling (security removed): ``container.env`` is dumped as-is.
        return self.model_dump(mode="json")

    def workload_hash(self) -> str:
        """Hash of workload-defining inputs (framework, model, seed, knobs, params, topology).

        ``seed`` is included: a different seed produces different numerical
        results, so two otherwise-identical configs are *not* the same workload
        for result-comparison/caching purposes and must hash differently.

        Deliberately excludes ``container`` (volumes/devices/env) so workload
        identity is stable across sites and independent of the token value.
        """
        dump = self._dump()
        payload = {
            "framework": dump.get("framework"),
            "model": dump.get("model"),
            "seed": dump.get("seed"),
            "knobs": dump.get("knobs"),
            "params": dump.get("params"),
            "sweep": dump.get("sweep"),
            # Selector tokens on each role encode the GPU class (e.g. "mi300"),
            # so the GPU dimension survives in the hash without a separate
            # target_gpu field.
            "topology": dump.get("topology"),
        }
        return _stable_hash(payload)

    def verification_hash(self) -> str:
        """Hash of verification inputs (thresholds + opted-in benchmarks)."""
        dump = self._dump()
        payload = {"thresholds": dump.get("thresholds"), "benchmarks": dump.get("benchmarks")}
        return _stable_hash(payload)

    def config_hash(self) -> str:
        """Hash of the full resolved config (the exact config, env included)."""
        return _stable_hash(self._dump())
