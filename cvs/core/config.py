"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from cvs.core.errors import OrchestratorConfigError

# Re-export so callers can `from cvs.core.config import OrchestratorConfigError`.
__all__ = [
    "OrchestratorConfig",
    "OrchestratorConfigError",
    "load_config",
]


# Friend's intermediate cluster.json shape on ichristo/multi-orch-backend uses
# top-level `orchestrator` and `container` keys. That branch is not production;
# we explicitly reject the shape rather than write a translator that would
# become permanent maintenance for code nobody runs (working agreement #6).
_REJECTED_LEGACY_KEYS = ("orchestrator", "container")


@dataclass(frozen=True)
class OrchestratorConfig:
    """Backend-agnostic config produced by load_config.

    Each axis lives in its own sub-dict, validated by the corresponding
    backend class (Transport.* / Runtime.parse_config / Launcher.parse_config).
    Adding a new backend = one schema fragment; the central loader does not
    grow.

    Fields:
      transport - parsed cluster.json transport block (synthesized from flat
                  legacy keys if no `transport` overlay was present).
      runtime   - parsed cluster.json runtime block (default: hostshell).
      launchers - {name: launcher_config_dict} (default: {"mpi": {...}}).
      testsuite - parsed testsuite_config.json (the entire file, suite-keyed
                  -- e.g. {"rccl": {...}}). Per-test-module fixtures pull
                  their own suite key.
      raw       - merged raw dict, byte-for-byte. Escape hatch for util_lib
                  and any per-module fixture that wants the unparsed view.
                  Placeholder resolution is NOT applied here -- it stays in
                  the pytest fixtures.
    """

    transport: dict
    runtime: dict
    launchers: dict
    testsuite: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


# Top-level keys that came from the legacy main cluster.json shape and feed
# into the synthesized transport block.
_LEGACY_TRANSPORT_KEYS = (
    "node_dict",
    "head_node_dict",
    "username",
    "priv_key_file",
    "password",
    "env_vars",
)


def _read_json(path: Union[str, Path]) -> dict:
    p = Path(path)
    try:
        with p.open() as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise OrchestratorConfigError(f"config file not found: {p}") from e
    except json.JSONDecodeError as e:
        raise OrchestratorConfigError(f"invalid JSON in {p}: {e}") from e
    if not isinstance(data, dict):
        raise OrchestratorConfigError(
            f"top-level JSON in {p} must be an object, got {type(data).__name__}"
        )
    return data


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive merge: overlay wins for scalars and lists; nested dicts merge."""
    out: dict = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _validate_required(transport_block: dict, problems: list[str]) -> None:
    """Per-axis validation for the synthesized transport block. Aggregates
    problems into the caller's list rather than raising one at a time."""
    if not transport_block.get("node_dict"):
        problems.append(
            "transport.node_dict is required (legacy top-level node_dict missing)"
        )
    if not transport_block.get("username"):
        problems.append(
            "transport.username is required (legacy top-level username missing)"
        )
    if not transport_block.get("priv_key_file"):
        problems.append(
            "transport.priv_key_file is required (legacy top-level priv_key_file missing)"
        )


def _check_runtime(runtime_block: dict, problems: list[str]) -> None:
    """Run the runtime backend's own parse_config in dry-run mode so its
    validation errors get aggregated alongside transport errors."""
    from cvs.core.runtimes.factory import build_runtime

    try:
        build_runtime(runtime_block)
    except OrchestratorConfigError as e:
        problems.extend(e.problems or [str(e)])


def _check_launchers(launchers_block: dict, problems: list[str]) -> None:
    from cvs.core.launchers.factory import build_launchers

    try:
        build_launchers(launchers_block)
    except OrchestratorConfigError as e:
        problems.extend(e.problems or [str(e)])


def load_config(
    cluster_file: Union[str, Path],
    testsuite_file: Optional[Union[str, Path]] = None,
) -> OrchestratorConfig:
    """The one canonical loader.

    Schema (additive overlay):
      * Production main cluster.json has flat top-level keys (node_dict,
        username, priv_key_file, ...). These are the implicit baseline and
        must keep working unchanged -- they synthesize a {name: pssh}
        transport block + the default {name: hostshell} runtime + the
        default {mpi: {...}} launchers.
      * New cluster.json may include optional `transport` / `runtime` /
        `launchers` overlay blocks. transport is deep-merged into the
        synthesized one; runtime replaces the default; launchers deep-merge
        per launcher name.
      * Friend's intermediate `orchestrator` / `container` shape on
        ichristo/multi-orch-backend is rejected with a migration hint.

    Validation: dispatches to each backend's parse_config; aggregates all
    problems into one OrchestratorConfigError so users see every issue at
    once instead of one-at-a-time.
    """
    cluster_raw = _read_json(cluster_file)

    rejected_present = [k for k in _REJECTED_LEGACY_KEYS if k in cluster_raw]
    if rejected_present:
        raise OrchestratorConfigError(
            f"cluster.json has unsupported legacy key(s) {rejected_present}: "
            "the friend's ichristo/multi-orch-backend shape is not supported. "
            "Use the new schema: replace `orchestrator: 'container'` and the "
            "top-level `container: {...}` block with `runtime: {name: 'docker', "
            "config: {image: ..., container_name: ...}}`."
        )

    # 1. Synthesize the default transport block from legacy flat keys.
    synthesized_transport: dict[str, Any] = {"name": "pssh"}
    for k in _LEGACY_TRANSPORT_KEYS:
        if k in cluster_raw:
            synthesized_transport[k] = cluster_raw[k]

    # 2. Apply the optional `transport` overlay (block keys win).
    transport_overlay = cluster_raw.get("transport") or {}
    if not isinstance(transport_overlay, dict):
        raise OrchestratorConfigError(
            f"`transport` must be a dict, got {type(transport_overlay).__name__}"
        )
    transport_block = _deep_merge(synthesized_transport, transport_overlay)

    # 3. Runtime: default hostshell, replaced wholesale by overlay.
    runtime_block: dict[str, Any] = cluster_raw.get("runtime") or {"name": "hostshell"}
    if not isinstance(runtime_block, dict):
        raise OrchestratorConfigError(
            f"`runtime` must be a dict, got {type(runtime_block).__name__}"
        )

    # 4. Launchers: default mpi @ /opt/openmpi, deep-merged per launcher name.
    default_launchers: dict[str, dict] = {"mpi": {"install_dir": "/opt/openmpi"}}
    launchers_overlay = cluster_raw.get("launchers") or {}
    if not isinstance(launchers_overlay, dict):
        raise OrchestratorConfigError(
            f"`launchers` must be a dict, got {type(launchers_overlay).__name__}"
        )
    launchers_block = _deep_merge(default_launchers, launchers_overlay)

    # 5. testsuite_config.json: passthrough, suite-keyed.
    testsuite: dict = {}
    if testsuite_file is not None:
        testsuite = _read_json(testsuite_file)

    # 6. Aggregate validation across all axes -- one exception, all problems.
    problems: list[str] = []
    _validate_required(transport_block, problems)
    _check_runtime(runtime_block, problems)
    _check_launchers(launchers_block, problems)
    if problems:
        raise OrchestratorConfigError("configuration has problems", problems=problems)

    raw_merged = dict(cluster_raw)
    if testsuite:
        raw_merged["__testsuite__"] = testsuite

    return OrchestratorConfig(
        transport=transport_block,
        runtime=runtime_block,
        launchers=launchers_block,
        testsuite=testsuite,
        raw=raw_merged,
    )
