from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from cvs.lib.utils_lib import resolve_test_config_placeholders
from cvs.schema.rccl_config import (
    RcclConfigFileRoot,
    format_rccl_config_validation_error,
    parse_rccl_thresholds_payload,
    threshold_collective_names,
)


@dataclass(frozen=True)
class RcclConfig:
    """Resolved RCCL CVS config: base run fields plus ``config_echo`` (includes optional ``matrix`` for expansion)."""

    required_nodes: int
    collectives: list[str]
    datatype: str
    num_ranks: int
    ranks_per_node: int
    env_script: str
    artifacts_output_dir: str
    artifacts_remote_work_dir: str
    artifacts_export_raw: bool
    start_size: str
    end_size: str
    step_factor: str
    warmups: str
    iterations: str
    cycles: str
    validation_profile: str
    thresholds: dict[str, Any]
    # Echo of validated rccl.* input (run, validation, artifacts, matrix) for run.json config section.
    config_echo: dict[str, Any] = field(default_factory=dict)

    @property
    def is_single_node(self) -> bool:
        return self.required_nodes == 1


def _normalize_threshold_map(collective_thresholds: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    if not collective_thresholds:
        return {}

    if "bus_bw" in collective_thresholds and isinstance(collective_thresholds["bus_bw"], dict):
        collective_thresholds = collective_thresholds["bus_bw"]

    normalized: dict[str, dict[str, float]] = {}
    for message_size, value in collective_thresholds.items():
        if isinstance(value, dict):
            bus_bw = value.get("bus_bw")
        else:
            bus_bw = value
        if bus_bw is None:
            continue
        normalized[str(message_size)] = {"bus_bw": float(bus_bw)}
    return normalized


def _resolve_threshold_payload_from_file(
    path: str,
    allowed_collectives: set[str],
) -> dict[str, Any]:
    with open(path) as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("rccl.validation: thresholds_file must contain a JSON object")
    if len(loaded) == 0:
        raise ValueError(
            "rccl.validation: thresholds_file must contain a non-empty thresholds object "
            "with the same shape as rccl.validation.thresholds"
        )
    try:
        parsed = parse_rccl_thresholds_payload(loaded)
    except ValidationError as exc:
        raise ValueError(format_rccl_config_validation_error(exc)) from exc
    for key in parsed:
        if not key or not str(key).strip():
            raise ValueError("rccl.validation: thresholds_file collective keys must be non-empty strings")
        if key not in allowed_collectives:
            raise ValueError(
                f"rccl.validation: threshold key {key!r} is not among collectives for this run "
                "(rccl.run.collectives and matrix expansion)"
            )
    return {k: v.model_dump(mode="json") for k, v in parsed.items()}


def load_rccl_config(config_file: str, cluster_dict: dict[str, Any]) -> RcclConfig:
    with open(config_file) as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError("RCCL config file must be a JSON object")
    extra_top = [k for k in raw if k != "rccl"]
    if extra_top:
        raise ValueError(
            f"RCCL config file must have exactly one top-level key 'rccl' (found extra: {sorted(extra_top)!r})"
        )
    if "rccl" not in raw or not isinstance(raw["rccl"], dict):
        raise ValueError("RCCL config file must contain a top-level 'rccl' object")

    rccl = resolve_test_config_placeholders(raw["rccl"], cluster_dict)
    rccl_for_schema = {k: v for k, v in rccl.items() if not str(k).startswith("_")}

    try:
        parsed = RcclConfigFileRoot.model_validate({"rccl": rccl_for_schema})
    except ValidationError as exc:
        raise ValueError(format_rccl_config_validation_error(exc)) from exc

    nested = parsed.rccl
    run = nested.run
    validation = nested.validation
    artifacts = nested.artifacts

    num_ranks = run.num_ranks
    ranks_per_node = run.ranks_per_node
    required_nodes = num_ranks // ranks_per_node
    collectives = list(run.collectives)
    threshold_collectives = threshold_collective_names(run, nested.matrix)

    profile = validation.profile
    thresholds: dict[str, Any] = {}
    if profile in {"thresholds", "strict"}:
        # RcclValidationInput already enforces exactly one non-empty source for these profiles.
        if validation.thresholds:
            thresholds = {k: v.model_dump(mode="json") for k, v in validation.thresholds.items()}
        else:
            thresholds = _resolve_threshold_payload_from_file(validation.thresholds_file, threshold_collectives)

    config_echo = nested.model_dump(mode="json")

    return RcclConfig(
        required_nodes=required_nodes,
        collectives=collectives,
        datatype=run.datatype,
        num_ranks=num_ranks,
        ranks_per_node=ranks_per_node,
        env_script=run.env_script,
        artifacts_output_dir=artifacts.output_dir,
        artifacts_remote_work_dir=artifacts.remote_work_dir,
        artifacts_export_raw=artifacts.export_raw,
        config_echo=config_echo,
        start_size=run.start_size,
        end_size=run.end_size,
        step_factor=run.step_factor,
        warmups=run.warmups,
        iterations=run.iterations,
        cycles=run.cycles,
        validation_profile=profile,
        thresholds=thresholds,
    )
