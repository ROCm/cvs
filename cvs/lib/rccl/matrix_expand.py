"""
Pure RCCL matrix expansion: validated ``run`` slice + optional ``matrix`` → ordered resolved cases.

No I/O, SSH, or launcher coupling. Case-id rules match ``docs/rccl-phase1-data-model-spec.md`` §7.2.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .case_ids import _ensure_unique_case_id, _no_matrix_case_id, _slug


@dataclass(frozen=True)
class RcclMatrixExpansionInput:
    """Subset of validated RCCL config needed to expand cases (``run`` + optional ``matrix``)."""

    collectives: tuple[str, ...]
    datatype: str
    start_size: str
    end_size: str
    step_factor: str
    warmups: str
    iterations: str
    cycles: str
    matrix: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RcclResolvedCaseSpec:
    """One fully expanded benchmark case (inputs for execution + artifact naming)."""

    case_id: str
    name: str
    resolved: dict[str, Any]


def _resolved_payload(
    inp: RcclMatrixExpansionInput,
    collective: str,
    datatype: str,
    env: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "collective": collective,
        "datatype": datatype,
        "start_size": inp.start_size,
        "end_size": inp.end_size,
        "step_factor": inp.step_factor,
        "warmups": inp.warmups,
        "iterations": inp.iterations,
        "cycles": inp.cycles,
        "env": dict(env),
    }


def _display_name(collective: str, datatype: str, env: Mapping[str, str], variant_name: str | None = None) -> str:
    base = f"{collective} ({datatype})"
    if variant_name is not None:
        base = f"{variant_name}: {base}"
    if env:
        env_bits = ", ".join(f"{k}={env[k]}" for k in sorted(env))
        base = f"{base} | {env_bits}"
    return base


def _assign_unique_case_id(base: str, used: set[str], resolved: dict[str, Any]) -> str:
    return _ensure_unique_case_id(base, used, resolved)


def expand_rccl_no_matrix_cases(inp: RcclMatrixExpansionInput) -> list[RcclResolvedCaseSpec]:
    """Plain run: one case per ``inp.collectives`` entry, ignoring ``inp.matrix`` if set."""
    return _expand_plain(replace(inp, matrix=None))


def expand_rccl_matrix_cases(inp: RcclMatrixExpansionInput) -> list[RcclResolvedCaseSpec]:
    """Return deterministic ordered list of resolved cases."""
    m = inp.matrix
    if not m:
        return _expand_plain(inp)
    kind = m.get("kind")
    if kind == "variants":
        return _expand_variants(inp, m)
    if kind == "cartesian":
        return _expand_cartesian(inp, m)
    raise ValueError(f"rccl.matrix.kind must be 'variants' or 'cartesian', got {kind!r}")


def _expand_plain(inp: RcclMatrixExpansionInput) -> list[RcclResolvedCaseSpec]:
    used: set[str] = set()
    out: list[RcclResolvedCaseSpec] = []
    for i, collective in enumerate(inp.collectives):
        resolved = _resolved_payload(inp, collective, inp.datatype, {})
        cid = _assign_unique_case_id(_no_matrix_case_id(i, collective), used, resolved)
        out.append(
            RcclResolvedCaseSpec(
                case_id=cid,
                name=_display_name(collective, inp.datatype, {}),
                resolved=resolved,
            )
        )
    return out


def _expand_variants(inp: RcclMatrixExpansionInput, matrix: Mapping[str, Any]) -> list[RcclResolvedCaseSpec]:
    cases = matrix.get("cases")
    if not isinstance(cases, list) or len(cases) == 0:
        raise ValueError("rccl.matrix.cases must be a non-empty array for kind 'variants'")

    used: set[str] = set()
    out: list[RcclResolvedCaseSpec] = []

    for variant in cases:
        if not isinstance(variant, Mapping):
            raise ValueError("rccl.matrix.cases entries must be objects")
        name = variant.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("each variant must have a non-empty string name")
        raw_env = variant.get("env", {})
        if raw_env is None:
            raw_env = {}
        if not isinstance(raw_env, Mapping):
            raise ValueError("variant.env must be an object of string keys and string values when present")
        env_overlay: dict[str, str] = {str(k): str(v) for k, v in raw_env.items()}

        vslug = _slug(name)
        for i, collective in enumerate(inp.collectives):
            resolved = _resolved_payload(inp, collective, inp.datatype, env_overlay)
            base_id = f"v_{vslug}__c{i}_{_slug(collective)}"
            cid = _assign_unique_case_id(base_id, used, resolved)
            out.append(
                RcclResolvedCaseSpec(
                    case_id=cid,
                    name=_display_name(collective, inp.datatype, env_overlay, variant_name=name),
                    resolved=resolved,
                )
            )
    return out


def _parse_cartesian_dimensions(
    dimensions: Any,
) -> tuple[dict[str, list[str]], bool]:
    if not isinstance(dimensions, Mapping) or len(dimensions) == 0:
        raise ValueError("rccl.matrix.dimensions must be a non-empty object for kind 'cartesian'")

    parsed: dict[str, list[str]] = {}
    has_collective_dim = False
    for raw_key, raw_vals in dimensions.items():
        key = str(raw_key)
        if key == "collective":
            has_collective_dim = True
        elif key == "datatype":
            pass
        elif key.startswith("env."):
            suffix = key[4:]
            if not suffix or "=" in suffix:
                raise ValueError(f"invalid rccl.matrix dimension key {key!r}")
        else:
            raise ValueError(
                f"invalid rccl.matrix dimension key {key!r} (allowed: collective, datatype, env.<NAME>)"
            )

        if not isinstance(raw_vals, list) or len(raw_vals) == 0:
            raise ValueError(f"rccl.matrix.dimensions[{key!r}] must be a non-empty array")
        parsed[key] = [str(v) for v in raw_vals]

    return parsed, has_collective_dim


def _cartesian_tuples(dimensions: dict[str, list[str]]) -> list[dict[str, str]]:
    keys = sorted(dimensions.keys())
    rows: list[dict[str, str]] = []
    for combo in itertools.product(*(dimensions[k] for k in keys)):
        rows.append(dict(zip(keys, map(str, combo))))
    return rows


def _case_id_cartesian(tuple_map: dict[str, str], collective: str) -> str:
    keys = sorted(tuple_map.keys())
    parts = [f"{_slug(k)}={_slug(tuple_map[k])}" for k in keys]
    t = "__".join(parts)
    return f"x_{t}__{_slug(collective)}"


def _expand_cartesian(inp: RcclMatrixExpansionInput, matrix: Mapping[str, Any]) -> list[RcclResolvedCaseSpec]:
    dimensions, has_collective_dim = _parse_cartesian_dimensions(matrix.get("dimensions"))
    used: set[str] = set()
    out: list[RcclResolvedCaseSpec] = []

    if has_collective_dim:
        for tup in _cartesian_tuples(dimensions):
            collective = tup["collective"]
            datatype = tup.get("datatype", inp.datatype)
            env: dict[str, str] = {}
            for k, v in tup.items():
                if k.startswith("env."):
                    env[k[4:]] = v
            tuple_for_id = {k: v for k, v in tup.items()}
            resolved = _resolved_payload(inp, collective, datatype, env)
            base_id = _case_id_cartesian(tuple_for_id, collective)
            cid = _assign_unique_case_id(base_id, used, resolved)
            out.append(
                RcclResolvedCaseSpec(
                    case_id=cid,
                    name=_display_name(collective, datatype, env),
                    resolved=resolved,
                )
            )
        return out

    sub_dims = {k: v for k, v in dimensions.items() if k != "collective"}
    tuples = _cartesian_tuples(sub_dims)

    for collective in inp.collectives:
        for tup in tuples:
            datatype = tup.get("datatype", inp.datatype)
            env = {k[4:]: v for k, v in tup.items() if k.startswith("env.")}
            resolved = _resolved_payload(inp, collective, datatype, env)
            base_id = _case_id_cartesian(tup, collective)
            cid = _assign_unique_case_id(base_id, used, resolved)
            out.append(
                RcclResolvedCaseSpec(
                    case_id=cid,
                    name=_display_name(collective, datatype, env),
                    resolved=resolved,
                )
            )
    return out


def expansion_input_from_rccl_config(config: Any) -> RcclMatrixExpansionInput:
    """Build expansion input from :class:`cvs.lib.rccl.config.RcclConfig` (matrix from ``config_echo``)."""
    echo = getattr(config, "config_echo", None) or {}
    matrix = echo.get("matrix") if isinstance(echo, Mapping) else None
    if matrix == {}:
        matrix = None
    return RcclMatrixExpansionInput(
        collectives=tuple(getattr(config, "collectives", ()) or ()),
        datatype=str(getattr(config, "datatype", "")),
        start_size=str(getattr(config, "start_size", "")),
        end_size=str(getattr(config, "end_size", "")),
        step_factor=str(getattr(config, "step_factor", "")),
        warmups=str(getattr(config, "warmups", "")),
        iterations=str(getattr(config, "iterations", "")),
        cycles=str(getattr(config, "cycles", "")),
        matrix=matrix,
    )
