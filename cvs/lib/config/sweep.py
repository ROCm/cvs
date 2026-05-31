"""
Copyright 2025 Advanced Micro Devices, Inc.
All rights reserved. This notice is intended as a precaution against inadvertent publication and does not imply publication or any waiver of confidentiality.
The year included in the foregoing notice is the year of creation of the work.
All code contained here is Property of Advanced Micro Devices, Inc.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict


class SweepCell(BaseModel):
    """One expanded point of a sweep: a concrete set of params plus a stable ID.

    The ``id`` lowers directly to a pytest parametrize ID, so ``pytest -k`` and
    ``--collect-only`` line up with what the binder and manifests report.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    params: Dict[str, Any]


class SweepParams(BaseModel):
    """Base for per-framework sweep axes.

    Subclasses declare typed axes. Sweep semantics (matching
    ``pytest.mark.parametrize``):

    - A **scalar list** (``concurrency: [16, 32, 64]``) is one cartesian axis.
    - A **list of objects** (``sequence_combinations: [{isl, osl, name}]``) is a
      *paired* axis: the fields inside each object co-vary as a single option;
      an optional ``name`` becomes that option's ID token.
    - Axes cross cartesian-style against each other.
    - A bare scalar is a fixed param applied to every cell (not an axis).
    """

    model_config = ConfigDict(extra="forbid")

    def expand(self) -> List[SweepCell]:
        return expand_sweep(self)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool))


def _axis_from_value(key: str, value: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (options, tokens) for one sweep key.

    Each option is a dict merged into a cell's params; the matching token is a
    fragment of the cell ID.
    """
    options: List[Dict[str, Any]] = []
    tokens: List[str] = []
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        # Paired axis: each object is one bundled option.
        for idx, obj in enumerate(value):
            merged = {k: v for k, v in obj.items() if k != "name"}
            options.append(merged)
            # ``name`` may be absent OR present-but-None (pydantic dumps an
            # unset Optional as null), so a plain ``.get(key, default)`` would
            # yield the literal token "None". Fall back to a positional token.
            name = obj.get("name")
            tokens.append(str(name) if name not in (None, "") else f"{key}{idx}")
    elif isinstance(value, list) and value:
        # Scalar cartesian axis.
        for scalar in value:
            options.append({key: scalar})
            tokens.append(f"{key}{scalar}")
    return options, tokens


def expand_sweep(sweep: Any) -> List[SweepCell]:
    """Expand a sweep (a :class:`SweepParams` or plain dict) into cells."""
    raw: Dict[str, Any]
    if isinstance(sweep, SweepParams):
        raw = sweep.model_dump(mode="json")
    elif isinstance(sweep, dict):
        raw = dict(sweep)
    elif sweep is None:
        raw = {}
    else:
        raise TypeError(f"cannot expand sweep of type {type(sweep)!r}")

    fixed: Dict[str, Any] = {}
    axes: List[Tuple[List[Dict[str, Any]], List[str]]] = []

    # Sort keys for deterministic axis ordering -> deterministic cell IDs.
    for key in sorted(raw.keys()):
        value = raw[key]
        if value is None:
            continue
        if _is_scalar(value):
            fixed[key] = value
            continue
        options, tokens = _axis_from_value(key, value)
        if options:
            axes.append((options, tokens))

    if not axes:
        return [SweepCell(id="default", params=dict(fixed))]

    option_lists = [opts for opts, _ in axes]
    token_lists = [toks for _, toks in axes]

    cells: List[SweepCell] = []
    for combo_idx in itertools.product(*[range(len(opts)) for opts in option_lists]):
        params: Dict[str, Any] = dict(fixed)
        id_tokens: List[str] = []
        for axis_pos, opt_idx in enumerate(combo_idx):
            params.update(option_lists[axis_pos][opt_idx])
            id_tokens.append(token_lists[axis_pos][opt_idx])
        cells.append(SweepCell(id="-".join(id_tokens), params=params))
    ids = [c.id for c in cells]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(
            f"sweep expansion produced duplicate cell IDs {dupes}; cell IDs must be "
            f"unique (check for repeated scalar values or duplicate sequence names)"
        )
    return cells
