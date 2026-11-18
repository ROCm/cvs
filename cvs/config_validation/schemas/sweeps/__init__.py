'''
Copyright 2026 Advanced Micro Devices, Inc.
All rights reserved.

Sweeps and runs: benchmark points named by cell keys such as
``ISL=1024,OSL=1024,TP=4,PP=1,CONC=16``.

Each ``NAME=value`` part of a key is an axis. A key names every axis its config
class declares, once each and in the declared order, with a positive integer
written without leading zeros. ``runs`` names the cells to run, in order: at
least one, each a ``sweeps`` key, none repeated. Checks on per-cell overrides
stay with each engine.
'''

import functools
import re
from collections import namedtuple
from typing import Annotated, Any, ClassVar, Dict, List, Tuple

from pydantic import BeforeValidator, field_validator, model_validator

from cvs.config_validation.schemas.base import StrictBlock, drop_comment_keys

Axis = namedtuple("Axis", ["name", "field"])

_AXIS_VALUE_RE = re.compile(r"[1-9]\d*")


@functools.lru_cache(maxsize=None)
def _cell_type(fields):
    return namedtuple("Cell", ("key",) + fields)


def _parse_cell_key(key, axes):
    names = [axis.name for axis in axes]

    def refused(problem):
        expected = ",".join(f"{name}=<n>" for name in names)
        return ValueError(f"cell key {key!r} {problem}; expected {expected}")

    values = {}
    for part in key.split(","):
        name, _, value = part.partition("=")
        if name not in names:
            raise refused(f"has unknown axis {name!r}")
        if name in values:
            raise refused(f"repeats axis {name}")
        if not _AXIS_VALUE_RE.fullmatch(value):
            raise refused(f"needs a positive integer for {name}")
        values[name] = int(value)
    missing = [name for name in names if name not in values]
    if missing:
        raise refused(f"is missing axis {', '.join(missing)}")
    if list(values) != names:
        raise refused("has its axes out of order")
    return _cell_type(tuple(axis.field for axis in axes))(key, *values.values())


CellOverrides = Annotated[Dict[str, Any], BeforeValidator(drop_comment_keys)]


class SweepsAndRuns(StrictBlock):
    """``sweeps`` maps each cell key to its per-cell overrides; ``runs`` names cells to run.

    A config class sets ``axes`` to the ``Axis`` tuple its keys use. ``_`` keys
    are comments at both levels of ``sweeps``.
    """

    axes: ClassVar[Tuple[Axis, ...]] = ()
    sweeps: Annotated[Dict[str, CellOverrides], BeforeValidator(drop_comment_keys)]
    runs: List[str]

    @field_validator("sweeps")
    @classmethod
    def _check_cell_keys(cls, sweeps):
        for key in sweeps:
            _parse_cell_key(key, cls.axes)
        return sweeps

    @field_validator("runs")
    @classmethod
    def _check_runs_listed_once(cls, runs):
        if not runs:
            raise ValueError("needs at least one sweeps key to run")
        repeated = sorted({key for key in runs if runs.count(key) > 1})
        if repeated:
            raise ValueError(f"cell keys listed more than once: {repeated}")
        return runs

    @model_validator(mode="after")
    def _check_runs_are_sweeps_keys(self):
        missing = [key for key in self.runs if key not in self.sweeps]
        if missing:
            raise ValueError(f"runs names cell keys that sweeps does not define: {missing}")
        return self

    def cells(self):
        """Each sweeps key, in order, as a named tuple of ``key`` and one field per axis."""
        return {key: _parse_cell_key(key, self.axes) for key in self.sweeps}
