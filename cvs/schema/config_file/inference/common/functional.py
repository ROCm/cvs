"""Optional functional smoke gates for inference suites."""

from __future__ import annotations

from cvs.schema.base import _Forbid


class FunctionalConfig(_Forbid):
    api_smoke: bool = False
    health_check: bool = False
