"""Token substitution for workload configs.

Recognized tokens:
  {shared_fs}, {models_dir}, {datasets_dir}, {artifacts_dir}, {images_dir}
  {run_id}, {user-id}
  {model.id}, {model.path}, {model.precision}
  {params.<key>}

Resolved recursively. Unknown token -> ValueError (fail-loud).
"""

from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"\{([a-zA-Z][a-zA-Z0-9_.\-]*)\}")
_MAX_DEPTH = 8


def substitute(value: Any, ctx: dict[str, Any]) -> Any:
    """Recursively substitute tokens in str/list/dict values using ctx."""
    if isinstance(value, str):
        return _substitute_str(value, ctx)
    if isinstance(value, list):
        return [substitute(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, ctx) for k, v in value.items()}
    return value


def _substitute_str(s: str, ctx: dict[str, Any]) -> str:
    out = s
    for _ in range(_MAX_DEPTH):
        m = _TOKEN_RE.search(out)
        if not m:
            return out
        # Replace every token in one pass; loop handles nesting.
        def _repl(match: re.Match[str]) -> str:
            tok = match.group(1)
            if tok not in ctx:
                raise ValueError(f"unknown token: {{{tok}}} in {s!r} (known: {sorted(ctx)})")
            v = ctx[tok]
            if not isinstance(v, (str, int, float)):
                raise ValueError(f"token {{{tok}}} resolved to non-scalar {type(v).__name__}")
            return str(v)
        out = _TOKEN_RE.sub(_repl, out)
    raise ValueError(f"substitution exceeded depth {_MAX_DEPTH}: {s!r} -> {out!r}")


def build_context(
    *,
    paths: dict[str, str],
    run_id: str,
    user_id: str,
    model_id: str | None = None,
    model_path: str | None = None,
    model_precision: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the substitution dict. paths are already resolved (no tokens)."""
    ctx: dict[str, Any] = dict(paths)
    ctx["run_id"] = run_id
    ctx["user-id"] = user_id
    if model_id is not None:
        ctx["model.id"] = model_id
    if model_path is not None:
        ctx["model.path"] = model_path
    if model_precision is not None:
        ctx["model.precision"] = model_precision
    for k, v in (params or {}).items():
        ctx[f"params.{k}"] = v
    return ctx


def resolve_paths_block(paths: dict[str, str]) -> dict[str, str]:
    """Resolve cross-references inside paths block (e.g. {shared_fs} in models_dir).

    Iterates to fixed point; unknown tokens raise.
    """
    resolved = dict(paths)
    for _ in range(_MAX_DEPTH):
        changed = False
        for k, v in list(resolved.items()):
            new = _substitute_str(v, resolved) if isinstance(v, str) else v
            if new != v:
                resolved[k] = new
                changed = True
        if not changed:
            return resolved
    raise ValueError(f"paths block has circular references: {paths}")
