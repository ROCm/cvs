#!/usr/bin/env python3
"""Apply key=value overrides to a JSON or YAML config file (writes to output path)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore


def _split_pairs(raw: str) -> list[tuple[str, str]]:
    """Split key=value pairs on ';' or ','. Prefer ';' if values may contain commas."""
    if not raw or not raw.strip():
        return []
    out: list[tuple[str, str]] = []
    for chunk in re.split(r"\s*[;,]\s*", raw.strip()):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        k, _, v = chunk.partition("=")
        k, v = k.strip(), v.strip()
        if k:
            out.append((k, v))
    return out


def _parse_value(v: str) -> Any:
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "null" or low == "none":
        return None
    try:
        if re.fullmatch(r"-?\d+", v):
            return int(v)
        if re.fullmatch(r"-?\d+\.\d+", v):
            return float(v)
    except ValueError:
        pass
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v


def _set_nested(obj: Any, dotted: str, value: Any) -> None:
    keys = [k for k in dotted.split(".") if k]
    if not keys:
        raise ValueError("empty key path")
    cur = obj
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _load(path: Path) -> tuple[Any, str]:
    text = path.read_text(encoding="utf-8")
    suf = path.suffix.lower()
    if suf in (".yaml", ".yml"):
        if yaml is None:
            print("ERROR: PyYAML is required for YAML configs. pip install pyyaml", file=sys.stderr)
            sys.exit(2)
        data = yaml.safe_load(text)
        return data, "yaml"
    data = json.loads(text)
    return data, "json"


def _dump(data: Any, fmt: str, path: Path) -> None:
    if fmt == "yaml":
        if yaml is None:
            print("ERROR: PyYAML is required for YAML configs.", file=sys.stderr)
            sys.exit(2)
        path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Source config (.json / .yaml)")
    p.add_argument("--output", required=True, help="Patched output file")
    p.add_argument(
        "--overrides",
        default="",
        help='key=value pairs separated by ";" or "," (nested keys use dots: run_config.timeout=30)',
    )
    args = p.parse_args(argv)

    src = Path(args.input)
    if not src.is_file():
        print(f"ERROR: input not found: {src}", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)

    pairs = _split_pairs(args.overrides)
    if not pairs:
        print(f"No overrides; copied {src} -> {out}")
        return 0

    data, fmt = _load(out)
    if not isinstance(data, dict):
        print("ERROR: top-level config must be a JSON object or YAML mapping", file=sys.stderr)
        return 1

    for key, val in pairs:
        _set_nested(data, key, _parse_value(val))

    _dump(data, fmt, out)
    print(f"Patched {len(pairs)} key(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
