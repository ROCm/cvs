#!/usr/bin/env python3
"""Print absolute log scan folder from a CVS config (JSON/YAML). Default: $PWD/logs."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


def main(argv: list[str]) -> int:
    p = os.environ.get("PATCHED_CONFIG_FILE", "")
    if not p or not Path(p).is_file():
        print(Path.cwd() / "logs")
        return 0
    path = Path(p)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            print(Path.cwd() / "logs")
            return 0
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        print(Path.cwd() / "logs")
        return 0
    rc = data.get("run_config") or {}
    if not isinstance(rc, dict):
        print(Path.cwd() / "logs")
        return 0
    lf = rc.get("runner_log_folder") or "logs"
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "runner"
    lf = str(lf).replace("{headnode-user-id}", user)
    out = Path(lf).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    print(out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
