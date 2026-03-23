#!/usr/bin/env python3
"""Validate suite id vs node count and emit GITHUB_ENV lines for the CVS CI workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", required=True, help="Path to cvs-ci-suite-matrix.json")
    p.add_argument("--suite-id", required=True, help="Suite id from workflow_dispatch")
    p.add_argument("--node-count", type=int, choices=(1, 4), required=True)
    p.add_argument("--workspace", default=".", help="Repository root (GITHUB_WORKSPACE)")
    p.add_argument(
        "--config-override",
        default="",
        help="Optional path relative to repo root to use instead of the matrix default config",
    )
    args = p.parse_args(argv)

    ws = Path(args.workspace).resolve()
    matrix_path = Path(args.matrix)
    if not matrix_path.is_file():
        matrix_path = ws / matrix_path
    data = json.loads(matrix_path.read_text(encoding="utf-8"))
    suites = data.get("suites", [])
    found = None
    for s in suites:
        if s.get("id") == args.suite_id:
            found = s
            break
    if not found:
        print(f"ERROR: unknown suite id '{args.suite_id}'", file=sys.stderr)
        return 1

    mode = found.get("node_mode", "one")
    if mode == "one" and args.node_count != 1:
        print(
            f"ERROR: suite '{args.suite_id}' is for 1-node runs only (node_mode=one); got node_count={args.node_count}",
            file=sys.stderr,
        )
        return 1
    if mode == "four" and args.node_count != 4:
        print(
            f"ERROR: suite '{args.suite_id}' requires 4-node cluster (node_mode=four); got node_count={args.node_count}",
            file=sys.stderr,
        )
        return 1

    rel = found["default_config"].lstrip("/")
    cvs_test = found["cvs_test"]
    input_root = ws / "cvs" / "input"
    override = (args.config_override or "").strip()
    if override:
        src_config = (ws / override).resolve()
        if not src_config.is_file():
            print(f"ERROR: --config-override not found: {src_config}", file=sys.stderr)
            return 1
        try:
            rel = str(src_config.relative_to(input_root))
        except ValueError:
            rel = override
    else:
        src_config = input_root / rel
    if not src_config.is_file():
        print(f"ERROR: default config not found: {src_config}", file=sys.stderr)
        return 1

    out_dir = ws / "ci-work"
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = src_config.suffix.lower()
    if suf in (".yaml", ".yml"):
        patched = out_dir / "config.patched.yaml"
    else:
        patched = out_dir / "config.patched.json"

    cluster_file = out_dir / "cluster.generated.json"

    lines = [
        f"CVS_TEST={cvs_test}",
        f"SOURCE_CONFIG_REL={rel}",
        f"SOURCE_CONFIG_FILE={src_config.resolve()}",
        f"PATCHED_CONFIG_FILE={patched.resolve()}",
        f"CLUSTER_FILE={cluster_file.resolve()}",
        f"INPUT_ROOT={input_root.resolve()}",
    ]
    gh = os.environ.get("GITHUB_ENV")
    if gh:
        with open(gh, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
