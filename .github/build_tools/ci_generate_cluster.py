#!/usr/bin/env python3
"""Generate cluster JSON for CI using the same template as `cvs generate cluster_json`."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path


def get_mgmt_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("1.1.1.1", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hosts",
        required=True,
        help="Comma-separated node IPs/hostnames (use CLUSTER_N_IP or CLUSTER_4N_IPS)",
    )
    p.add_argument("--username", required=True, help="SSH username for cluster nodes")
    p.add_argument("--key-file", required=True, help="Path to SSH private key")
    p.add_argument(
        "--output",
        default="ci-work/cluster.generated.json",
        help="Output cluster JSON path",
    )
    p.add_argument(
        "--head-node",
        default="",
        help="Head/mgmt IP for head_node_dict.mgmt_ip (default: auto-detect runner IP)",
    )
    args = p.parse_args(argv)

    hosts = ",".join(h.strip() for h in args.hosts.split(",") if h.strip())
    if not hosts:
        print("ERROR: --hosts is empty", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    head = (args.head_node or "").strip() or get_mgmt_ip()

    # Prefer `cvs` on PATH (after venv activate); fallback to python -m
    cmd = [
        "cvs",
        "generate",
        "cluster_json",
        "--hosts",
        hosts,
        "--output_json_file",
        str(out.resolve()),
        "--username",
        args.username,
        "--key_file",
        args.key_file,
        "--head_node",
        head,
    ]

    env = os.environ.copy()
    r = subprocess.run(cmd, cwd=os.getcwd(), env=env)
    if r.returncode != 0:
        print(f"ERROR: {' '.join(cmd)} failed with {r.returncode}", file=sys.stderr)
        return r.returncode

    print(f"Wrote cluster file: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
