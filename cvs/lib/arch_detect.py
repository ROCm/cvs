"""Cluster GPU-arch detection (CVS docker-mode P6).

v1 single-arch cluster invariant: every node in a single `cluster.json` MUST
report the same `gfx<X>` architecture. This module detects the arch on each
node by exec'ing `rocminfo` *inside* the running CVS-runner container, then:

  * Asserts that every node reports the same arch (single-arch invariant).
  * Optionally verifies against `runtime.expected_gfx_arch` from cluster.json.

Detection happens after `prepare_runtime` has started the container, so we
always have a known-good rocminfo available even on hosts without a host-side
ROCm install (the smci250 fresh-machine case).
"""

from __future__ import annotations

import re
from typing import Dict


# Match a top-level Agent "Name: gfx<X>" line in rocminfo output -- the one
# CPU agents have e.g. "Name: AMD EPYC ..." which won't match. We deliberately
# require the line to start with whitespace + "Name:" + whitespace + "gfx".
_GFX_LINE = re.compile(r"^\s+Name:\s+(gfx[0-9a-fA-F]+)\s*$", re.MULTILINE)


def _first_gfx_in_rocminfo(text: str) -> str | None:
    """Return the first `gfx<X>` arch in `rocminfo` output, or None."""
    m = _GFX_LINE.search(text or "")
    return m.group(1) if m else None


def detect_cluster_gfx_arch(phdl_host, runtime_cfg) -> str:
    """Detect the cluster's GPU arch via `docker exec rocminfo` on every node.

    Args:
        phdl_host: a Pssh handle whose wrapper is NoOp (i.e. commands run on
            the host, not inside the container -- because we *are* invoking
            `docker exec` from the host).
        runtime_cfg: a parsed RuntimeConfig (provides container_name and
            optional expected_gfx_arch).

    Returns:
        The detected arch string (e.g. "gfx942"). Identical across all nodes.

    Raises:
        RuntimeError on any of:
            * a node where rocminfo could not be parsed
            * arch mismatch between nodes (single-arch cluster invariant)
            * mismatch with runtime_cfg.expected_gfx_arch (if set)
    """
    # `docker exec` returns the stdout of rocminfo (banner-safe by construction
    # because we parse content -- the SSH banner is contaminating our stdout
    # too, but it never contains a top-level "Name: gfx<X>" line, so the regex
    # is robust against it).
    container = runtime_cfg.container_name
    cmd = (
        f"sudo docker exec {container} bash -lc "
        f"'/opt/rocm/bin/rocminfo 2>/dev/null'"
    )
    out_dict = phdl_host.exec(cmd, timeout=60)

    archs: Dict[str, str] = {}
    for node, out in out_dict.items():
        arch = _first_gfx_in_rocminfo(out)
        if arch is None:
            raise RuntimeError(
                f"could not detect gfx arch on {node}: rocminfo did not "
                f"contain a top-level 'Name: gfx<X>' line"
            )
        archs[node] = arch

    unique = set(archs.values())
    if len(unique) > 1:
        details = ", ".join(f"{n}={a}" for n, a in sorted(archs.items()))
        raise RuntimeError(
            f"single-arch cluster invariant violated: {details}. "
            f"v1 of CVS docker-mode assumes one cluster.json = one arch; "
            f"split this fleet into per-arch cluster files."
        )

    detected = unique.pop()

    expected = runtime_cfg.expected_gfx_arch
    if expected and detected != expected:
        first_mismatch = next(n for n, a in archs.items() if a != expected)
        raise RuntimeError(
            f"runtime.expected_gfx_arch={expected!r} but {first_mismatch}={detected!r}"
        )

    return detected
