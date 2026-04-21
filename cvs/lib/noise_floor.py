"""Noise floor probe: measure run-to-run variability via repeated TransferBench (CVS docker-mode P12).

Runs a fixed, fast TransferBench config N times inside the cvs-runner
container, parses GpuMem-to-GpuMem bandwidth (GB/s) from each run, and
computes the coefficient of variation:

    CV = stddev / mean

A clean, exclusively-owned node should report CV < ~1% for a steady-state
intra-node copy. A node with competing CPU work (stress-ng, kernel build,
etc.) shows CV >> 1% because each run sees different scheduling/PCIe noise.

Configurable via `runtime.noise_floor` block in cluster.json (all optional):

    "noise_floor": {
        "iterations":  5,         // number of repeats
        "threshold_cv": 0.01,     // PASS if CV <= this value
        "mode":        "warn"     // "warn" (default) or "strict"
    }
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

from cvs.lib import globals as cvs_globals

log = cvs_globals.log

# TransferBench probe: a single CPU0 -> GPU0 -> GPU0 transfer with 4 CUs.
# Crossing CPU↔GPU (rather than intra-GPU) makes the bandwidth sensitive to
# host memory pressure and PCIe contention -- the noise floor is meant to
# detect competing workloads, so the probe must touch the host memory path.
# TransferBench requires a config file path -- there is no inline size
# argument in v1.66+.
DEFAULT_TB_PROBE_LINE = "1 4 (C0->G0->G0)"

# Regex for parsing TransferBench result lines like:
#   "Test 1:    Executor: GPU 00 |       64.00 MB ...   123.45 GB/s"
_BW_LINE_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*GB/s\b")


def _stddev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def parse_transferbench_bw(stdout: str) -> Optional[float]:
    """Parse the FIRST GB/s value out of TransferBench stdout. None if absent."""
    if not stdout:
        return None
    m = _BW_LINE_RE.search(stdout)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def compute_cv(samples: List[float]) -> Optional[float]:
    """Coefficient of variation; None if mean is 0 or fewer than 2 samples."""
    samples = [s for s in samples if s is not None]
    if len(samples) < 2:
        return None
    mean = sum(samples) / len(samples)
    if mean <= 0:
        return None
    return _stddev(samples) / mean


def measure_noise_floor(
    phdl,
    iterations: int = 5,
    tb_probe_line: str = DEFAULT_TB_PROBE_LINE,
) -> Dict[str, dict]:
    """Run TransferBench `iterations` times per node; return per-node samples + cv."""
    log.info(
        "[P12] noise_floor probe: %d iterations of TransferBench '%s'",
        iterations,
        tb_probe_line,
    )
    per_node_samples: Dict[str, List[float]] = {}

    # Write the probe config once.
    phdl.exec(
        f"echo '{tb_probe_line}' > /tmp/cvs_noise_probe.cfg",
        timeout=15,
    )

    for i in range(iterations):
        out = phdl.exec(
            "cd /opt/INSTALL/TransferBench && "
            "./TransferBench /tmp/cvs_noise_probe.cfg 2>&1 | tail -40",
            timeout=120,
        )
        for node, raw in out.items():
            bw = parse_transferbench_bw(raw)
            per_node_samples.setdefault(node, []).append(bw)
            log.info("[P12]   iter=%d node=%s bw=%s GB/s", i + 1, node, bw)

    summary: Dict[str, dict] = {}
    for node, samples in per_node_samples.items():
        clean = [s for s in samples if s is not None]
        cv = compute_cv(samples)
        mean = (sum(clean) / len(clean)) if clean else None
        summary[node] = {
            "samples_gbps": clean,
            "missing": iterations - len(clean),
            "mean_gbps": mean,
            "cv": cv,
        }
    return summary


def evaluate_cv(summary: Dict[str, dict], threshold_cv: float) -> Dict[str, str]:
    """Per-node 'pass' / 'fail' / 'inconclusive' decision."""
    out: Dict[str, str] = {}
    for node, s in summary.items():
        cv = s.get("cv")
        if cv is None:
            out[node] = "inconclusive"
        elif cv <= threshold_cv:
            out[node] = "pass"
        else:
            out[node] = "fail"
    return out
