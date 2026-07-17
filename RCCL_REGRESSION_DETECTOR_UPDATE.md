# CVS RCCL Regression Strategy (AIMVT-196)

This document describes how the CVS RCCL performance-regression pipeline detects
real RCCL performance regressions in CI **without** false positives, including the
design rationale, the algorithm, configuration, how to run it, and the evidence
that it is trustworthy.

- **Code (cvs)** — branch `aimvt-196-rccl-regression-robustness` (origin: `ROCm/cvs`)
- **Orchestration (cvs-sbatch)** — branch `aimvt-196-rccl-regression-robustness` (origin: `speriaswamy-amd/cvs-sbatch`)
- **Companion docs**: `RCCL_REGRESSION_FINDINGS.md` (a concrete candidate regression + bisection handoff)

---

## 1. Goal & guiding principles

Run in CI as a gate on RCCL changes and answer one question reliably:
**"Did this RCCL build get slower than a known-good reference?"** across message
sizes **1 KiB → 4 GiB**.

Priorities, in order:

1. **No false positives.** A flaky CI gate is worse than no gate — it erodes trust
   and gets ignored/disabled. Stability is the #1 requirement.
2. **Trustworthy detection of real regressions**, especially at large messages.
3. It is **acceptable to miss small regressions** (~1–2%), particularly for small,
   latency-bound messages with high run-to-run variance.

Everything below follows from these priorities.

---

## 2. Why not a static baseline?

The previous approach compared measured bus bandwidth against **hand-maintained
expected numbers** (e.g. `330`, `350` GB/s) in config. Problems:

- CVS has **no way to compute a baseline**, so the numbers were guesses and went stale.
- Small/mid messages are **latency-bound** and noisy; a fixed threshold either
  fires on noise (false positives) or is so loose it hides real regressions.
- The comparison code also had a **group-by bug** (see §6) that silently dropped
  half the data.

**Decision: replace static baselines with paired A/B testing.**

---

## 3. Core idea — paired A/B testing

Run the **candidate** build (B) and a **reference** build (A) **back-to-back,
interleaved, on the same nodes within the same SLURM allocation**, repeated N times:

```
repeat 1:  A  B
repeat 2:  A  B
...
repeat N:  A  B
```

Both builds are identical except for `librccl.so` (same HIP, same MPI, same GPUs,
same fabric — selected automatically via each binary's rpath). Because A and B run
in the same time window on the same hardware, environmental noise (thermals,
neighbor jobs, NIC/fabric state, slow drift) is **common-mode and cancels in the
A−B comparison**. We never ask "is this absolute number good?" (unanswerable for
small messages); we ask "is B worse than A, side-by-side, right now?"

This is the key to small-message stability: the *absolute* small-message bandwidth
is unstable, but the *paired difference* is not.

---

## 4. The detection algorithm (triple gate)

Implemented in `cvs/cvs/lib/regression_lib.py` (pure, dependency-free, unit-tested).

For every fully-qualified key **`(collective, size, type, inPlace)`**, we collect a
sample of bus-bandwidth measurements for A and for B (one per repeat). A key is
flagged as a regression **only if all three independent gates agree** — the
conjunction is what makes false positives extremely unlikely:

### Gate 1 — size-tiered relative threshold
`median(B)` must be lower than `median(A)` by more than the tier's threshold:

| tier  | size range      | why                              |
|-------|-----------------|----------------------------------|
| small | ≤ 1 MiB         | latency-bound, noisiest → loosest |
| mid   | 1 MiB – 64 MiB  | transitional                     |
| large | > 64 MiB        | bandwidth-bound, stable → tightest |

Thresholds are **derived from measured noise** (see §5), not guessed.

### Gate 2 — non-parametric separation
Require **`p75(B) < p25(A)`** — B's upper quartile below A's lower quartile, i.e.
the two distributions barely overlap. This is a distribution-free significance test
that is robust to a single straggler run and is the specific antidote to wide,
noisy small-message distributions (which overlap and therefore won't pass).

### Gate 3 — adjacency confirmation
A candidate size is confirmed only if it belongs to a run of **≥ `adjacency_min_run`
(default 2) consecutive candidate sizes** within the same `(collective, type,
inPlace)` group. Real regressions occupy a contiguous band of sizes; isolated noise
spikes do not.

### Safety rails
- **Median** (not mean) over repeats → robust to outlier runs.
- **`min_bandwidth_floor` (0.5 GB/s)**: the smallest sizes (~1K–64K) where busBw is
  near zero and relative noise explodes are marked **`inconclusive`** and excluded
  from pass/fail — we refuse to judge the region where no judgment is safe.
- **`min_repeats`**: too few samples → `inconclusive`, never a regression.
- Direction-aware: only flags **B worse than A**, never improvements.

### Output
Per-key verdicts (`pass` / `regression` / `inconclusive`) with A/B medians, drop%,
the threshold used, and the reasons each gate passed/failed. Aggregated to a single
job verdict; any confirmed regression → the test fails (non-zero exit) → CI fails.

---

## 5. Threshold calibration (control run)

Thresholds are **measured on the actual hardware**, not picked by hand.

1. Run in **control mode** (`control_mode: true`): the *reference* build is used as
   **both** A and B.
2. Since A and B are the same build, any spread is pure run-to-run noise. We compute
   the per-tier coefficient of variation and set:

   ```
   threshold[tier] = safety_factor (default 2.0) × p95(CV[tier])    # floored per tier
   ```

3. The control run **must report 0 regressions** (A vs A). If it doesn't, the
   detector/thresholds are not trustworthy on this hardware and the job fails loudly.

Re-run control calibration whenever the **hardware, RCCL build, or cluster config**
changes. Calibrated values are written to `ab_derived_thresholds.json`.

> Measured on 4-node MI350X (full matrix): `p95 CV` ≈ small 10% / mid 7% / large 3.7%
> → adopted thresholds **small 20% / mid 15% / large 7.5%**.

---

## 6. Correctness: group-by keys 

The original comparison/report code bucketed results by **message size alone**,
silently collapsing the `(data type, inPlace)` dimensions — the last row written for
a size overwrote the others. This both **hid real regressions** (overwritten rows
vanished) and **manufactured fake ones** (a data-type boundary looked like a giant
bandwidth dip).

Fix (in `rccl_lib.py`):
- `group_rccl_results()` — canonical grouping by `(type, inPlace)` + sort by size.
- `convert_to_graph_dict()` — expands each `(type, inPlace)` into its own series; no overwrites.
- `check_bw_dip` / `check_lat_dip` / `check_bus_bw` — group + sort before comparing.

Comparing **like-for-like on the full key** is a prerequisite for any verdict to be
meaningful, and is the foundation the A/B detector builds on.

---

## 7. Robustness features

### Retry transient failures (`ci_robustness_lib.run_with_retries`)
- A sweep that fails transiently (NCCL/MPI bootstrap, network, timeout) is retried
  up to `retry.max_retries` with linear backoff.
- **Data-corruption / schema-validation failures are never retried**
  (`classify_failure`) — retrying would only hide a genuine bug.
- Retries replace a failed run (they don't add samples), so statistics stay clean.
- Transparent on healthy runs (no behavior change when nothing fails).

### Kill stale GPU state before launch (`ci_robustness_lib.build_gpu_cleanup_script`)
- Runs **first** (test `test_00_cleanup_stale_gpu_state`) and **between retries**.
- Kills leftover RCCL/MPI processes (`pkill -f`, self-match-safe via the `[x]yz`
  trick), optionally GPU-holding PIDs (`rocm-smi --showpids`) and stale
  docker/podman containers. Best-effort — never fails the job.
- On exclusively-allocated nodes, any leftover process is stale by definition.

---

## 8. Architecture / code layout

Pure decision logic is separated from cluster orchestration so it can be
exhaustively unit-tested on a login node (no GPUs) — **61 unit tests**.

| File | Role |
|------|------|
| `cvs/cvs/lib/regression_lib.py` | **Pure** A/B detector: gates, percentiles, threshold derivation, report. |
| `cvs/cvs/lib/ci_robustness_lib.py` | **Pure** retry + GPU-cleanup builders/parsers. |
| `cvs/cvs/lib/rccl_lib.py` | Runs one RCCL sweep (`rccl_regression`), `group_rccl_results`, `cleanup_gpus_on_nodes`. |
| `cvs/cvs/tests/rccl/rccl_ab_regression.py` | Pytest orchestration: cleanup → interleaved A/B sweeps (with retry) → analyze. |
| `cvs/cvs/lib/unittests/test_regression_lib.py` | Detector tests incl. Monte-Carlo FP/detection sweeps. |
| `cvs/cvs/lib/unittests/test_ci_robustness_lib.py` | Retry + cleanup tests. |
| `cvs-sbatch/env/thor_rccl_env.sh` | NCCL/IB transport env (cv350 / MI350X + Broadcom Thor RoCE). |
| `cvs-sbatch/config_ab*.json` | A/B run configs. |
| `cvs-sbatch/sbatch/ab_regression.sbatch` | SLURM job (`sp_tests`, 4 nodes / 32 ranks). |
| `cvs-sbatch/run.sh`, `lib/python_env.sh` | Orchestrator: cluster.json gen, per-job uv venv. |

---

## 9. Configuration reference (`rccl` block)

```jsonc
{
  "rccl": {
    "mpi_params":  { "no_of_nodes": "4", "no_of_local_ranks": "8", "mpi_pml": "ob1",
                     "mpi_dir": "/apps/sp/ompi-install", "mpi_oob_port": "10.190.162.57/21" },
    "env_source_script": ".../thor_rccl_env.sh",
    "rccl_test_params": { "start_msg_size": "1024", "end_msg_size": "4G", "step_function": "2",
                          "no_of_iterations": "20", "warmup_iterations": "10", ... },
    "cvs_params": { "nic_model": "thor", "verify_bus_bw": "False", ... },

    "rccl_collective": ["all_reduce_perf", "reduce_scatter_perf", ...],
    "data_types": ["float", "bfloat16"],
    "regression": { "NCCL_ALGO": ["Ring"], "NCCL_PROTO": ["Simple"], "NCCL_PXN_DISABLE": ["0","1"] },

    "gpu_cleanup": { "enabled": true, "kill_gpu_pids": true, "kill_containers": false, "use_sudo": false },
    "retry":       { "max_retries": 2, "backoff_sec": 15 },

    "ab_regression": {
      "repeats": 7,
      "control_mode": false,            // true = reference-vs-itself calibration/stability proof
      "safety_factor": 2.0,             // thresholds = safety_factor x p95 noise (control mode)
      "thresholds": { "small": 0.20, "mid": 0.15, "large": 0.075 },
      "tier_boundaries": { "small_max_bytes": 1048576, "mid_max_bytes": 67108864 },
      "adjacency_min_run": 2,
      "min_repeats": 2,
      "min_bandwidth_floor": 0.5,
      "metric": "busBw", "higher_is_better": true,
      "output_dir": "/apps/sp/AIMVT-196/ab_artifacts",
      "reference": { "label": "ref",  "rccl_tests_dir": ".../reference/.../rccl-tests/build" },
      "candidate": { "label": "cand", "rccl_tests_dir": ".../candidate/.../rccl-tests/build" }
    }
  }
}
```

Notes:
- `librccl.so` is selected automatically by each binary's **rpath** — no
  `ld_library_path` needed (but `reference.ld_library_path` / `candidate.ld_library_path`
  are supported if a build needs it).
- The test matrix = `rccl_collective` × `data_types` × Cartesian product of `regression`
  env vars, each run for A and B × `repeats`.

---

## 10. How to run

### Local checkout (`/it-share/rccl-ci`)

Branches: `aimvt-196-rccl-regression-robustness` in both `cvs/` and `cvs-sbatch/`.
Cluster: **amd-tw**, reservation **rccl_dev**, 4 nodes / 32 ranks.

```bash
# Fast pipeline smoke (control mode, ~30 min): env + orchestration + detector, 0 regressions expected
sbatch /it-share/rccl-ci/sbatch/rccl_ab.sbatch

# Full-matrix control calibration (reference as both sides). Writes ab_derived_thresholds.json;
# MUST report 0 regressions.
sbatch --export=ALL,CONFIG_JSON=/it-share/rccl-ci/configs/ab_control.json \
       /it-share/rccl-ci/sbatch/rccl_ab.sbatch

# Real detection (reference vs candidate), using calibrated thresholds.
sbatch --export=ALL,CONFIG_JSON=/it-share/rccl-ci/configs/ab_detect.json \
       /it-share/rccl-ci/sbatch/rccl_ab.sbatch
```

**Build paths on this cluster**

| Side | rccl-tests dir | librccl (via rpath) |
|------|----------------|---------------------|
| reference | `/it-share/rccl-tests/build` | `/it-share/rccl/install/lib` |
| candidate | `/it-share/sp-tests/therock/bin` | `/it-share/sp-tests/therock/lib` |

**Logs** (under `/it-share/rccl-ci/logs/`):

- `sp_tests-<jobid>.out` / `.err` — Slurm capture (tee'd from the job).
- `run_<YYYYMMDD_HHMMSS>_<jobid>/` — timestamped run bundle:
  - `pytest.log` — pytest output
  - `slurm.out` / `slurm.err` — copies of the Slurm logs
  - `ab_artifacts/` — detector report, thresholds, `rccl_runs.log`
- `latest` — symlink to the most recent `run_*` directory.

**Artifacts** (also at `ab_artifacts/` during the run):

- `ab_regression_report.json` — per-key verdicts.
- `ab_derived_thresholds.json` — calibrated thresholds + measured noise (control mode).
- `rccl_runs.log` — clean per-run record: MPI launch command + rccl-tests output.

Exit code: non-zero (CI fail) if any confirmed regression.

**Prerequisites on amd-tw**

- `~/.ssh/cluster_id_ed25519` must exist and authorize SSH to all nodes in the
  allocation (auto-detected by `cvs-sbatch/run.sh`).
- RCCL reference/candidate binaries at the paths in `configs/ab_*.json` (see table above).

### Original cv350 / MI350X cluster (`/apps/sp/AIMVT-196`)

```bash
# Calibrate + prove stability (reference as both sides). Writes ab_derived_thresholds.json
# and MUST report 0 regressions.
sbatch --export=ALL,CONFIG_JSON=config_ab_full.json \
       /apps/sp/AIMVT-196/cvs-sbatch/sbatch/ab_regression.sbatch    # control_mode: true

# Real detection (reference vs candidate), using calibrated thresholds.
sbatch --export=ALL,CONFIG_JSON=config_ab_full.json \
       /apps/sp/AIMVT-196/cvs-sbatch/sbatch/ab_regression.sbatch    # control_mode: false
```

- All jobs are named **`sp_tests`**, 4 nodes / 32 ranks, partition `meta64` / `xgmi36`.

---

## 11. Trust model — how we know it's trustworthy

| Mechanism | What it buys |
|-----------|--------------|
| Paired A/B, interleaved | Cancels common-mode noise; stable even for small messages |
| Triple gate (threshold ∧ separation ∧ adjacency) | A false positive needs three unlikely things at once |
| Median + percentile separation | Resistant to single bad/straggler runs |
| Thresholds derived from measured noise (2× p95) | Bar provably sits above real run-to-run spread |
| `min_bandwidth_floor` → inconclusive | Abstains on the region where no judgment is safe |
| Correct full-key group-by | Compares like-for-like; no hidden/spurious signals |
| Pure, unit-tested core (61 tests) | Deterministic, auditable, regression-proof logic |
| A=A control = 0 regressions | Empirically measures the false-positive rate on real HW |

### Evidence collected
- **Monte-Carlo (simulated noise):** 0/400 false positives; 400/400 detection of an
  injected 15% regression.
- **Real 4-node MI350X control (A=A):** **0 false positives over 920 keys**
  (5 collectives × 2 dtypes × PXN {0,1}).
- **Real candidate detection (7.0.2 vs develop):** 106 confirmed regressions with a
  coherent, structured signature (selective per collective + PXN-dependent for
  all_gather; alltoall clean) — strong evidence of a real, localized change rather
  than noise. See `RCCL_REGRESSION_FINDINGS.md`.

---

## 12. Limitations & future work

- **Global per-tier thresholds** are set by the noisiest collective. A *global*
  `large = 7.5%` (driven by alltoall/broadcast noise) means a ~6% large-message
  all_reduce regression is missed. **Per-collective thresholds** would recover that
  sensitivity without sacrificing stability.
- **Sub-floor tiny messages (1K–64K)** are `inconclusive` (busBw ≈ 0). A
  **latency-based comparison** (`metric: "time"`, already supported by the detector)
  would extend trustworthy coverage to the smallest sizes, where latency is the
  meaningful quantity and pairs just as well.
- **Retry path** is proven by unit tests; a fault-injection run would also exercise a
  real transient retry + cleanup cycle on hardware.
- **Single-node** runs hit an OpenMPI intra-node bootstrap issue on this cluster; the
  validated path is multi-node under `sbatch` (which is the CI path anyway).
- Periodic **A=A canary** runs in CI are recommended to continuously confirm the
  false-positive rate stays 0 as the cluster/software evolves.

---

## 13. One-line summary

**Trust = paired design to cancel noise + a triple gate and robust statistics to
resist what's left + thresholds calibrated from measured on-hardware noise + an A=A
control that empirically proves zero false positives — all in a pure, unit-tested,
auditable core, wrapped with retry and stale-GPU cleanup for CI resilience.**
