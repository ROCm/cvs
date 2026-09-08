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
- **`min_bandwidth_floor`, now per tier** (`{small: 0.005, mid: 0.05, large: 0.5}`
  GB/s): sizes whose busBw sits under the floor for their tier are marked
  **`inconclusive`** and excluded from pass/fail — we refuse to judge the region
  where no judgment is safe. A single scalar `0.5` is still accepted for old
  configs, but it was a mistake: 0.5 GB/s is a *large-message* floor, and applying
  it flat silently excluded the entire small tier, so the gate was scoring the
  small band against thresholds calibrated on no small-band data at all.
- **`min_repeats`**: too few samples → `inconclusive`, never a regression.
- **`require_balanced_samples`**: A and B must have the same number of surviving
  repeats for a key, or it is `inconclusive`. Unequal counts mean the two sides
  were not measured under the same conditions, so the comparison is not paired.
- **`max_inconclusive_frac`** (0.1): if more than this fraction of a group's keys
  came back inconclusive, the *group* is untrustworthy. A detector that abstained
  on most of what it looked at has not established anything, and must not be
  allowed to report "0 regressions" as though it had.
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
   the per-key coefficient of variation and set:

   ```
   threshold[tier] = safety_factor × (median(CV[tier]) + mad_k × MAD(CV[tier]))
   threshold[tier] = min(threshold[tier], max_thresholds[tier])      # policy ceiling
   ```

   This used to be `safety_factor × p95(CV[tier])`. p95 of a CV distribution *is*
   an outlier statistic — one flaky key in a tier dragged the whole tier's
   threshold up and blinded the gate for every other key in it. `median + k·MAD`
   (MAD scaled by 1.4826) is the robust equivalent: it tracks the bulk of the
   distribution and a few bad keys cannot move it far.

   Thresholds are derived **per collective**, not globally, so the noisiest
   collective no longer sets the bar for the quietest.

3. `max_thresholds` is a hand-set ceiling. Calibration can only ever make the gate
   *tighter* than this; a pathologically noisy control run cannot loosen the gate
   into uselessness. It is the one threshold knob that is meant to be edited by
   hand — see `ci/rccl_perf_gate/configs/README.md`.

4. The control run **must report 0 regressions** (A vs A) **and** must be
   trustworthy. A control run that tripped the circuit breaker, came back mostly
   inconclusive, or failed to score every group does **not** publish its
   calibration: it writes only into its own run artifacts, and logs loudly that it
   withheld. Publishing from a run that failed its own checks would poison every
   later detect run with a threshold nobody chose, and because that run also
   reports NO VERDICT, nobody would think to go look at what it had published.

Re-run control calibration whenever the **hardware, RCCL build, or cluster config**
changes. Calibrated values are written to `ab_derived_thresholds.json`.

> Measured on 4-node MI355X (full matrix, job 16131): derived
> **small 12.5% / mid 6.1% / large 3.0%**, all under the `max_thresholds` ceiling
> of 15% / 8% / 6% — so the ceiling was not binding and the measured noise is what
> the gate actually uses.

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

### Refuse to answer rather than answer wrongly

The organising rule of the CI wrapper: **0 confirmed regressions is only a PASS if
the detector actually looked.** Every layer that could turn "we did not measure"
into "✅ PASS" has been closed:

- Each group carries `summary.trustworthy`. A group that tripped the circuit
  breaker, exceeded `max_inconclusive_frac`, or had unbalanced A/B sample counts
  is not trustworthy.
- The report carries a top-level `trustworthy` and `untrustworthy_reasons`, plus
  `groups_scored` / `groups_expected` so a silently-dropped group is visible.
- `format_report.py` renders **⚠️ NO VERDICT (not measured)** and exits **2** for
  an untrustworthy report — neither a pass nor a regression. The workflow gate
  step propagates exit 2, and the PR comment says so in as many words.
- Reports predating the flag are treated as untrustworthy, not as passes.

### Transport capability pre-flight

`run_rccl_ab.sh` checks the A/B transport capabilities before spending an
allocation. A run whose interconnect quietly fell back to a slower path measures
something real, but not the thing the gate is supposed to be gating.

### Circuit breaker and right-sized timeouts

`circuit_breaker_failures` (2) abandons a group after consecutive failures instead
of grinding through the whole matrix at the per-collective timeout.
`per_collective_timeout_sec` is 360s, not the 1800s it was: with 8 groups × 7
repeats × 2 sides, the old value put the all-timeout path well past a working day.
The Slurm wall clock, the poller's run and queue budgets, and the Actions job
timeout are now nested innermost-first so the inner layer always fires first and
gets to say *why* — see the table in `submit_and_poll.sh`.

### Per-run workspace isolation

Every run gets its own directory under `runs/<run_key>/`: an immutable `cvs/`
snapshot (so a mid-flight deploy cannot change the code a running job executes),
its own config copy, build output, artifacts and logs. Concurrent runs cannot
read or clobber each other's state, and the report records which detector commit
produced it. A cron janitor reclaims old runs, old build-cache entries and old
Slurm logs, reporting what it kept as well as what it removed.

---

## 8. Architecture / code layout

Pure decision logic is separated from cluster orchestration so it can be
exhaustively unit-tested on a login node (no GPUs).

**Test inventory** (all runnable without an allocation): 34 collected pytest cases
across `test_regression_lib.py` + `test_ci_robustness_lib.py`, 8 more in
`ci/rccl_perf_gate/tests/test_regression_lib.py`, and 21 shell assertions across
the two `.sh` tests below.

| File | Role |
|------|------|
| `cvs/cvs/lib/regression_lib.py` | **Pure** A/B detector: gates, percentiles, threshold derivation, report. |
| `cvs/cvs/lib/ci_robustness_lib.py` | **Pure** retry + GPU-cleanup builders/parsers. |
| `cvs/cvs/lib/rccl_lib.py` | Runs one RCCL sweep (`rccl_regression`), `group_rccl_results`, `cleanup_gpus_on_nodes`. |
| `cvs/cvs/tests/rccl/rccl_ab_regression.py` | Pytest orchestration: cleanup → interleaved A/B sweeps (with retry) → analyze. |
| `cvs/cvs/lib/unittests/test_regression_lib.py` | Detector tests incl. Monte-Carlo FP/detection sweeps. |
| `cvs/cvs/lib/unittests/test_ci_robustness_lib.py` | Retry + cleanup tests. |
| `cvs-sbatch/env/thor_rccl_env.sh` | NCCL/IB transport env (cv350 / MI350X + Broadcom Thor RoCE). |
| `cvs-sbatch/env/ainic_rccl_env.sh` | NCCL/IB transport env (tensorwave / MI355X + AINIC). |
| `cvs-sbatch/config_ab*.json` | A/B run configs. |
| `cvs-sbatch/sbatch/ab_regression.sbatch` | SLURM job (`sp_tests`, 4 nodes / 32 ranks). |
| `cvs-sbatch/run.sh`, `lib/python_env.sh` | Orchestrator: cluster.json gen, per-job uv venv. |

### The CI wrapper (`cvs/ci/rccl_perf_gate/`)

Everything the GitHub gate needs that is not detection logic. Added after the
detector itself, and version-controlled here rather than living loose on NFS.

| File | Role |
|------|------|
| `submit_and_poll.sh` | Submits `rccl_ab.sbatch`, polls to a terminal state, maps the job's exit code to the step's. Owns the run/queue budgets and a `scancel` trap so a killed step never orphans an allocation. |
| `format_report.py` | Renders the PR comment. Exit **0** pass / **1** regression / **2** NO VERDICT. |
| `sbatch/rccl_ab.sbatch` | Detect job: 4 pinned nodes, `--exclusive`, 4h wall clock. |
| `sbatch/rccl_build.sbatch` | Build job: single node, CPU compile. Deliberately **not** in the detect reservation — see below. |
| `sbatch/run_rccl_ab.sh` | In-allocation orchestration: transport pre-flight, workspace setup, detector invocation, verdict recount. |
| `sbatch/run_rccl_build.sh` | Content-addressed build of reference + candidate `librccl.so`, keyed on git rev + recipe hash. |
| `sbatch/lib/workspace.sh` | Per-run workspace creation, build cache, GC, and the cron janitor. |
| `configs/*.json` | Snapshots of the live NFS configs, so the gate's decision boundary has version history. **NFS is still what runs** — see `configs/README.md`. |
| `tests/test_regression_lib.py` | 8 detector tests aimed at the trustworthiness paths. |
| `tests/test_workspace_gc.sh` | 13 assertions; runs GC against a throwaway root. |
| `tests/test_build_submit_trap.sh` | 8 assertions; kills the build step mid-flight and checks the allocation is released. |
| `tests/lint_workflow.py` | `bash -n` over every `run:` block in the workflow. |

> **Reservations.** `rccl_ci` is exactly the four pinned detect nodes
> (`mia1-p01-g[22,26,28,32]`) and the detect job takes all four `--exclusive`, so
> it has no slack. The build must therefore stay out of it: it is a single-node
> CPU compile that needs no GPU, and it runs in `rccl_dev` via
> `SLURM_BUILD_RESERVATION`. It used to pin `--nodelist=mia1-p01-g28`, inside the
> detect set, where it queued behind — and delayed — anything using the pool.

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
      "safety_factor": 2.0,
      "mad_k": 3.0,                     // thresholds = safety_factor x (median + mad_k x MAD)
      "thresholds":     { "small": 0.15, "mid": 0.08, "large": 0.06 },
      "max_thresholds": { "small": 0.15, "mid": 0.08, "large": 0.06 },  // hand-set ceiling
      "tier_boundaries": { "small_max_bytes": 1048576, "mid_max_bytes": 67108864 },
      "adjacency_min_run": 2,
      "min_repeats": 2,
      "min_bandwidth_floor": { "small": 0.005, "mid": 0.05, "large": 0.5 },  // per tier

      // Trustworthiness gates -- see section 7.
      "circuit_breaker_failures": 2,    // abandon a group after N consecutive failures
      "require_balanced_samples": true, // A and B must have equal surviving repeats
      "max_inconclusive_frac": 0.1,     // a group that abstained on >10% is untrustworthy
      "publish_derived_thresholds": true, // control mode only; withheld if untrustworthy

      // alltoall is excluded: pooling it inflated the derived large-tier threshold
      // by 60-100x, blinding the gate for every other collective. Per-collective
      // thresholds make re-enabling it possible, but only after a fresh calibration.
      "skip_keys": [["alltoall_perf", "float"], ["alltoall_perf", "bfloat16"]],

      "metric": "busBw", "higher_is_better": true,
      "output_dir": "/it-share/rccl-ci/runs/<run_key>/artifacts",   // per-run, not shared
      "reference": { "label": "ref",  "rccl_tests_dir": "...", "ld_library_path": "..." },
      "candidate": { "label": "cand", "rccl_tests_dir": "...", "ld_library_path": "..." }
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
| Thresholds from measured noise (`median + 3·MAD`, per collective) | Bar sits above real run-to-run spread, and a few flaky keys cannot move it |
| `max_thresholds` ceiling | Calibration can only tighten the gate, never loosen it into uselessness |
| Per-tier `min_bandwidth_floor` → inconclusive | Abstains on the region where no judgment is safe, *per tier* |
| Correct full-key group-by | Compares like-for-like; no hidden/spurious signals |
| Pure, unit-tested core (42 pytest cases + 21 shell assertions) | Deterministic, auditable, regression-proof logic |
| A=A control = 0 regressions | Empirically measures the false-positive rate on real HW |
| **Trustworthiness propagated to the verdict** | "0 regressions" cannot be reported unless the detector actually measured; otherwise NO VERDICT (exit 2) |
| Calibration withheld from an untrustworthy control run | A control run that failed its own checks cannot poison later detect runs |
| Immutable per-run `cvs/` snapshot + provenance in the report | The verdict names the exact detector commit and both builds that produced it |

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

- ~~**Global per-tier thresholds** are set by the noisiest collective.~~
  **Done.** Thresholds are now derived per collective, and the estimator is
  `median + k·MAD` rather than `p95`, so one flaky key no longer sets the bar for
  its whole tier. `alltoall_perf` remains in `skip_keys` and can be re-enabled
  after a fresh calibration.
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
- **`sbatch --wait` instead of polling.** `submit_and_poll.sh` polls `squeue` on a
  30s interval. `--wait` would remove the polling loop, but it also removes the
  per-state visibility the budgets are built on (queued-vs-running is what lets
  the run budget be charged against run time only). Deliberately deferred.
- **The configs are hand-synced.** The gate loads from `/it-share/rccl-ci/configs/`
  on NFS; `ci/rccl_perf_gate/configs/` is a snapshot for history that nothing reads.
  Pointing the readers at the repo checkout would close the drift window but
  changes where a live gate loads from, so it is a follow-up, not part of a
  robustness pass. Diff before trusting either copy.
- **The detect reservation is shared.** `rccl_ci` is four nodes and the detect job
  needs all four; a neighbouring single-node pipeline occupying one of them is
  enough to stall the gate, and its declared `TimeLimit` (not its real runtime) is
  what Slurm's backfill scheduler plans around. Node contention, not detector
  runtime, is now the dominant term in end-to-end latency.

---

## 13. One-line summary

**Trust = paired design to cancel noise + a triple gate and robust statistics to
resist what's left + thresholds calibrated from measured on-hardware noise + an A=A
control that empirically proves zero false positives — all in a pure, unit-tested,
auditable core, wrapped with retry and stale-GPU cleanup for CI resilience — and a
gate that refuses to say PASS when it did not actually measure.**
