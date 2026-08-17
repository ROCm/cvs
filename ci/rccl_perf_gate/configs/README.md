# Perf-gate configs

These are the configs that decide what the gate measures and what counts as a
regression. They used to exist **only** at `/it-share/rccl-ci/configs/` on NFS,
which meant the gate's decision boundary had no history: a threshold could be
widened by hand at 2am and nothing would record that it happened, who did it, or
what it was before. A green check is only as trustworthy as the numbers behind
it, so those numbers are now version-controlled.

## Which file does what

| file | used by | role |
|---|---|---|
| `ci_detect_prod.json` | the PR gate (`rccl_perf_regression.yml`, default `config` input) | reference-vs-candidate detection: test matrix, repeats, thresholds, timeouts |
| `ci_control.json` | calibration / control (A=A) runs | derives the noise floor that the detect thresholds are set against |

## NFS is still the live copy

Nothing reads from this directory at runtime. `/it-share/rccl-ci/configs/` remains
the deployment target, because the workflow, the sbatch scripts and hand-run
`workflow_dispatch` invocations all pass absolute paths into it.

So this directory is a **source of truth that must be kept in sync by hand**:

```bash
# after editing a config here
scp cvs/ci/rccl_perf_gate/configs/ci_detect_prod.json \
    tensorwave-slurm-rccl:/it-share/rccl-ci/configs/ci_detect_prod.json

# to check for drift
ssh tensorwave-slurm-rccl 'md5sum /it-share/rccl-ci/configs/ci_detect_prod.json'
md5sum cvs/ci/rccl_perf_gate/configs/ci_detect_prod.json
```

Wiring the readers to pull straight from the repo checkout would remove the
manual step, but it changes where a live gate loads its config from, so it is
deliberately left as a follow-up rather than folded into a robustness pass.

## Editing thresholds

Don't hand-tune `thresholds`. Run a control (A=A) calibration, which writes
`configs/ab_derived_thresholds.json` via `median + k*MAD`, and let the detector
pick it up. `max_thresholds` is the ceiling that stops a noisy calibration from
loosening the gate into uselessness — that one is a policy decision and *is*
meant to be edited by hand.

`_comment` fields inside the configs record why individual collectives are
skipped. Read them before re-enabling anything: `alltoall_perf` is excluded
because pooling it inflated the derived large-tier threshold by 60-100x, which
would have blinded the gate for every other collective. Per-collective
thresholds now make re-enabling it possible, but only after a fresh calibration.
