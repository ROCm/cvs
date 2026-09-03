# rccl_perf_gate

Slurm submission/polling/reporting glue for the RCCL paired A/B performance
regression gate used by `ROCm/rocm-systems`'s
[`rccl_perf_regression.yml`](https://github.com/ROCm/rocm-systems/blob/main/.github/workflows/rccl_perf_regression.yml)
GitHub Actions workflow.

The workflow's self-hosted runner invokes these scripts directly:

- `sbatch/rccl_build.sbatch`, `sbatch/run_rccl_build.sh` — build RCCL (via
  `cvs-sbatch`) as a Slurm job.
- `submit_and_poll.sh`, `sbatch/rccl_ab.sbatch`, `sbatch/run_rccl_ab.sh` —
  submit the paired A/B regression job (`cvs/tests/rccl/rccl_ab_regression.py`),
  poll it to completion, and map its exit code to a CI-gatable result.
- `format_report.py` — render the A/B run's JSON result into a Markdown
  summary for the workflow's job summary / PR comment.

All scripts honor an `RCCL_CI_ROOT` env override (default `/it-share/rccl-ci`)
so they aren't tied to one cluster's NFS layout.

## Status

This is a stopgap. It exists because CVS does not yet submit and manage Slurm
(or Kubernetes) jobs natively — these scripts are thin bash wrappers around
`sbatch`/`squeue` bridging that gap. Once CVS gains native scheduler
integration, this directory should be retired in favor of driving the A/B
regression test directly through CVS.
