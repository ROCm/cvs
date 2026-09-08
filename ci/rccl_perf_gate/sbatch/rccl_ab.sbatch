#!/usr/bin/env bash

##############################################################
# RCCL paired A/B regression CI entry point.
#
# Submit from anywhere:
#   sbatch /it-share/rccl-ci/cvs/ci/rccl_perf_gate/sbatch/rccl_ab.sbatch
#
# Override the benchmark config (default: fast control / robustness):
#   sbatch --export=ALL,CONFIG_JSON=/it-share/rccl-ci/configs/ab_control.json \
#          /it-share/rccl-ci/cvs/ci/rccl_perf_gate/sbatch/rccl_ab.sbatch
#
# Presets:
#   configs/ab_robustness.json  - fast 4-node control (pipeline smoke)
#   configs/ab_control.json     - full-matrix control + threshold calibration
#   configs/ab_detect.json      - reference vs candidate detection
##############################################################

#SBATCH --job-name=sp_tests
#SBATCH --partition=amd-tw
# reservation passed via CLI --reservation=<name> when SLURM_RESERVATION is set
#SBATCH --account=amd-tw
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
# Pin to a fixed 4-node set so calibration and detection share identical hardware
# (keeps derived thresholds valid across runs). Override with --nodelist at submit.
#SBATCH --nodelist=mia1-p01-g[22,26,28,32]
# 4h. A healthy full-matrix run is ~27min at the current 8 groups graded in a
# single orientation; it was ~35min at 10 groups grading both (job 16368,
# 00:34:56). The pathological path is bounded too: a 360s per-collective timeout
# and a circuit breaker that abandons a group after 2 consecutive failures cap
# the all-timeouts case at groups x 2 sides x 2 failures x 360s.
#
# That arithmetic, not the round number, is what sets this. At 8 groups it is
# 3.2h, which leaves the breaker room to report which group wedged before Slurm
# kills the job. At the 10 groups this ran with broadcast_perf enabled it was
# exactly 4.0h -- equal to the wall clock, so the diagnostic would have been
# unreachable, and 5h was the answer to that. Dropping broadcast_perf restored
# the margin and the budget came back down. Retune alongside MAX_WAIT_SEC in
# submit_and_poll.sh if the matrix changes again; do not just widen this.
#
# The 8h it asked for before either retune was not a safety margin, it was a
# hostage window. This job holds four nodes --exclusive, and the GitHub runner
# polling it is occupied the whole time, so a wedged run took the pinned pool
# and the pipeline down with it for a working day. 4h is ~9x the healthy runtime
# and still above the breaker-bounded worst case; anything slower than that is a
# fault to be reported, not waited out.
#SBATCH --time=04:00:00
#SBATCH --chdir=/it-share/rccl-ci

#SBATCH --output=/it-share/rccl-ci/logs/sp_tests-%j.out
#SBATCH --error=/it-share/rccl-ci/logs/sp_tests-%j.err

set -euo pipefail

# Only the batch-script process on the head node runs the orchestrator; other
# allocated tasks exit immediately so we keep a true 4-node exclusive job.
if [[ "${SLURM_PROCID:-0}" -ne 0 ]]; then
  exit 0
fi

readonly RCCL_CI_ROOT="${RCCL_CI_ROOT:-/it-share/rccl-ci}"
exec bash "${RCCL_CI_ROOT}/cvs/ci/rccl_perf_gate/sbatch/run_rccl_ab.sh"
