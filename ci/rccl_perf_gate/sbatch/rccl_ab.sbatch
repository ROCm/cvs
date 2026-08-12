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
# 5h, not the 8h this used to ask for. A healthy full-matrix run is ~35min
# (job 16368, 10 groups, 00:34:56; it was ~25min at 8 groups, jobs 16092/16108).
# The pathological path is bounded too: a 360s per-collective timeout and a
# circuit breaker that abandons a group after 2 consecutive failures cap the
# all-timeouts case at groups x 2 sides x 2 failures x 360s.
#
# That bound is why this is 5h and not 4h. At the 8 active (collective,dtype)
# groups this ran before alltoall_perf was re-enabled it came to 3.2h, safely
# inside 4h. At 10 groups it is exactly 4.0h -- equal to the wall clock, which
# means Slurm would kill the job at the moment the breaker was about to report
# which group wedged, and the diagnostic would be unreachable. Retune this
# alongside MAX_WAIT_SEC in submit_and_poll.sh if the matrix changes again.
#
# 8h was not a safety margin, it was a hostage window. This job holds four
# nodes --exclusive, and the GitHub runner polling it holds one of the two
# runner slots the whole time, so a wedged run took the pinned pool and the
# pipeline down with it for a working day. 5h is ~8x the healthy runtime and
# still above the breaker-bounded worst case; anything slower than that is a
# fault to be reported, not waited out.
#SBATCH --time=05:00:00
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
