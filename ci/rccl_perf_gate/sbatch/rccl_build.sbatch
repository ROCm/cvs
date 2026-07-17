#!/usr/bin/env bash

##############################################################
# RCCL build-only Slurm entry point.
#
# Builds reference + candidate librccl.so on a single node, independent of the
# 4-node detection allocation (rccl_ab.sbatch). Split out because the build is
# a single-node CPU compile with no RDMA/MPI involved -- holding all 4 pinned
# nodes exclusively for the ~15-20min build duration wastes 3 of them, and
# ties up the pinned-node concurrency lock longer than the detection step
# actually needs it.
#
# Submit from anywhere:
#   sbatch --export=ALL,CANDIDATE_SRC=...,BASE_REF=...,CONFIG_JSON=... \
#          /it-share/rccl-ci/cvs/ci/rccl_perf_gate/sbatch/rccl_build.sbatch
##############################################################

#SBATCH --job-name=sp_build
#SBATCH --partition=amd-tw
# reservation passed via CLI --reservation=<name> when SLURM_RESERVATION is set
#SBATCH --account=amd-tw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
# Build is single-node CPU compile only (no RDMA/MPI) -- pin to one of the
# reserved nodes. Override with --nodelist at submit if g28 is unavailable
# (e.g. mia1-p01-g32).
#SBATCH --nodelist=mia1-p01-g28
#SBATCH --time=01:00:00
#SBATCH --chdir=/it-share/rccl-ci

#SBATCH --output=/it-share/rccl-ci/logs/sp_build-%j.out
#SBATCH --error=/it-share/rccl-ci/logs/sp_build-%j.err

set -euo pipefail

readonly RCCL_CI_ROOT="${RCCL_CI_ROOT:-/it-share/rccl-ci}"
exec bash "${RCCL_CI_ROOT}/cvs/ci/rccl_perf_gate/sbatch/run_rccl_build.sh"
