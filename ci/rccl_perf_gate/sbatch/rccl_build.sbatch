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
# Build is a single-node CPU compile (no RDMA/MPI), so it does not care which
# node it lands on -- and it must NOT land on one of the detect nodes.
#
# This used to pin --nodelist=mia1-p01-g28, which is inside the 4-node rccl_ci
# detect reservation. The build is submitted with `sbatch --wait` and takes the
# node --exclusive, so for up to an hour it held a quarter of the pool that the
# detect job then needs all of. Build and detect were serialised against each
# other on the same four nodes for no reason: a self-inflicted contribution to
# the multi-hour end-to-end times.
#
# No --nodelist here. The caller picks the pool via --reservation (the workflow
# uses SLURM_BUILD_RESERVATION=rccl_dev), and Slurm picks a free node from it.
# Pass --nodelist at submit only if you need a specific machine.
#SBATCH --time=01:00:00
#SBATCH --chdir=/it-share/rccl-ci

#SBATCH --output=/it-share/rccl-ci/logs/sp_build-%j.out
#SBATCH --error=/it-share/rccl-ci/logs/sp_build-%j.err

set -euo pipefail

readonly RCCL_CI_ROOT="${RCCL_CI_ROOT:-/it-share/rccl-ci}"
exec bash "${RCCL_CI_ROOT}/cvs/ci/rccl_perf_gate/sbatch/run_rccl_build.sh"
