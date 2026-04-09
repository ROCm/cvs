#!/usr/bin/env bash
# RCCL environment for Thor2-class networking.
# Copy to a stable path on your nodes, edit the values, and set rccl.run.env_script to that path.
# Intentionally a sibling of cx7_env_script.sh (not a shared include) so NIC choice stays obvious.

export RCCL_TESTS_BUILD_DIR="<changeme>/rccl-tests/build"
export ROCM_HOME="/opt/rocm"
export RCCL_HOME="${ROCM_HOME}"
export MPI_HOME="<changeme>/openmpi"

export PATH="${MPI_HOME}/bin:${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${RCCL_HOME}/lib:${MPI_HOME}/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL_IB_* / UCX / Open MPI MCA — add Thor2-specific tuning below.
