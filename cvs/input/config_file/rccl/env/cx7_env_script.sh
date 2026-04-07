#!/usr/bin/env bash
# RCCL environment for ConnectX-7 (cx7) class InfiniBand.
# Copy to a stable path on your nodes, edit the values, and set rccl.run.env_script to that path.
# Thor2 uses thor2_env_script.sh so cx7 and thor2 stay explicit even when tuning matches today.

export RCCL_TESTS_BUILD_DIR="<changeme>/rccl-tests/build"
export ROCM_HOME="/opt/rocm"
export RCCL_HOME="${ROCM_HOME}"
export MPI_HOME="<changeme>/openmpi"

export PATH="${MPI_HOME}/bin:${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${RCCL_HOME}/lib:${MPI_HOME}/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"

# NCCL_IB_* / UCX / Open MPI MCA — add cx7-specific tuning below.
