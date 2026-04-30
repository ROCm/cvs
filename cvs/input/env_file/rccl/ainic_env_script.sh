#!/usr/bin/env bash
# RCCL environment for AINIC + ANP plugin.
# Copy to a stable path on your nodes, edit values for your site, and point the RCCL config at it.

export ROCM_HOME="/opt/rocm"
export RCCL_HOME="${ROCM_HOME}"
export MPI_HOME="<changeme>/openmpi"
export ANP_HOME_DIR="<changeme>/path/to/amd-anp"

# NCCL tuning parameters
export NCCL_DEBUG=ERROR
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=1
export NCCL_IB_TIMEOUT=30
export NCCL_IB_SL=0
export NCCL_IB_TC=0
export NCCL_IB_FIFO_TC=0
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_PXN_DISABLE=0
export IB_RX_QUEUE_LEN=8192
export HCOLL_ENABLE_MCAST_ALL=0
export NCCL_CUMEM_ENABLE=0
export IONIC_LOCKFREE=all
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_NET_OPTIONAL_RECV_COMPLETION=1
export NCCL_IB_USE_INLINE=1
export NCCL_DMABUF_ENABLE=0
export NCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
export NCCL_NET_PLUGIN=librccl-anp.so

export PATH="${MPI_HOME}/bin:${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${RCCL_HOME}/lib:${ANP_HOME_DIR}:${MPI_HOME}/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${ANP_HOME_DIR}/build/librccl-anp.so:${RCCL_HOME}/lib/librccl.so"