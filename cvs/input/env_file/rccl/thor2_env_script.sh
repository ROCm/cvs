#!/usr/bin/env bash
# RCCL environment for Thor2-class networking.
# Copy to a stable path on your nodes, edit values for your site, and point the RCCL config at it.

export ROCM_HOME="/opt/rocm"
export RCCL_HOME="${ROCM_HOME}"
export MPI_HOME="<changeme>/openmpi"

# NCCL tuning parameters
export NCCL_DEBUG=ERROR
export NCCL_IB_HCA="bnxt_re0,bnxt_re1,bnxt_re2,bnxt_re3,bnxt_re4,bnxt_re5,bnxt_re6,bnxt_re7"
export UCX_NET_DEVICES="ens28np0,ens27np0,ens25np0,ens26np0,ens24np0,ens23np0,ens21np0,ens22np0"
export UCX_TLS=tcp
export NCCL_SOCKET_IFNAME=eth1,eth0
export NCCL_IB_GID_INDEX=1
export NCCL_IB_TIMEOUT=30
export NCCL_IB_SL=0
export NCCL_IB_TC=0
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_PXN_DISABLE=0
export IB_RX_QUEUE_LEN=8192
export HCOLL_ENABLE_MCAST_ALL=0
export NCCL_CUMEM_ENABLE=0
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DMABUF_ENABLE=1
export NCCL_NET_PLUGIN=none

export PATH="${MPI_HOME}/bin:${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${RCCL_HOME}/lib:${MPI_HOME}/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"