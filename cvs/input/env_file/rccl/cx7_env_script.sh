#!/usr/bin/env bash
# RCCL environment for ConnectX-7 (cx7) class InfiniBand.
# Copy to a stable path on your nodes, edit values for your site, and point the RCCL config at it.

export ROCM_HOME="/opt/rocm"
export RCCL_HOME="${ROCM_HOME}"
export MPI_HOME="<changeme>/openmpi"

# NCCL tuning parameters
export NCCL_DEBUG=ERROR
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
export UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1"
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