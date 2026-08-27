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
# NCCL_CUMEM_ENABLE must be 1 on bnxt_re (Broadcom Thor2) + ROCm/RCCL builds where
# HIP_VERSION >= 71260540 (ROCm 7.1+). RCCL's net.cc only compiles the ROCm-native
# hsa_amd_portable_export_dmabuf() registration path when HIP_VERSION < 71260540;
# on newer builds it takes the CUDA-style path instead, which is gated behind
# ncclCuMemEnable(). With CUMEM_ENABLE=0 (this script's old default) that gate is
# never satisfied, so RCCL silently falls back to a plain ibv_reg_mr_iova2() call
# on a raw GPU pointer -- which bnxt_re's kernel driver rejects with
# "ib_umem_get failed! rc = -14" (EFAULT / "Bad address"), symmetrically on every
# node, because GPU VRAM isn't get_user_pages()-able without a peer-mem module.
# Setting CUMEM_ENABLE=1 also lets NCCL recognize UALoE/scale-up-fabric GPUs as
# directly P2P-reachable (P2P/CUMEMMNNVL), bypassing NET/IB entirely when a
# scale-up fabric is present. Verified on a 2-node Helios (bnxt_re) cluster:
# with CUMEM_ENABLE=0 all RCCL collectives failed at ncclCommInitRank; with
# CUMEM_ENABLE=1 all collectives (AllReduce/AllGather/ReduceScatter/AllToAll/
# Broadcast, fp32+bf16, 1KB-1GB) passed. See docs/reference/configuration-files/network/rccl.rst.
export NCCL_CUMEM_ENABLE=1
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DMABUF_ENABLE=1
export NCCL_NET_PLUGIN=none

export PATH="${MPI_HOME}/bin:${ROCM_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${RCCL_HOME}/lib:${MPI_HOME}/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"