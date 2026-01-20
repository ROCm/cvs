#!/bin/bash
##############################################################
# Environment setup script for RCCL tests
# Sourced before running rccl-tests via env_source_script config
##############################################################

# GPU Memory management - prevent scratch memory reclaim issues
export HSA_NO_SCRATCH_RECLAIM=1

# CPU affinity - let NCCL handle its own affinity
export NCCL_IGNORE_CPU_AFFINITY=1

# NCCL socket interface for bootstrap communication
# Using the interfaces from the reference configuration
export NCCL_SOCKET_IFNAME=fenic0,enp49s0f0np0

echo "Environment setup complete: HSA_NO_SCRATCH_RECLAIM=$HSA_NO_SCRATCH_RECLAIM, NCCL_IGNORE_CPU_AFFINITY=$NCCL_IGNORE_CPU_AFFINITY"

