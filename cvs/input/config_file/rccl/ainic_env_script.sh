# Recommended AINIC env setting
export IONIC_LOCKFREE=all
export NCCL_IB_FIFO_TC=185
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_NET_OPTIONAL_RECV_COMPLETION=1
export NCCL_IB_USE_INLINE=1
export NCCL_DMABUF_ENABLE=0
export NCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
export NCCL_NET_PLUGIN=librccl-anp.so
export RCCL_HOME_DIR=/opt/rocm
export ANP_HOME_DIR=<changeme>/path/to/amd-anp/> # Replace '<changeme>/path/to/amd-anp/' with the absolute path to your amd-anp installation directory
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${RCCL_HOME_DIR}/lib:${ANP_HOME_DIR} # Check RCCL_HOME_DIR and ANP_HOME_DIR structures; use /build if libraries are in build/ subdir, otherwise /lib
export LD_PRELOAD="${ANP_HOME_DIR}/build/librccl-anp.so:${RCCL_HOME_DIR}/lib/librccl.so" # Check RCCL_HOME_DIR and ANP_HOME_DIR structures; use /lib if libraries are in lib/ subdir, otherwise /build