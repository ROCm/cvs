# Production CVS-runner image (P5).
#
# Substrate-only: TheRock provides ROCm + rccl-tests; system apt provides
# build prereqs + python deps + ibverbs runtime + sudo + git. CVS source is
# copied in so install_* scripts (install_rvs, install_transferbench,
# install_agfhc, install_ibperf_tools) can run inside the container at
# prepare_runtime time (P6).
#
# This image deliberately does NOT bundle TransferBench, RVS, AGFHC, or
# ibperf-tools as pre-built artifacts. Those land at prepare_runtime time
# via `cvs run install_*` in docker mode -- single source of truth.

FROM ubuntu:22.04

# --- build args ---------------------------------------------------------
# OFFLOAD_ARCHES is the GPU_TARGETS string passed to install_transferbench
# at prepare_runtime time (recorded in the manifest, P6 reads it). Required.
# THEROCK_TARBALL_NAME is the basename of the TheRock dist tarball that
# build.sh has placed alongside this Dockerfile in the build context.
ARG OFFLOAD_ARCHES
ARG THEROCK_TARBALL_NAME

RUN test -n "${OFFLOAD_ARCHES}" || \
    (echo "ERROR: --build-arg OFFLOAD_ARCHES is required (e.g. gfx942)" >&2; exit 2)
RUN test -n "${THEROCK_TARBALL_NAME}" || \
    (echo "ERROR: --build-arg THEROCK_TARBALL_NAME is required" >&2; exit 2)

# --- system prereqs ----------------------------------------------------
# These cover install_rvs (cmake/build-essential/libpci/libyaml-cpp/doxygen),
# install_transferbench (git, build tools -- HIP comes from TheRock),
# install_ibperf_tools (autotools, libibumad-dev), install_agfhc (python3,
# pyyaml, psutil, libnuma1, numactl), and the docker exec wrapper from P4
# (sudo, bash). One layer for size.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        sudo openssh-client \
        cmake build-essential pkg-config \
        git \
        pciutils kmod \
        libnuma1 numactl \
        python3 python3-pip \
        libpci3 libpci-dev doxygen libyaml-cpp-dev \
        libibverbs1 librdmacm1 libibumad3 \
        libibverbs-dev librdmacm-dev libibumad-dev ibverbs-providers rdma-core \
        autoconf automake libtool \
        openmpi-bin libopenmpi-dev \
        bzip2 \
 && pip3 install --no-cache-dir pyyaml psutil \
 && rm -rf /var/lib/apt/lists/*

# --- TheRock extraction -----------------------------------------------
# TheRock dist tarball already has bin/ lib/ include/ share/ .info/ at its
# root, so it extracts directly into /opt/rocm.
COPY ${THEROCK_TARBALL_NAME} /tmp/therock.tar.gz
RUN mkdir -p /opt/rocm \
 && tar -xf /tmp/therock.tar.gz -C /opt/rocm \
 && rm /tmp/therock.tar.gz

# --- AGFHC compatibility shim -----------------------------------------
# AGFHC reads /opt/rocm/.info/version-utils via cat. TheRock ships
# .info/version but not version-utils. Without this shim AGFHC bails out
# with PROG_TOOL_VERSION_CHECK [97]. (Documented in cvs-config-gen skill.)
RUN cp /opt/rocm/.info/version /opt/rocm/.info/version-utils

# --- RCCL env source script -------------------------------------------
# rccl_singlenode_cvs reads `env_source_script` from its config and bash-
# sources it before launching tests. cvs-config-gen creates this on bare
# metal; we bake it into the image so docker mode doesn't need to.
RUN printf '%s\n' \
        'export ROCM_PATH=/opt/rocm' \
        'export PATH=/opt/rocm/bin:${PATH}' \
        'export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}' \
        > /opt/rocm/env_source_file.sh \
 && chmod +x /opt/rocm/env_source_file.sh

# --- ld.so cache for /opt/rocm/lib ------------------------------------
RUN echo "/opt/rocm/lib" > /etc/ld.so.conf.d/rocm.conf && ldconfig

# --- CVS source for install_* scripts at prepare_runtime time ---------
# Image carries the same CVS install scripts the orchestrator invokes via
# `cvs run install_<name>` -- the wrap from P1/P4 makes those scripts run
# their commands inside the container.
COPY cvs/ /opt/cvs/

# --- AGFHC install dir (empty until install_agfhc populates it) -------
RUN mkdir -p /opt/amd/agfhc

# --- Manifest ----------------------------------------------------------
# P6 reads this for arch validation and threads OFFLOAD_ARCHES into
# install_transferbench/install_rvs as their GPU_TARGETS.
RUN ROCM_VER="$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)" \
 && printf '{\n  "offload_arches": "%s",\n  "rocm_version": "%s",\n  "image_build_time": "%s"\n}\n' \
        "${OFFLOAD_ARCHES}" "${ROCM_VER}" "$(date -u +%FT%TZ)" \
        > /etc/cvs-runner-manifest.json

CMD ["sleep", "infinity"]
