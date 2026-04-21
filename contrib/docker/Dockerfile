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
# Stage 1: bootstrap (no cmake yet -- we add Kitware PPA next for newer cmake).
# These cover install_rvs (libpci/libpciaccess/libyaml-cpp/doxygen),
# install_transferbench (git, build tools -- HIP comes from TheRock),
# install_ibperf_tools (autotools, libibumad-dev), install_agfhc (python3,
# pyyaml, psutil, libnuma1, numactl), and the docker exec wrapper from P4
# (sudo, bash).
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg wget \
        sudo openssh-client \
        build-essential pkg-config \
        git \
        pciutils kmod \
        libnuma1 numactl \
        python3 python3-pip \
        libpci3 libpci-dev libpciaccess-dev doxygen libyaml-cpp-dev \
        libibverbs1 librdmacm1 libibumad3 \
        libibverbs-dev librdmacm-dev libibumad-dev ibverbs-providers rdma-core \
        autoconf automake libtool \
        openmpi-bin libopenmpi-dev \
        bzip2 \
 && pip3 install --no-cache-dir pyyaml psutil \
 && rm -rf /var/lib/apt/lists/*

# Stage 2: Kitware PPA cmake. Ubuntu's apt cmake is 3.22 which is too old for
# TheRock 7.13.0's `hipblaslt-config.cmake` (uses cmake `block()` introduced in
# 3.25). cvs-config-gen Phase 1 installs cmake from this same PPA on bare metal;
# we mirror that here so install_rvs source builds succeed inside the container.
RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
        | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" \
        > /etc/apt/sources.list.d/kitware.list \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake \
 && rm -rf /var/lib/apt/lists/* \
 && cmake --version

# Stage 3: cmake 4.x compatibility shim. install_rvs's source build clones
# RVS release/rocm-rel-7.2 which generates an ExternalProject CMakeLists.txt
# for googletest with `cmake_minimum_required(VERSION 2.8.2)`. cmake 4.0
# removed compatibility with `< 3.5`. Setting CMAKE_POLICY_VERSION_MINIMUM=3.5
# in the environment makes cmake honor old policies and lets the nested
# downloader CMakeLists.txt parse. Confirmed (April 2026): with this env var
# in place, install_rvs builds RVS 1.3.0 cleanly inside the container against
# TheRock 7.13.0, matching cvs-config-gen bare-metal results on banff.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# --- TheRock extraction -----------------------------------------------
# TheRock dist tarball already has bin/ lib/ include/ share/ .info/ at its
# root, so it extracts directly into /opt/rocm.
COPY ${THEROCK_TARBALL_NAME} /tmp/therock.tar.gz
RUN mkdir -p /opt/rocm \
 && tar -xf /tmp/therock.tar.gz -C /opt/rocm \
 && rm /tmp/therock.tar.gz

# --- AGFHC compatibility shims ----------------------------------------
# Shim 1: AGFHC reads /opt/rocm/.info/version-utils via cat. TheRock ships
# .info/version but not version-utils. Without this shim AGFHC bails out
# with PROG_TOOL_VERSION_CHECK [97]. (Documented in cvs-config-gen skill.)
RUN cp /opt/rocm/.info/version /opt/rocm/.info/version-utils

# Shim 2: AGFHC's environment-detection step calls `systemd-detect-virt` to
# distinguish bare-metal from VM. Minimal containers don't ship systemd, so
# the call fails and AGFHC bails with PROG_ENV_DETECTION_FAIL [45]. Provide a
# tiny stub that reports "none" (i.e. bare-metal-equivalent) -- accurate
# semantically since the CVS-runner container is --privileged with direct
# /dev/kfd + /dev/dri device pass-through.
RUN printf '#!/bin/sh\necho none\n' > /usr/local/bin/systemd-detect-virt \
 && chmod +x /usr/local/bin/systemd-detect-virt

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

# --- P13: sshd-in-image (multi-node MPI bootstrap) --------------------
# rccl-tests / multi-node MPI launchers need SSH access INTO the
# CVS-runner container on every rank's node. Hosts already run sshd on :22,
# so we expose an in-container sshd on :2222 (host-network mode means this
# is reachable via <hostname>:2222 from any peer). Ephemeral keys are
# pushed at prepare_runtime time -- the image ships only sshd config.
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /var/run/sshd /root/.ssh \
 && chmod 700 /root/.ssh \
 && sed -i 's/^#\?Port .*/Port 2222/'                                /etc/ssh/sshd_config \
 && sed -i 's/^#\?PermitRootLogin .*/PermitRootLogin yes/'           /etc/ssh/sshd_config \
 && sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config \
 && sed -i 's@^#\?AuthorizedKeysFile .*@AuthorizedKeysFile /root/.ssh/authorized_keys@' /etc/ssh/sshd_config \
 && ssh-keygen -A
EXPOSE 2222

# --- P13: container entrypoint ----------------------------------------
# Start sshd in foreground; sleep keeps the container alive if sshd exits
# cleanly. Backwards compatible: docker exec still works for non-MPI tests.
RUN printf '%s\n' \
        '#!/bin/sh' \
        'set -e' \
        '/usr/sbin/sshd' \
        'exec sleep infinity' \
        > /usr/local/bin/cvs-runner-entrypoint.sh \
 && chmod +x /usr/local/bin/cvs-runner-entrypoint.sh

# --- Manifest ----------------------------------------------------------
# P6 reads this for arch validation and threads OFFLOAD_ARCHES into
# install_transferbench/install_rvs as their GPU_TARGETS.
RUN ROCM_VER="$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)" \
 && printf '{\n  "offload_arches": "%s",\n  "rocm_version": "%s",\n  "image_build_time": "%s"\n}\n' \
        "${OFFLOAD_ARCHES}" "${ROCM_VER}" "$(date -u +%FT%TZ)" \
        > /etc/cvs-runner-manifest.json

CMD ["/usr/local/bin/cvs-runner-entrypoint.sh"]
