.. meta::
  :description: Reference for the CVS preflight checks configuration file, covering GPU node health, RDMA connectivity, InfiniBand, and Node Smoke tier parameters.
  :keywords: CVS, preflight, ROCm, cluster, validation, node smoke, RDMA, InfiniBand, GPU, AMD, JSON, IB, SSH

********************************************************************************
Preflight checks configuration file reference for Cluster Validation Suite (CVS)
********************************************************************************

The preflight checks validate cluster health and configuration consistency before
running performance tests, RCCL training, or inference workloads. Checks include
GPU node health, optional MI4XX scale-up fabric admission, IFoE L2 and
TransferBench gates, RDMA inventory and connectivity, and **Node Smoke
Tier 1**, **Tier 2**, and **Tier 3** Primus checks. The Node Smoke tiers run
by default. Set ``connectivity_mode`` to ``"skip"`` (and ``tier2_perf`` to
``false`` for Tier 2) to disable one.

Configuration file location: |cvs-input-config-file-preflight-preflight-config-json|_

For the full parameter reference, examples, and troubleshooting, see
`README_preflight_config.md <https://github.com/ROCm/cvs/blob/main/cvs/input/config_file/preflight/README_preflight_config.md>`_
in the repository.

Run preflight checks
====================

Use the following commands to run the full preflight suite or individual Node Smoke tiers.

.. code:: bash

  cvs run preflight_checks \
    --cluster_file cluster.json \
    --config_file cvs/input/config_file/preflight/preflight_config.json

Run individual Node Smoke tiers:

.. code:: bash

  cvs run preflight_checks test_node_smoke_tier1 \
    --cluster_file cluster.json \
    --config_file preflight_config.json

  cvs run preflight_checks test_node_smoke_tier3 \
    --cluster_file cluster.json \
    --config_file preflight_config.json

Configuration structure
=======================

.. code:: text

  preflight/
  ├── node_check/              # GPU visibility, ROCm, optional MI4XX fabric
  ├── connectivity_check/
  │   ├── rdma/                # GID, interfaces, ibv_rc_pingpong connectivity
  │   └── ifoe/                # L2 ping and TransferBench smoketest
  ├── node_smoke_tier1/        # Node Smoke Tier 1/2 (Primus node_smoke)
  ├── node_smoke_tier3/        # Node Smoke Tier 3 (Primus preflight --host --gpu --network)
  ├── host_readiness/          # Host configuration, environment, idle state, inventory
  ├── reporting/               # HTML report and artifact paths
  └── debug/                   # ScriptLet and troubleshooting options

Legacy keys ``node_smoke`` and ``tier3_info`` are normalized to
``node_smoke_tier1`` and ``node_smoke_tier3`` at load time.

Node Smoke tiers
================

Node Smoke checks run by default. Set a tier's ``connectivity_mode`` to
``"skip"`` to disable it, or set ``node_smoke_tier1.tier2_perf`` to ``false``
to disable Tier 2 only. Preflight reports each tier separately in the
console summary and HTML report. The console count is the number of checks
Primus reported, not a fixed catalog size.

.. list-table::
   :header-rows: 1
   :widths: 15 35 20 15

   * - Tier
     - Config
     - Primus command
     - Count (8 GPU)
   * - Tier 1
     - ``node_smoke_tier1.connectivity_mode: "run"`` (default)
     - ``node_smoke`` (per node)
     - one row per GPU plus 7 collectors
   * - Tier 2
     - ``node_smoke_tier1.tier2_perf: true`` (default)
     - ``node_smoke --tier2-perf`` (per node)
     - GEMM and HBM per GPU, plus local RCCL
   * - Tier 3
     - ``node_smoke_tier3.connectivity_mode: "run"`` (default)
     - ``preflight --host --gpu --network`` (cluster)
     - host, GPU, and network groups

Tier 1 and Tier 2 counts are **per node** (the summary does not multiply by node
count). Tier 3 is **cluster-wide**. A row is passed only when Primus reported
that check. A node-level pass does not mark unreported checks as passed.

Example console output:

.. code:: text

  ✅ Node Smoke Tier 1: PASS - 2/2 nodes passed Node Smoke Tier 1; 15 tests run per node
  ✅ Node Smoke Tier 2: PASS - 2/2 nodes passed Node Smoke Tier 2; 17 tests run per node
  ✅ Node Smoke Tier 3: PASS - 2/2 nodes passed Node Smoke Tier 3; 3 tests run cluster-wide

Sample configuration
====================

.. note::

  In this configuration file, ``{user-id}`` is resolved to the current username
  at runtime.

.. dropdown:: Minimal Node Smoke Tier 1 + Tier 3

  .. code:: json

    {
      "preflight": {
        "node_smoke_tier1": {
          "connectivity_mode": "run",
          "auto_setup": true,
          "primus_dir": "/home/{user-id}/INSTALL/Primus",
          "venv_activate": "/home/{user-id}/envs/preflight/.venv/bin/activate",
          "gpus_per_node": 8
        },
        "node_smoke_tier3": {
          "connectivity_mode": "run",
          "ssh_timeout": 600
        },
        "reporting": {
          "artifacts_root_dir": "/home/{user-id}/preflight"
        }
      }
    }

Key parameters
==============

Node check (``node_check``)
---------------------------

- ``enabled`` — GPU visibility, AMDGPU/KFD, kernel health, ROCm validation
- ``gpus_per_node`` — Expected GPU count on every node
- ``expected_rocm_version`` — ROCm version string (must match ``amd-smi version``)

RDMA connectivity (``connectivity_check.rdma``)
-----------------------------------------------

- ``connectivity_mode`` — ``"basic"``, ``"full_mesh"``, or ``"skip"``
- ``interfaces`` — Expected RDMA device names on all nodes
- ``gid_index`` — GID index validated on those interfaces
- ``gid_type`` — Expected sysfs GID type (default ``"RoCE v2"``); ``"any"`` disables type checking, and InfiniBand link layers are exempt
- ``mtu_check`` — Enable RDMA active MTU validation, plus jumbo-frame netdev MTU validation on RoCE ports (default ``true``)
- ``min_netdev_mtu`` / ``min_active_mtu`` — Minimum netdev MTU on RoCE ports (default ``4200``; InfiniBand ports are exempt) and active MTU on all ports (default ``4096``); ``0`` disables either threshold
- ``ibv_test_timeout`` / ``ibv_test_port_range`` — ``ibv_rc_pingpong`` test tuning

Node Smoke Tier 1 (``node_smoke_tier1``)
----------------------------------------

- ``connectivity_mode`` — ``"run"`` or ``"skip"`` (default ``"run"``)
- ``auto_setup`` — Clone Primus and create venv before running
- ``primus_dir`` / ``venv_activate`` — Required when ``connectivity_mode`` is ``"run"``
- ``tier2_perf`` — Node Smoke Tier 2 perf sanity checks (default ``true``)
- ``gemm_tflops_min`` / ``hbm_gbs_min`` / ``rccl_gbs_min`` — Tier 2 thresholds

Node Smoke Tier 3 (``node_smoke_tier3``)
----------------------------------------

- ``connectivity_mode`` — ``"run"`` (default) or ``"skip"`` (independent of Tier 1)
- ``primus_dir`` / ``venv_activate`` — Optional; empty inherits from Tier 1
- ``dist_timeout_sec`` — ``torch.distributed`` init timeout
- ``report_file_name`` — Base name for Primus markdown report (default ``node_smoke_tier3``)

Host readiness (``host_readiness``)
-----------------------------------

Host configuration and environment checks run right after node health, before
any preflight workload touches the GPUs. Each section is optional. A section
that is absent, or has ``enabled: false``, is reported as SKIPPED, so existing
configuration files keep working. Every section accepts:

- ``enabled`` — Run this check (default ``true`` once the section is present)
- ``on_mismatch`` — ``"fail"`` (default) fails a node on a configuration gap;
  ``"warn"`` reports it as a warning. A node that cannot be read (missing tool,
  refused ``sudo -n``, unreadable file) always fails.

``idle_state`` (``test_gpu_idle_state``):

- ``samples`` / ``interval_sec`` — ``amd-smi metric`` samples per node and the
  gap between them. Utilization uses the lowest sample; temperature uses the highest.
- ``gfx_util_max_pct`` / ``mem_util_max_pct`` / ``vram_used_max_pct`` — Idle limits; ``null`` skips one
- ``temp_max_c`` — Idle temperature limit per ``amd-smi`` sensor (``hotspot``,
  ``mem``, ``edge``); sensors that report N/A are skipped
- ``fail_on_gpu_processes`` / ``allowed_procs`` — Fail on any GPU process not in the comma-separated allowlist

``kernel_params`` (``test_host_kernel_params``):

- ``required`` — ``key=value`` or bare ``key`` tokens that must be on ``/proc/cmdline``.
  Comma-list parameters such as ``pci=`` and ``modprobe.blacklist=`` match per
  option, so ``pci=realloc=off,bfsort`` satisfies both ``pci=realloc=off`` and ``pci=bfsort``.
- ``required_intel`` / ``required_amd`` — Extra tokens for hosts with that CPU vendor
- ``forbidden`` — Tokens that must not be present

When ``numa_balancing`` is required, ``kernel.numa_balancing`` must also be ``0``
at runtime. When ``amdgpu`` is blacklisted on the command line, the module must
still be loaded after boot.

``os_tuning`` (``test_host_os_tuning``):

- ``disable_idle_states_from`` — Every cpuidle state at this index or deeper must
  be disabled on all CPUs (``cpupower idle-set -d <index>``); ``null`` skips
- ``max_cstate_params`` — Optional kernel tokens such as ``processor.max_cstate=1``
- ``cpu_governor`` — Required ``scaling_governor`` on every CPU; ``null`` skips

``env`` (``test_host_persistent_env``):

- ``required`` — Variables and values that must be set persistently
- ``sources`` — Files, or directories of ``*.sh`` files, read statically for the
  definitions (default ``/etc/environment``); the last definition wins
- ``check_session`` — Also compare the value a remote command sees. With the
  container orchestrator, the value inside the container is checked too.

``inventory`` (``test_host_inventory``):

- ``sku_profiles`` — List of named profiles with ``nodes``, ``dmi_product_name``,
  ``cpu_sockets``, ``min_cpu_cores``, ``dimm_count``, ``dimm_size_gb``,
  ``nvme_count``, ``nvme_min_size_gb``, ``frontend_nic_count``,
  ``frontend_nic_interfaces``, and ``min_ram_gb``. A ``null`` field is recorded but not checked.
- ``default_profile`` — Profile for nodes that no ``nodes`` list or ``dmi_product_name`` matches
- ``export_json`` — Write per-node inventory JSON under
  ``<reporting.artifacts_root_dir>/host_inventory_<timestamp>/``

A node uses the profile that lists it in ``nodes``, then the profile whose
``dmi_product_name`` matches ``/sys/class/dmi/id/product_name``, then
``default_profile``. With no profiles, the inventory is only recorded and
exported. DIMM checks need passwordless ``sudo dmidecode``; without it, DIMM
counts fall back to EDAC sysfs. Frontend NICs are physical, non-VF interfaces that are
not listed in ``connectivity_check.rdma.interfaces``. When that list is empty,
RDMA-capable interfaces are treated as backend.

``platform_extras`` (``test_host_platform_extras``):

- ``large_bar`` — CPU-visible VRAM must equal total VRAM on every GPU
- ``atomic_ops`` — PCIe AtomicOps requests must be enabled on every GPU (needs passwordless ``sudo lspci``)
- ``cwsr_enable`` — Required ``amdgpu`` ``cwsr_enable`` value; ``null`` records it only
- ``limits`` — ``memlock`` and ``nofile_min`` for the SSH user, resolved from
  ``/etc/security/limits.conf`` and ``limits.d`` with ``pam_limits`` precedence; ``null`` skips
- ``rdma_modules`` — Kernel modules that must be loaded; an empty list skips the RDMA check
- ``rocm_path`` / ``rocm_lib_deps`` — Optional ``ldd`` scan of ``<rocm_path>/lib``
  for unresolved dependencies (``enabled``, ``max_depth``, ``timeout_sec``)

Related resources
=================

- :doc:`/reference/cluster/cluster-file` — Cluster topology and SSH
- |cvs-tests-preflight-readme|_ — Test suite overview and architecture


.. |cvs-input-config-file-preflight-preflight-config-json| replace:: ``cvs/input/config_file/preflight/preflight_config.json``
.. _cvs-input-config-file-preflight-preflight-config-json: https://github.com/ROCm/cvs/blob/main/cvs/input/config_file/preflight/preflight_config.json

.. |cvs-tests-preflight-readme| replace:: ``cvs/tests/preflight/README.md``
.. _cvs-tests-preflight-readme: https://github.com/ROCm/cvs/blob/main/cvs/tests/preflight/README.md
