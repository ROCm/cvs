.. meta::
  :description: Configure the preflight checks configuration file
  :keywords: preflight, ROCm, cluster, validation, node smoke

**********************************
Preflight configuration file
**********************************

The preflight checks validate cluster health and configuration consistency before
running performance tests, RCCL training, or inference workloads. Checks include
GPU node health, optional MI4XX scale-up fabric admission, IFoE L2 and
TransferBench gates, RDMA inventory and connectivity, and optional **Node Smoke
Tier 1**, **Tier 2**, and **Tier 3** Primus checks.

Configuration file location: ``cvs/input/config_file/preflight/preflight_config.json``

For the full parameter reference, examples, and troubleshooting, see
`README_preflight_config.md <https://github.com/ROCm/cvs/blob/main/cvs/input/config_file/preflight/README_preflight_config.md>`_
in the repository.

Run preflight checks
====================

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
  ├── reporting/               # HTML report and artifact paths
  └── debug/                   # ScriptLet and troubleshooting options

Legacy keys ``node_smoke`` and ``tier3_info`` are normalized to
``node_smoke_tier1`` and ``node_smoke_tier3`` at load time.

Node Smoke tiers
================

Node Smoke checks are opt-in. Preflight reports each tier separately in the
console summary and HTML report.

.. list-table::
   :header-rows: 1
   :widths: 15 35 20 15

   * - Tier
     - Config
     - Primus command
     - Count (8 GPU)
   * - Tier 1
     - ``node_smoke_tier1.connectivity_mode: "run"``
     - ``node_smoke`` (per node)
     - 39 per node
   * - Tier 2
     - ``node_smoke_tier1.tier2_perf: true``
     - ``node_smoke --tier2-perf`` (per node)
     - 17 per node
   * - Tier 3
     - ``node_smoke_tier3.connectivity_mode: "run"``
     - ``preflight --host --gpu --network`` (cluster)
     - 27 cluster-wide

Tier 1 and Tier 2 counts are **per node** (the summary does not multiply by node
count). Tier 3 is **cluster-wide** (27 collector checks from the validation
tracker, not the 13 aggregated markdown report sections).

Example console output:

.. code:: text

  ✅ Node Smoke Tier 1: PASS - 2/2 nodes passed Node Smoke Tier 1; 39 tests run per node
  ✅ Node Smoke Tier 2: PASS - 2/2 nodes passed Node Smoke Tier 2; 17 tests run per node
  ✅ Node Smoke Tier 3: PASS - 2/2 nodes passed Node Smoke Tier 3; 27 tests run cluster-wide

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
- ``ibv_test_timeout`` / ``ibv_test_port_range`` — ``ibv_rc_pingpong`` test tuning

Node Smoke Tier 1 (``node_smoke_tier1``)
----------------------------------------

- ``connectivity_mode`` — ``"run"`` or ``"skip"`` (default ``"skip"``)
- ``auto_setup`` — Clone Primus and create venv before running
- ``primus_dir`` / ``venv_activate`` — Required when ``connectivity_mode`` is ``"run"``
- ``tier2_perf`` — Enable Node Smoke Tier 2 perf sanity checks
- ``gemm_tflops_min`` / ``hbm_gbs_min`` / ``rccl_gbs_min`` — Tier 2 thresholds

Node Smoke Tier 3 (``node_smoke_tier3``)
----------------------------------------

- ``connectivity_mode`` — ``"run"`` or ``"skip"`` (independent of Tier 1)
- ``primus_dir`` / ``venv_activate`` — Optional; empty inherits from Tier 1
- ``dist_timeout_sec`` — ``torch.distributed`` init timeout
- ``report_file_name`` — Base name for Primus markdown report (default ``node_smoke_tier3``)

Further reading
===============

- :doc:`/reference/cluster/cluster-file` — Cluster topology and SSH
- `cvs/tests/preflight/README.md <https://github.com/ROCm/cvs/blob/main/cvs/tests/preflight/README.md>`_ — Test suite overview and architecture
