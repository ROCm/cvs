.. meta::
  :description: Configure RCCL benchmark configuration file variables
  :keywords: RCCL, ROCm, benchmark, CVS

**********************************************************************
ROCm Communication Collectives Library (RCCL) test configuration files
**********************************************************************

RCCL in CVS uses a single suite, ``rccl_cvs``, and a strict nested configuration (this page and the ``Result artifact`` section below). The RCCL JSON describes **benchmark intent** only (ranks, collectives, sweep parameters, validation, artifacts). **Required:** ``rccl.run.env_script`` must point to a site script that exports install locations and library paths (see ``Environment variables`` below). NCCL, UCX, plugin, and site-specific tuning belong in that script only — not in the JSON.

The file must contain **only** the top-level key ``rccl`` (no sibling keys). Unsupported legacy fields—``mode``, ``results`` / ``results_file``, flat run fields, or any key not defined in this schema—are rejected at load time.

RCCL test suite
===============

CVS provides one RCCL suite:

.. list-table::
   :widths: 3 7
   :header-rows: 1

   * - Test suite
     - What it does
   * - ``rccl_cvs``
     - Runs an RCCL benchmark flow inferred from rank layout, performs lightweight preflight, validates rccl-tests JSON output according to ``rccl.validation.profile``, and writes run output under ``rccl.artifacts.output_dir`` (see Result artifact below).

How to run
==========

From the CVS repo root:

.. code-block:: bash

  cvs list rccl_cvs

Run RCCL with the nested ``rccl`` config:

.. code-block:: bash

  cvs run rccl_cvs \
      --cluster_file input/cluster_file/cluster.json \
      --config_file input/config_file/rccl/rccl_config.json \
      --html=/var/www/html/cvs/rccl.html --capture=tee-sys --self-contained-html \
      --log-file=/tmp/rccl.log -vvv -s

.. note::

  Update ``rccl.validation.thresholds`` (or the file referenced by ``thresholds_file``) for your platform before relying on pass/fail for ``thresholds`` or ``strict`` profiles.
  Placeholders such as ``{user-id}`` in config paths are resolved by CVS at runtime.

Topology (no ``mode`` field)
=============================

Single-node versus multi-node is **inferred** from ``rccl.run.num_ranks`` and ``rccl.run.ranks_per_node``:

- ``required_nodes = num_ranks / ranks_per_node`` (integer division, remainder must be zero).
- ``required_nodes == 1``: local shell launch (no ``mpirun``).
- ``required_nodes > 1``: MPI hostfile launch. ``rccl.run.env_script`` is **always** required; the script must export ``MPI_HOME`` (MPI install prefix) so the runner can invoke ``${MPI_HOME}/bin/mpirun``.

The cluster file must list at least ``required_nodes`` nodes; the runner uses the first nodes in stable file order.

Environment variables (``env_script`` only)
============================================

The runner sources ``env_script`` before each benchmark. It does **not** read ROCm, MPI, or rccl-tests paths from JSON. Configure the shell environment in that script.

Normative variables the runner or benchmarks expect (export what applies on your site):

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Variable
     - Role
   * - ``RCCL_TESTS_BUILD_DIR``
     - Directory containing rccl-tests ``*_perf`` binaries (runner checks executability at launch).
   * - ``ROCM_HOME``
     - ROCm root; use to extend ``PATH`` / ``LD_LIBRARY_PATH`` in the script.
   * - ``MPI_HOME``
     - MPI install prefix (directory that contains ``bin/mpirun``). **Required for multi-node** runs.
   * - ``RCCL_HOME``
     - Typical site location for RCCL libraries; add to ``LD_LIBRARY_PATH`` in the script when needed.

If a required variable is unset when the benchmark runs, the shell fails fast with a clear error.

NIC-specific example env scripts
================================

Under ``input/config_file/rccl/env/`` the tree ships three **separate** starter scripts (no shared includes):

- ``cx7_env_script.sh`` — ConnectX-7 class InfiniBand baseline.
- ``thor2_env_script.sh`` — Thor2-class NIC baseline (same skeleton as cx7 until Thor2-specific flags are needed).
- ``ainic_env_script.sh`` — AINIC + ANP plugin starter, including the same required baseline path exports plus plugin-specific variables.

Copy the script that matches your NIC to a stable path on the target nodes, edit the values, and set ``rccl.run.env_script`` to that path.

``rccl_config.json``
====================

The top-level object must contain **only** the key ``rccl`` (object) with three required sections: ``run``, ``validation``, and ``artifacts``. Optional ``matrix`` may be omitted, null, or ``{}``; **non-empty** matrix objects are rejected by this CVS build until matrix expansion is implemented.

.. dropdown:: ``rccl_config.json`` (example — paths only via ``env_script``)

  ``env_script`` must ``export RCCL_TESTS_BUILD_DIR``, ``ROCM_HOME``, ``MPI_HOME`` (for multi-node), and extend ``LD_LIBRARY_PATH`` using ``RCCL_HOME`` when needed. The ``run`` object contains **no** path fields.

  .. code:: json

    {
      "rccl": {
        "run": {
          "env_script": "/root/env_source_file.sh",
          "num_ranks": 16,
          "ranks_per_node": 8,
          "collectives": ["all_reduce_perf", "all_gather_perf", "broadcast_perf"],
          "datatype": "float",
          "start_size": "1024",
          "end_size": "16g",
          "step_factor": "2",
          "warmups": "10",
          "iterations": "20",
          "cycles": "1"
        },
        "validation": {
          "profile": "strict",
          "thresholds": {
            "all_reduce_perf": { "bus_bw": { "8589934592": 330.0, "17179869184": 350.0 } },
            "all_gather_perf": { "bus_bw": { "8589934592": 330.0, "17179869184": 350.0 } },
            "broadcast_perf": { "bus_bw": { "8589934592": 310.0, "17179869184": 312.0 } }
          }
        },
        "artifacts": {
          "output_dir": "/tmp/rccl_cvs_out",
          "export_raw": false
        }
      }
    }

``rccl.run``
------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Example
     - Description
   * - ``env_script``
     - ``/root/env_source_file.sh``
     - **Required.** Script sourced before benchmarks. Must export ``RCCL_TESTS_BUILD_DIR`` and ROCm/MPI paths via ``ROCM_HOME``, ``MPI_HOME`` (multi-node), ``RCCL_HOME`` / ``LD_LIBRARY_PATH`` as needed. NCCL, UCX, and plugin tuning belong here.
   * - ``num_ranks``
     - ``16`` (JSON integer)
     - Total MPI ranks; must be ≥ 1 and divisible by ``ranks_per_node`` (string numerals are coerced, but prefer integers in configs).
   * - ``ranks_per_node``
     - ``8`` (JSON integer)
     - Ranks per node; must be ≥ 1.
   * - ``collectives``
     - ``["all_reduce_perf"]``
     - Non-empty array of binary basenames (no path separators).
   * - ``datatype``
     - ``float``
     - Single rccl-tests datatype.
   * - ``start_size`` / ``end_size``
     - ``1024`` / ``16g``
     - Message size sweep bounds (rccl-tests ``-b`` / ``-e``).
   * - ``step_factor``
     - ``2``
     - Size multiplier (rccl-tests ``-f``).
   * - ``warmups`` / ``iterations`` / ``cycles``
     - ``10`` / ``20`` / ``1``
     - Warmups, measured iterations, cycles (``-w`` / ``-n`` / ``-N``).

``rccl.validation``
-------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Example
     - Description
   * - ``profile``
     - ``smoke``
     - One of ``none``, ``smoke``, ``thresholds``, ``strict``. See profile semantics below.
   * - ``thresholds``
     - object
     - Optional inline per-collective map (only with ``profile`` ``thresholds`` or ``strict``). At most one of ``thresholds`` and ``thresholds_file``.
   * - ``thresholds_file``
     - ``/path/to/baseline.json``
     - Optional path to JSON with the same shape as ``thresholds``.

**Profile semantics** (after successful command execution):

- ``none``: JSON array must parse; no row schema validation; no threshold or dip checks.
- ``smoke``: parse plus Pydantic/schema validation on rows.
- ``thresholds``: smoke plus minimum ``busBw`` vs thresholds for sizes present in the map.
- ``strict``: thresholds plus bandwidth-dip and latency-dip checks (for sizes present in the threshold map), matching the previous strict behavior.

For ``thresholds`` or ``strict``, exactly one of ``thresholds`` or ``thresholds_file`` must be present and non-empty after load. Threshold keys must correspond to entries in ``rccl.run.collectives`` (no extra collectives, and no legacy ``results`` / ``results_file`` keys). For ``none`` or ``smoke``, do not supply ``thresholds`` or ``thresholds_file``.

**Threshold JSON shape:**

.. code:: json

  {
    "all_reduce_perf": {
      "bus_bw": {
        "8589934592": 330.0
      }
    }
  }

``rccl.artifacts``
------------------

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Field
     - Example
     - Description
   * - ``output_dir``
     - ``/tmp/rccl_cvs_out``
     - Base directory for this invocation’s run folder (see Result artifact).
   * - ``export_raw``
     - ``false``
     - If ``true``, optional per-case raw rccl-tests JSON is written under ``{output_dir}/{run_id}/raw/<case_id>.json``. ``run.json`` remains authoritative; raw files are for debugging only.

Result artifact
===============

Each ``cvs run rccl_cvs`` invocation creates one UTC-named run directory and a single canonical ``run.json`` inside it:

- **Directory:** ``{rccl.artifacts.output_dir}/{run_id}/`` where ``run_id`` is like ``2026-04-03T10-15-00Z`` (filesystem-safe ISO time).
- **Canonical file:** ``run.json`` with ``schema_version`` ``rccl_cvs.run.v1``, topology, cluster selection, echoed config, filtered environment variables (after ``env_script``), per-collective ``cases[]`` (including ``case_id``, ``resolved``, ``metrics.rows``, and validation sub-results), and a ``summary`` roll-up.

There is **no** separate ``summary.json``; roll-up lives in ``run.json`` under ``summary``. Optional raw rccl-tests JSON is written only when ``export_raw`` is ``true``, under ``raw/<case_id>.json`` in the same run directory. Downstream tools must not depend on raw files for pass/fail or metrics; do not treat legacy per-suite ``output_json`` paths as authoritative.

AINIC/ANP net plugin setup (optional)
=====================================

To run with AINIC + ANP plugin:

1. Ensure AMD ANP is installed and available on all target nodes.
2. Edit ``input/config_file/rccl/env/ainic_env_script.sh`` and set ``ANP_HOME_DIR`` (and library paths) for your install layout.
3. Point ``rccl.run.env_script`` at your copy of that script on the nodes.
4. Run ``cvs run rccl_cvs ...``; CVS sources the script before benchmark execution.

Collective meanings
-------------------

- ``all_reduce_perf``: all ranks reduce then receive the reduced result.
- ``all_gather_perf``: each rank receives data from all ranks.
- ``scatter_perf``: root rank distributes shards to all ranks.
- ``gather_perf``: all ranks send data to a root rank.
- ``reduce_scatter_perf``: reduction followed by scatter.
- ``sendrecv_perf``: point-to-point pair communication.
- ``alltoall_perf``: equal-sized all-to-all exchange.
- ``alltoallv_perf``: variable-sized all-to-all exchange.
- ``broadcast_perf``: one-to-all data broadcast.
