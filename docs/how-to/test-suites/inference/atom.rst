.. meta::
  :description: Run ATOM inference benchmarks with CVS, single-node and multinode
  :keywords: CVS, ATOM, inference, benchmark, multinode, LLM, ROCm

******************************
Run ATOM inference benchmarks
******************************

The ATOM suite benchmarks LLM serving on AMD Instinct GPUs using the ATOM stack
(``atom.entrypoints.openai_server`` + ``atom.benchmarks.benchmark_serving`` on
single-node variants, or a PP coordinator with ``params.driver: vllm_atom`` /
``sglang`` on multinode stems). One suite name ΓÇö ``atom`` ΓÇö covers single-node
and multinode pipeline-parallel runs; topology comes from the config and cluster
file.

This page walks through a first run. For the full schema, profiles, thresholds,
and parameter reference, see :doc:`/reference/configuration-files/inference/atom`.

Prerequisites
=============

On every **GPU node**:

- Docker with ROCm device passthrough (``sudo docker`` or docker group).
- ATOM (or vLLM-ATOM / SGLang) container image available on the node.
- Model weights at ``paths.models_dir`` when ``model.remote: 0``.
- Shared log path mounted when using multinode PP.

On the **launcher** (where you run ``cvs run``):

- CVS installed (see :doc:`/getting-started/install`).
- SSH key access to cluster nodes (``priv_key_file`` in the cluster file).
- Hugging Face token file at ``paths.hf_token_file`` when required.

On **Spur or Slurm managed compute**:

- Launch CVS inside a **job step** (one task per node), for example
  ``spur run --mpi=none`` or ``srun --mpi=none``. A bare allocation or a
  ``spur submit`` / ``sbatch`` script that never starts a step is not
  managed CVS (``SLURM_STEP_ID`` stays unset).
- Use the managed cluster file produced from ``SPUR_NODES`` / HTTP agents.
  ATOM still uses the ``orch`` fixture and the config ``container`` block;
  do not add nested ``spur run`` inside ``AtomJob``.
- Prove an existing single-node stem first, then run the ``mi355x_atom_*``
  stems (Kimi TP4, V4-Pro TP8). V4-Flash-Base stays on gfx942.

Spur example:

.. code:: bash

  spur run -A <account> -p <partition> \
    -N 1 --gpus-per-node 8 --exclusive -t 04:00:00 --mpi=none \
    bash -lc 'source ~/.cvs_venv/bin/activate &&
      cvs run atom --config_file <atom-config.json> --html <report.html>'

Slurm example:

.. code:: bash

  srun -A <account> -p <partition> \
    -N 1 --gpus-per-node 8 --exclusive -t 04:00:00 --mpi=none \
    bash -lc 'source ~/.cvs_venv/bin/activate &&
      cvs run atom --config_file <atom-config.json> --html <report.html>'

In managed mode, ``cvs run`` is the **job step on the GPU node**.
``--cluster_file`` is optional. CVS builds the live cluster file from
scheduler hosts and starts one HTTP agent per scheduler task.

``make install`` needs ``python3-venv`` (``ensurepip``). Scheduler **login**
nodes often lack it; do not ``apt install`` packages there. Pull the tree
on the shared home from login, then create ``.cvs_venv`` **inside** the job
step on the GPU node. Deactivate any existing venv first so Make does not
bind ``PYTHON`` to ``.cvs_venv/bin/python3`` and then delete that path.

.. code:: bash

  # on login: git pull only. deactivate before make install.
  spur run -A <account> -p <partition> \
    -N 1 --gpus-per-node 8 --exclusive -t 04:00:00 --mpi=none \
    bash -lc 'export PATH=/usr/bin:/bin:$PATH
      cd ~/cvs && make install &&
      source ~/.cvs_venv/bin/activate &&
      cvs run atom --config_file <atom-config.json> --html <report.html>'

MI355X Spur stems (same job-step pattern)
-----------------------------------------

On gfx950, copy **Kimi TP4** then **V4-Pro TP8** into **separate** directories
so each ``--config_file`` sees only its matching ``*threshold.json``. Both
stems use ``driver=atom``, ``lifetime: per_run``, and a writable
``HF_HUB_CACHE`` / ``HF_HOME`` under ``paths.shared_fs`` (not the read-only
``/models`` mount). Do not run Flash-Base here.

.. code:: bash

  KIMI_DIR=~/input/config_file/inference/atom/kimi355
  PRO_DIR=~/input/config_file/inference/atom/pro355
  mkdir -p "$KIMI_DIR" "$PRO_DIR"

  cvs config copy inference/atom/mi355x_atom_kimi-k27-code_mxfp4_single.json \
    --output "$KIMI_DIR/mi355x_atom_kimi-k27-code_mxfp4_single.json"
  cvs config copy inference/atom/mi355x_atom_kimi-k27-code_mxfp4_single_threshold.json \
    --output "$KIMI_DIR/mi355x_atom_kimi-k27-code_mxfp4_single_threshold.json"

  cvs config copy inference/atom/mi355x_atom_deepseek-v4-pro_single.json \
    --output "$PRO_DIR/mi355x_atom_deepseek-v4-pro_single.json"
  cvs config copy inference/atom/mi355x_atom_deepseek-v4-pro_single_threshold.json \
    --output "$PRO_DIR/mi355x_atom_deepseek-v4-pro_single_threshold.json"

In each copy set ``container.image``, the host side of the models volume,
``paths.shared_fs``, and ``model.id`` to the in-container weights path when
``model.remote`` is ``0`` (for example ``/models/<local-folder>``). Keep
``tensor_parallelism`` / ``-tp`` at **4** for Kimi and **8** for V4-Pro.

Launch Kimi first, then Pro, with the same ``spur run --mpi=none`` wrapper
(one exclusive 8-GPU node). Run the **full** suite: accuracy eval needs the
server started by earlier lifecycle tests; ``-k test_accuracy_eval`` alone
fails with ``docker exec None``.

.. code:: bash

  spur run -A <account> -p <partition> \
    -N 1 --gpus-per-node 8 --exclusive -t 04:00:00 --mpi=none \
    bash -lc 'source ~/.cvs_venv/bin/activate &&
      cvs run atom \
        --config_file ~/input/config_file/inference/atom/kimi355/mi355x_atom_kimi-k27-code_mxfp4_single.json \
        --html ~/cvs_reports/atom_kimi.html --self-contained-html'

Then the same command with
``~/input/config_file/inference/atom/pro355/mi355x_atom_deepseek-v4-pro_single.json``
and ``atom_v4pro.html``.

For **multinode PP** (``params.nnodes: 2``, ``pipeline_parallel_size: 2``):

- Two hosts in ``node_dict`` matching ``params.nnodes``.
- ``params.master_addr`` set to the head node VPC IP.
- IB/socket fabric discoverable, or explicit ``roles.server.ib_hca_devices`` /
  ``roles.server.ib_netdev`` (socket netdev must **not** be an ``mlx5_*`` HCA name).

Step 1: Copy config and cluster files
=====================================

List shipped ATOM configs:

.. code:: bash

  cvs config list inference/atom

Copy each variant into its **own subdirectory** so only one ``*threshold.json``
sits beside the config you pass to ``--config_file``:

.. code:: bash

  SINGLE_DIR=~/input/config_file/inference/atom/single
  mkdir -p "$SINGLE_DIR"

  cvs config copy inference/atom/mi3xx_atom_deepseek-r1_fp8_single.json \
    --output "$SINGLE_DIR/mi3xx_atom_deepseek-r1_fp8_single.json"
  cvs config copy inference/atom/mi325x_atom_deepseek-r1_fp8_single_threshold.json \
    --output "$SINGLE_DIR/mi325x_atom_deepseek-r1_fp8_single_threshold.json"
  cvs config copy cluster_file/atom_cluster.json --output ~/input/cluster_file/atom_cluster.json

Config stems use ``mi3xx_*`` (gfx942) or ``mi355x_*`` (gfx950). Threshold
files use ``mi325x_*`` or ``mi355x_*`` — match the platform you calibrated on.

Step 2: Edit placeholders
=========================

Replace cluster node IPs and trim ``node_dict`` to one host for single-node runs
(two hosts for distributed PP). In the config, set at minimum:

- ``container.image`` — your ATOM ROCm image.
- ``container.runtime.args.volumes`` — replace ``<changeme-models-mount>`` with the
  host models directory (for example ``/it-share-prj2-1/models`` on prj2 lab nodes).
- ``paths.shared_fs``, ``paths.log_dir``, ``paths.hf_token_file``.
- ``model.id`` — model under test.

``paths.models_dir`` is ``/models``, the in-container weights mount (usually
read-only). Keep it as shipped; only customize the host side of the models
volume. ATOM still defaults ``HF_HUB_CACHE`` to that path for Hub weight
lookups. SGLang ``bench_serving`` with ``dataset_name=random`` also downloads
ShareGPT into ``HF_HUB_CACHE``, so Qwen SGLang samples override
``HF_HUB_CACHE`` / ``HF_HOME`` to ``{paths.shared_fs}/.cache/huggingface``.

For multinode PP, also set ``params.master_addr`` and verify
``roles.server.ib_netdev`` (``"auto"`` is the default on shipped distributed stems).

.. tip::

  Leave ``enforce_thresholds: false`` on first lab runs until thresholds are
  calibrated for your hardware. MI355X shipped stems often ship record-only.

Step 3: Run the suite
=====================

Single-node W1 (``driver=atom``):

.. code:: bash

  cvs run atom \
    --cluster_file ~/input/cluster_file/atom_cluster.json \
    --config_file "$SINGLE_DIR/mi3xx_atom_deepseek-r1_fp8_single.json" \
    --html ~/cvs_results/atom-w1-single.html --self-contained-html -vvv

Multinode PP (``driver=vllm_atom``):

.. code:: bash

  cvs run atom \
    --cluster_file ~/input/cluster_file/atom_cluster.json \
    --config_file ~/input/config_file/inference/atom/distributed/mi3xx_atom_vllm_deepseek-r1_fp8_distributed.json \
    --html ~/cvs_results/atom-w1-distributed.html --self-contained-html -vvv

MTP-3 speculative decode (``schema_version: 2`` profile on a native single-node
stem — DeepSeek R1 or Qwen FP8):

.. code:: bash

  cvs run atom \
    --cluster_file ~/input/cluster_file/atom_cluster.json \
    --config_file "$SINGLE_DIR/mi3xx_atom_deepseek-r1_fp8_single.json" \
    --config_profile mtp3 \
    --html ~/cvs_results/atom-w1-mtp3.html --self-contained-html -vvv

vLLM / SGLang parity use the unified serving schema in ``inference/atom/``
(for example ``mi3xx_atom_vllm_deepseek-r1_fp8_single.json``) — still run with
``cvs run atom``, not ``cvs run vllm`` or ``cvs run sglang``.

Smoke one cell with pytest ``-k``, for example ``-k "w1_1k_1k-conc128"``.

After ``git pull``, run ``make install`` before ``source .cvs_venv/bin/activate``.
On Spur or Slurm, do that on the GPU node inside the job step if login cannot
create a venv (see the managed-compute notes above).

Test lifecycle
==============

Tests run in a fixed order. ``[cell]`` = one row per sweep cell; ``[cell-tier]`` =
one row per metric tier per cell.

.. list-table::
   :widths: 1 3 6
   :header-rows: 1

   * - Order
     - Test
     - Purpose
   * - 1
     - ``test_launch_container``
     - Launch and verify the container
   * - 2
     - ``test_setup_sshd``
     - Multinode SSH setup (skipped on single-node)
   * - 3
     - ``test_discover_topology``
     - Resolve IB HCAs and socket netdev
   * - 4
     - ``test_model_fetch``
     - Verify model cache on nodes
   * - 5
     - ``test_atom_inference[cell]``
     - Server start (or reuse), bench client, parse results
   * - 6
     - ``test_cell_metrics[cell-tier]``
     - Threshold PASS/FAIL per metric tier
   * - 7
     - ``test_print_results_table``
     - Console summary tables
   * - 8
     - ``test_teardown``
     - Tear down the container

On inference failure, ``lifecycle.failed`` skips downstream cells and metric
rows. When ``reuse_server_across_sweep: true``, a warm server is kept across cells
with the same ``server_session_key``.

Sweeps and metrics
==================

Each **sweep cell** is one ``(ISL, OSL, concurrency)`` pair from ``sweep.runs``.
Parametrize IDs look like ``w1_1k_1k-conc128`` or ``w1_1k_1k-conc128-throughput``.

Threshold **cell keys** (must match the sibling threshold file):

- Single-node: ``ISL=1024,OSL=1024,TP=8,PP=1,CONC=128``
- Multinode PP: ``ISL=1024,OSL=1024,TP=8,PP=2,CONC=128``

Each ``test_cell_metrics[cell-tier]`` gates one tier when ``enforce_thresholds:
true``:

.. list-table::
   :widths: 2 6
   :header-rows: 1

   * - Tier
     - Example metrics (bare names in thresholds)
   * - ``throughput``
     - ``output_throughput``, ``per_gpu_throughput``, ΓÇª
   * - ``ttft`` / ``tpot``
     - ``mean_ttft_ms``, ``p99_ttft_ms``, ΓÇª
   * - ``health``
     - ``success_rate``, ``failed``
   * - ``scaling``
     - ``scaling.efficiency_pct`` (multinode)
   * - ``record``
     - Remaining metrics ΓÇö logged, not gated

Benchmark artifacts still expose ``client.*`` keys internally; threshold JSON
uses bare metric names. ATOM may omit tail percentiles even when
``metric_percentiles`` requests them ΓÇö only present metrics are gated.

Reports and logs
================

- **pytest HTML** ΓÇö one row per lifecycle stage and per metric tier.
- **Console tables** ΓÇö ``test_print_results_table`` prints per-cell throughput and latency.
- **Run Deck** ΓÇö when ``--html`` is set, ``atom_run_deck.html`` / ``.json`` /
  ``_viewer.html`` are bundled into the pytest zip (render-only; does not affect gates).
- **Per-cell logs** ΓÇö under ``paths.log_dir`` on cluster nodes (server + client logs).

Launcher vs GPU node
====================

SSH / jumphost labs: ``cvs run`` and the venv live on the **launcher**. Spur or
Slurm managed labs: ``cvs run`` and ``make install`` run **inside the job step**
on the GPU node (shared home for ``~/input/`` and reports).

.. list-table::
   :widths: 4 2 2 3
   :header-rows: 1

   * - Item
     - SSH launcher
     - GPU node (SSH lab)
     - GPU node (Spur / Slurm step)
   * - ``cvs run``, venv, reports
     - Yes
     - No
     - Yes
   * - ``~/input/`` (shared home)
     - Yes
     - If NFS-mounted
     - If NFS-mounted
   * - ``priv_key_file`` (unmanaged SSH)
     - Yes
     - No
     - No (HTTP agents)
   * - HF token file
     - Yes
     - If NFS-mounted
     - If NFS-mounted
   * - Models host mount (``<changeme-models-mount>`` → ``/models`` in container)
     - No
     - Yes
     - Yes
   * - Container image, ``sudo docker``
     - No
     - Yes
     - Yes
   * - ``~/LOGS/`` (volume mount)
     - No
     - Yes
     - Yes

See also
========

- :doc:`/reference/configuration-files/inference/atom` — configuration schema and thresholds
- :doc:`/reference/configuration-files/cluster-file` ΓÇö cluster file format
- :doc:`/how-to/run-with-containers` ΓÇö container backend
- :doc:`/how-to/run-cvs-tests` ΓÇö other CVS suites
