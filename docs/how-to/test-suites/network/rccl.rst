.. meta::
  :description: Run CVS RCCL performance and regression test suites to validate collective communication bandwidth across AMD Instinct GPU cluster nodes with ROCm.
  :keywords: CVS, RCCL, collective communication, AMD Instinct, ROCm, AMD, GPU, InfiniBand, all-reduce, all-gather, performance

***********************************************
Run CVS RCCL performance and regression tests
***********************************************

RCCL tests validate collective communication bandwidth and correctness across AMD Instinct GPU nodes using all-reduce, all-gather, broadcast, and other collectives. Run RCCL after IB Perf passes to confirm that GPU-to-GPU communication meets performance thresholds.

- ``rccl_perf`` runs configured collectives across ``rccl_test_params.data_types``.
- ``rccl_regression`` runs the Cartesian product of the ``regression`` environment axes and collectives.
- ``rccl_pairwise`` runs Phase 0 reference sanity, Phase 1 reference/candidate pairs, and a Phase 2 incremental group.

.. _rccl-set-up-config:

Set up config
=============

1. Copy the RCCL configuration file:

   .. code:: bash

     cvs config copy rccl/rccl_config.json --output ~/cvs_workspace/rccl/rccl_config.json

2. Edit directory paths and any ``<changeme>`` placeholders:

   - ``rccl_dir``, ``rccl_tests_dir``, ``mpi_dir``
   - ``mpi_path_var``, ``rccl_path_var``, ``rocm_path_var``

3. All three suites read the same file. ``rccl_regression`` additionally requires a non-empty ``regression`` object; ``rccl_pairwise`` uses ``cvs_params.pairwise_min_bw`` and ``cvs_params.pairwise_results_file``.

For the complete field reference, see :doc:`/reference/configuration-files/network/rccl`.

.. _rccl-run-tests:

Run tests
=========

You can list all available RCCL test cases using the CLI:

.. code:: bash

  cvs list rccl_perf

.. code:: text

  Available tests in rccl_perf:
    - test_collect_hostinfo
    - test_collect_networkinfo
    - test_disable_firewall
    - test_gen_graph
    - test_print_env_once
    - test_rccl_perf[all_gather_perf]
    - test_rccl_perf[all_reduce_perf]
    - test_rccl_perf[alltoall_perf]
    - test_rccl_perf[alltoallv_perf]
    - test_rccl_perf[broadcast_perf]
    - test_rccl_perf[gather_perf]
    - test_rccl_perf[reduce_scatter_perf]
    - test_rccl_perf[scatter_perf]
    - test_rccl_perf[sendrecv_perf]

.. code:: bash

  cvs list rccl_regression

.. code:: text

  Available tests in rccl_regression:
    - test_collect_hostinfo
    - test_collect_networkinfo
    - test_disable_firewall
    - test_gen_graph
    - test_print_env_once
    - test_rccl_perf

.. code:: bash

  cvs list rccl_pairwise

.. code:: text

  Available tests in rccl_pairwise:
  --------------------------------------------------------------------------------
    • test_collect_hostinfo
    • test_collect_networkinfo
    • test_gen_graph
    • test_rccl_incremental
    • test_rccl_pairwise

  Total: 5 tests

Prerequisites for environment script staging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before running RCCL tests, ensure the environment script specified in ``env_source_script`` is available on all cluster nodes:

With shared storage
"""""""""""""""""""

Place the environment script in a shared directory accessible from all nodes.

Without shared storage
""""""""""""""""""""""

Use ``cvs scp`` to copy the environment script to all nodes. See :doc:`/how-to/copy-to-cluster` for detailed instructions and examples.

.. _rccl-run-baremetal:

Run on bare metal or with the container backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Run RCCL performance suite:

   .. code:: bash

     cvs run rccl_perf \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/rccl/rccl_config.json \
       --capture=tee-sys -vvv -s

2. Run RCCL regression suite:

   .. code:: bash

     cvs run rccl_regression \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/rccl/rccl_config.json \
       --capture=tee-sys -vvv -s

3. Run RCCL pairwise suite:

   .. code:: bash

     cvs run rccl_pairwise \
       --cluster_file ~/cvs_workspace/cluster.json \
       --config_file ~/cvs_workspace/rccl/rccl_config.json \
       --capture=tee-sys -vvv -s

4. Generate RCCL performance heatmap:

   .. code:: bash

     cvs generate heatmap --actual /tmp/rccl_perf_results.json --reference /path/to/golden_reference.json --output /var/www/html/cvs/rccl_heatmap.html --title "RCCL Performance Comparison"

Each ``cvs run`` command writes its pytest HTML report and log under the run directory, and generates a Run Deck when ``test_gen_graph`` collects results. See :ref:`rccl-read-results`. Pass ``--workspace <shared path>`` to choose where the run directory is created.

.. _rccl-run-managed:

Run under SPUR or Slurm (managed mode)
======================================

Start one CVS task per node inside an allocation. Rank 0 runs pytest and the other ranks serve commands through HTTP agents; an allocation shell alone does not start the agents.

For SPUR:

.. code:: bash

  spur run --mpi=none -N 2 --ntasks-per-node 1 -- \
    cvs run rccl_perf \
      --config_file /shared/user/rccl_config.json \
      --workspace /shared/user/cvs \
      --capture=tee-sys -vvv -s

For Slurm:

.. code:: bash

  srun --mpi=none -N 2 --ntasks-per-node 1 -- \
    cvs run rccl_perf \
      --config_file /shared/user/rccl_config.json \
      --workspace /shared/user/cvs \
      --capture=tee-sys -vvv -s

- ``--workspace`` or ``CVS_WORKSPACE`` is required for every managed run. It must name writable shared storage visible at the same path on all nodes; otherwise CVS exits before starting. The run directory is ``<workspace>/cvs_runs/<job_id>/``, where ``<job_id>`` is ``$SLURM_JOB_ID`` (also set by SPUR).
- ``--cluster_file`` is optional. CVS synthesizes a job-owned cluster file using each node's hostname as its ``vpc_ip``. Pass a cluster file when selecting the container orchestrator.
- Scheduler detection checks SPUR before Slurm. Override it with ``CVS_SCHEDULER=spur`` or ``CVS_SCHEDULER=slurm``. The override does not remove the job-step requirement. Configure ``SPUR_CONTROLLER_ADDR`` as your Spur deployment requires.
- CVS launches each RCCL workload as a nested ``spur run --overlap --mpi=pmix`` or ``srun --overlap --mpi=pmix`` step. It supplies MPI/PMIx settings from ``mpi_params``. The ``env_source_script`` must still exist at the same path on every node.
- Before the first launch, CVS writes a sentinel through the head node into the result directory (``cvs_params.rccl_result_file``, default ``{run_dir}/rccl_result_file.json``) and reads it back. If the directory is unwritable or unreachable, the run fails early and the error names ``--workspace``/``CVS_WORKSPACE``.
- Under SPUR, ``test_rccl_pairwise`` and ``test_rccl_incremental`` are skipped with the reason "SPUR 0.11 ignores --nodelist on job steps; pairwise/incremental RCCL is not supported until Spur applies -w to nested steps." Host and network collection still runs; ``test_gen_graph`` finds no results, so no Run Deck is written. Under Slurm, subset steps use ``-w`` and pairwise runs normally.
- Every suite run in the same allocation shares one run directory because the run ID is the job ID. Different suites use files named by test-file stem. Re-running the same suite in that allocation overwrites ``<stem>.html`` and ``<stem>.log`` and clears ``<stem>_html/``. Pass distinct ``--html`` and ``--log-file`` paths per repeat to keep both; a new timestamped zip is written each time.
- Only rank 0 writes the report. Collect artifacts from the shared workspace, rather than node-local paths.

.. _rccl-read-results:

Read the results
================

Pytest HTML report and bundle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Artifact
     - Default location
     - Notes
   * - Pytest HTML report
     - ``<run_dir>/<test-file-stem>.html``
     - For example, ``<workspace>/cvs_runs/<run_id>/rccl_perf.html``. The auto-derived report is self-contained. ``<run_id>`` is ``local-<YYYYmmdd-HHMMSS>`` outside a scheduler or the job ID under SPUR/Slurm.
   * - Text log
     - ``<run_dir>/<test-file-stem>.log``
     - Pytest run log.
   * - Attachments
     - ``<run_dir>/<test-file-stem>_html/``
     - Per-test "Full Log" pages, the amCharts report for perf/regression, the Run Deck, and a copy of the run log. Recreated at the start of each run.
   * - Zip bundle
     - ``<run_dir>/<test-file-stem>_<YYYY-mm-ddTHHMMSS>.zip``
     - Contains the report, attachments directory, and cluster/config files. Links still work after extraction.

``<run_dir>`` is ``<workspace>/cvs_runs/<run_id>``. ``--html <path>`` moves the report, its ``_html/`` directory, the zip, and the Run Deck next to ``<path>``. Add ``--self-contained-html`` to make an explicit HTML path portable. ``--no-html`` disables the report, zip, and Run Deck. With direct ``pytest``, the report and Run Deck require ``--html``.

The report's **Reports** section, between Environment and Summary, links "RCCL Performance Report" for ``rccl_perf`` or "RCCL Multi Node Performance Report" for ``rccl_regression``, plus "RCCL Run Deck" when those artifacts are generated. The amCharts link also appears on the ``test_gen_graph`` row. ``rccl_pairwise`` has no amCharts report; its Run Deck HTML is linked from Reports, while the JSON sits beside it in the bundle.

See :doc:`/reference/cli/cvs-run` for report and log options.

Run Deck
~~~~~~~~

The RCCL Run Deck is a render-only HTML/JSON dashboard shared by all three suites: ``rccl_run_deck.html`` and ``rccl_run_deck.json``. It does not affect pass/fail. Deck-generation errors are logged and the suite result is kept.

Run any RCCL suite with HTML reporting on (the ``cvs run`` default). The deck is built at session end from results collected by ``test_gen_graph``. Include that test when selecting individual tests, for example:

.. code:: bash

  cvs run rccl_perf "test_rccl_perf[all_reduce_perf]" test_gen_graph \
    --cluster_file ~/cvs_workspace/cluster.json \
    --config_file ~/cvs_workspace/rccl/rccl_config.json

Without collected results, the log shows ``Skipping Run Deck generation: no results in session store``. On success it shows ``Run Deck written (rccl): …``. Both files are written to ``<run_dir>/<test-file-stem>_html/``. RCCL has no interactive viewer.

.. list-table::
   :header-rows: 1

   * - Card
     - Shows
   * - Run card
     - Framework, Nodes, MPI nodes, Local ranks, MPI ranks, Collectives, Msg size (bytes), optional NIC model, Thresholds, Pytest report, and Run log. Thresholds reflects only the ``verify_bus_bw``, ``verify_bw_dip``, and ``verify_lat_dip`` switches; it does not confirm a threshold was found or applied. The report and log links point back to the bundle.
   * - Bus bandwidth vs message size (GB/s)
     - Bus bandwidth by message size for each series.
   * - Algorithm bandwidth vs message size (GB/s)
     - Algorithm bandwidth by message size for each series.
   * - Time vs message size (us)
     - Time by message size for each series.
   * - Full results
     - Collective, Message size (raw bytes), Bus BW (GB/s), Alg BW (GB/s), and Time (us).

Series labels vary by suite:

- ``rccl_perf`` uses the collective name, for example ``all_reduce_perf``.
- ``rccl_regression`` uses ``<collective>-<ENV=val ENV=val …>``.
- ``rccl_pairwise`` uses ``Phase0 sanity <node>``, ``Phase1 <ref> <-> <cand>``, and ``Phase2 <N>-node cluster (adding <cand>)``. Its run card's MPI nodes and MPI ranks list each distinct count used across phases; Nodes counts distinct nodes used.

Keep these limits in mind when reading the deck:

- Each series has its own chart, sorted by label. At most 40 charts are drawn per card. Truncated cards say "Showing N of M series charts"; the table and JSON keep all results.
- Chart x-labels are humanized (``1K``, ``1M``, ``1G``); the table and JSON use raw bytes.
- When rows share a series and message size, the last row wins across in-place/out-of-place, data type, and cycle.
- The deck shows only collectives and sizes actually collected.

The JSON includes ``results_table.headers`` and ``results_table.rows``, ``datasets.series.charts.bus_bw|alg_bw|time``, ``run_card_display``, ``provenance``, ``generated_at``, and ``cvs_version``. Its ``overall_status`` is always ``record`` because the deck is not a verdict. For example, extract one collective's rows with:

.. code:: bash

  jq -r '.results_table.rows[] | select(.[0]=="all_reduce_perf") | @tsv' rccl_run_deck.json

The deck is built from in-memory results. Per-run JSON files under ``{run_dir}`` (``rccl_result_file*.json``, ``*_aggregated.json``, ``*_topology_check.json``) are separate. Later pairwise and regression sub-runs overwrite those files.

Related resources
=================

- :doc:`/reference/configuration-files/network/rccl`
- :doc:`/reference/cli/cvs-run`
- :doc:`/how-to/copy-to-cluster`
