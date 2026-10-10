.. meta::
  :description: Run the Aorta training benchmark with CVS over SSH or under SPUR or Slurm to validate iteration time, compute ratio, and rank balance on AMD Instinct GPU clusters with ROCm.
  :keywords: CVS, Aorta, training, benchmark, AMD Instinct, ROCm, AMD, GPU, RCCL, distributed, PyTorch, Docker, Slurm, SPUR, HTML report

**********************************************
Run the Aorta training benchmark
**********************************************

Aorta runs an RCCL/training workload in containers, collects PyTorch profiler traces, and
validates iteration time, compute ratio, overlap ratio and rank balance.
Each run produces a pytest HTML report with one row per stage and a JSON benchmark report
with the measured metrics.

.. _aorta-set-up-config:

Set up config
=============

Copy a JSON variant and its sibling threshold file:

.. code-block:: bash

   cvs config copy benchmark/aorta/mi3xx_aorta_profile_overlap_2gpu_single.json --output ./aorta_single.json
   cvs config copy benchmark/aorta/mi3xx_aorta_profile_overlap_2gpu_distributed.json --output ./aorta_distributed.json
   cvs config copy benchmark/aorta/mi3xx_aorta_profile_overlap_2gpu_threshold.json --output ./mi3xx_aorta_profile_overlap_2gpu_threshold.json

Replace every ``<changeme>`` in the selected variant. Set the repository storage path, GPU
count, and distributed fabric interfaces for your cluster. Place Aorta at ``aorta_path`` on
each node, or enable ``aorta_auto_clone`` and provide ``aorta_clone_url``. Keep the configured
GPU count consistent with the selected Aorta YAML and profiling workload. On bare metal,
prefer writable local/scratch storage when root-squashed NFS prevents the container from
writing artifacts. Under SPUR or Slurm, ``aorta_path`` must be on shared storage; see
:ref:`aorta-spur-slurm`.

Full parameter list: :doc:`/reference/configuration-files/training/aorta`.

.. _aorta-run-tests:

Run tests
=========

Use one node with ``aorta_single`` or two or more with ``aorta_distributed``:

.. code-block:: bash

   cvs list aorta_single
   cvs list aorta_distributed
   cvs run aorta_single --cluster_file cluster-single.json --config_file aorta_single.json
   cvs run aorta_distributed --cluster_file cluster-distributed.json --config_file aorta_distributed.json

Both suites run these stages in order:

#. Launch and verify containers.
#. Clone or verify Aorta on every node.
#. Verify torchrun and configured RDMA devices (distributed only).
#. Build RCCL unless ``skip_rccl_build`` is true.
#. Launch the workload, poll every node's exit status, and scan each node's kernel log (``dmesg``) for the benchmark window.
#. Collect fresh profiler traces and execution logs.
#. Run optional TraceLens/GEMM analysis.
#. Parse results and validate thresholds.
#. Generate the JSON report and tear down.

A failed stage gates dependent stages. Trace collection, parsing of surviving artifacts,
report generation and teardown remain possible after a distributed execution failure.
Optional analysis failures fall back to raw traces. Multi-node metrics always use the
collected raw traces, since head-node Excel reports cover only that node.

See :ref:`aorta-spur-slurm` to run inside a scheduler job and :ref:`aorta-reports` for the
HTML report.

.. _aorta-spur-slurm:

Run under SPUR or Slurm
=======================

On a SPUR or Slurm cluster, launch ``cvs run`` as a job step with one task per node.
Rank 0 runs pytest and drives every stage. The other ranks serve HTTP agents that run
the suite's commands on their nodes, so the run needs no SSH between nodes. A bare
allocation shell, or a batch script that never starts a step, is not a managed run
(``SLURM_STEP_ID`` stays unset). CVS then requires ``--cluster_file`` and SSH access,
as on bare metal.

- Request one node for ``aorta_single`` and two or more for ``aorta_distributed``, with
  ``--ntasks-per-node 1`` and ``--mpi=none``. Request at least ``gpus_per_node`` GPUs
  on each node.
- Pass ``--workspace`` or set ``CVS_WORKSPACE`` to writable storage shared by every
  node in the job. A managed run fails without it.
- ``--cluster_file`` is optional. CVS builds the cluster file from the job's node list
  and uses each node's hostname as its ``vpc_ip``, so torchrun rendezvous uses the
  first node's hostname. To use a fabric address instead, either:

  - Set ``multi_node.master_addr`` in the Aorta config; or
  - Pass a ``--cluster_file`` whose ``node_dict`` lists only this job's nodes, with
    their ``vpc_ip``. Nodes outside the job are rejected.

- Keep the config file, its sibling threshold file and ``aorta_path`` on storage that
  every node sees at the same path. If compute nodes cannot reach ``aorta_clone_url``,
  clone Aorta from a login node and leave ``aorta_auto_clone`` false.
- Set ``output_dir`` to an absolute path on shared storage. A relative ``output_dir``
  resolves against the working directory of ``cvs run`` on the job's first node.
- Every node needs Docker access for the job user and passwordless ``sudo dmesg`` for
  the kernel-log scan.
- Do not add a nested ``spur run`` or ``srun`` around the workload. CVS starts one
  torchrun process per node inside the Aorta containers.
- Set the job time limit higher than ``timeout_seconds`` plus the time for container
  launch and the RCCL build.

SPUR example:

.. code-block:: bash

   spur run -A <account> -p <partition> \
     -N 2 --ntasks-per-node 1 --gpus-per-node 8 --exclusive -t 02:00:00 --mpi=none \
     bash -lc 'source <cvs-checkout>/.cvs_venv/bin/activate &&
       cvs run aorta_distributed \
         --config_file /shared/<user>/aorta/aorta_distributed.json \
         --workspace /shared/<user>/cvs'

Slurm example:

.. code-block:: bash

   srun -A <account> -p <partition> \
     -N 2 --ntasks-per-node 1 --gpus-per-node 8 --exclusive -t 02:00:00 --mpi=none \
     bash -lc 'source <cvs-checkout>/.cvs_venv/bin/activate &&
       cvs run aorta_distributed \
         --config_file /shared/<user>/aorta/aorta_distributed.json \
         --workspace /shared/<user>/cvs'

For ``aorta_single``, use ``-N 1`` and the single-node config. Scheduler detection
checks SPUR before Slurm; set ``CVS_SCHEDULER=spur`` or ``CVS_SCHEDULER=slurm`` to
override it. Set ``SPUR_CONTROLLER_ADDR`` if your SPUR deployment requires it.

.. _aorta-reports:

Reports and outputs
===================

Pytest HTML report
------------------

``cvs run`` writes a self-contained pytest HTML report and a text log to
``<workspace>/cvs_runs/<run_id>/``, named after the suite: ``aorta_single.html`` and
``aorta_single.log``, or ``aorta_distributed.html`` and ``aorta_distributed.log``.
``<run_id>`` is ``local-<timestamp>`` on bare metal and the scheduler job ID under
SPUR or Slurm. Pass ``--html`` or ``--log-file`` to choose other paths, or
``--no-html`` to skip the report. See :doc:`/reference/cli/cvs-run`.

The report has one row per stage, in run order:

#. ``test_launch_container``
#. ``test_clone_aorta``
#. ``test_setup_rdma`` (distributed only)
#. ``test_build_rccl``
#. ``test_run_benchmark``
#. ``test_collect_traces``
#. ``test_analyze``
#. ``test_parse_results``
#. ``test_validate_thresholds``
#. ``test_generate_report``
#. ``test_teardown``

Each row with captured output has a ``Full Log`` link. Skipped rows show the reason:

- ``a prior lifecycle stage failed``: an earlier stage failed. Start with the first
  failed row.
- ``skip_rccl_build=true``: the config disabled the RCCL build.
- ``Optional TraceLens/GEMM analysis is disabled``: the config disabled analysis.
- ``enforce_thresholds=false; metrics are recorded without threshold assertions``:
  the config disabled threshold assertions.
- ``No parsed benchmark result available``: parsing produced no result to report.

Threshold validation reports each configured threshold as a pytest sub-test
(``[threshold] (metric=<key>)``), and trace collection reports each node it collected
from (``[trace collection] (host=<node>)``), plus a failed
``[trace collection] (artifact=torch_traces)`` sub-test when no node produced usable
traces. Expand the ``test_validate_thresholds`` or ``test_collect_traces`` row to see
one row per sub-test. Threshold rows show the measured value (Actual) next to the
gate, and a failed row gives the reason. A failed sub-test also fails its parent row,
so dependent stages are gated and ``validation_passed`` is false. Kernel-log matches
and per-node launch failures appear on ``test_run_benchmark``.

In the JUnit XML, each sub-test is a ``cvs_aorta_subtest`` property of the parent test
case whose value gives the outcome, the measured value and gate for thresholds, and any
failure message. JUnit keeps one ``<testcase>`` per stage, but each sub-test adds one
to the ``testsuite`` ``tests`` total, and a failed sub-test adds its own ``<failure>``
to the parent ``<testcase>`` next to the parent's failure.

Next to the report, ``aorta_single_html/`` (or ``aorta_distributed_html/``) holds
the per-stage logs and copies of the cluster and config files.
``aorta_single_<timestamp>.zip`` (or ``aorta_distributed_<timestamp>.zip``) bundles
the report with those files for sharing. The bundle does not contain Aorta traces
or the JSON report; those stay under ``output_dir``.

Aorta does not produce a Run Deck report. Use the HTML report for stage verdicts and
``aorta_benchmark_report.json`` for metrics.

Benchmark artifacts
-------------------

CVS downloads artifacts to ``output_dir/<run-id>/`` on the machine running CVS, which is
the job's first node under SPUR or Slurm. ``<run-id>`` is a new hexadecimal ID for each
invocation and is separate from the CVS ``<run_id>`` above. Distributed
traces use ``combined_traces/node_<rank>/<original-output>/torch_profiler/``. The JSON report,
``aorta_benchmark_report.json``, records cluster configuration, aggregate performance,
per-rank summaries, execution status, validation status and collection errors. Benchmark and
RCCL logs are also downloaded. On bare metal over SSH, CVS copies these files from each
node, so no shared filesystem with the CVS machine is required. Under SPUR or Slurm, CVS
reads them from ``aorta_path`` on shared storage; see :ref:`aorta-spur-slurm`.

Calibrate the sample thresholds for your hardware, node count and workload. Set
``enforce_thresholds: false`` to record metrics without threshold assertions.

.. _aorta-run-deck:

Run Deck
========

``cvs run`` writes ``<suite>_html/aorta_run_deck.html`` and
``aorta_run_deck.json`` beside the pytest HTML under
``<workspace>/cvs_runs/<run_id>/``. The pytest report links the deck as
"Aorta Run Deck", and the files are included in the zip bundle. The deck is
render-only: it does not change pass/fail. A run has one result, so there is no
interactive viewer.

The run card shows workload, framework, image, cluster dimensions, metrics
source, threshold state, one verdict with actual and limit per configured
threshold, pytest stage outcome, and links to the pytest report and log. The
lifecycle timeline shows each stage's call duration. Threshold gates shows
``iteration_time``, ``compute_ratio``, ``overlap_ratio``, and ``rank_balance``.
The benchmark result card summarizes the run, and the full results table lists
all reported metrics. Per-rank breakdown graphs time, share of iteration, and
communication overlap when there are at least two ranks; graph ratios are
percentages.

Deck JSON metric keys use the ``training.`` prefix, such as
``training.avg_iteration_time_ms``. Table and JSON ratios remain fractions in
the range 0..1, matching the threshold file. Iteration time is the parser's
per-rank ``total_time_us`` averaged across ranks, and matches
``avg_iteration_time_ms`` in ``aorta_benchmark_report.json``. The reported
standard deviation, minimum, and maximum are across ranks. Time variance is
standard deviation divided by mean iteration time; it is absent when the mean
is zero. Compute and communication ratios are their respective times divided
by total time. Overlap is
``max(0, compute + comm - total) / comm``, or zero when comm is zero.

The metrics source depends on the parser path. Aorta TraceLens Excel reports
and TraceLens parsing of raw traces use the GPU timeline; "comm" then means
exposed communication time. The basic raw-trace scan, used when TraceLens is
unavailable or fails for a rank, reports total communication-kernel time as
"comm" and the sum of trace-event durations as "total". The run card names
the parser. Because per-rank fallback is not recorded, the deck uses the
neutral label "Comm time".

Each threshold verdict comes from the parser's per-threshold validation call.
Null thresholds are omitted and appear as n/a in the gate matrix. With
``enforce_thresholds: false``, the deck shows *record-only* and overall status
``record``. When an earlier stage prevents threshold validation, it shows
*not evaluated* and overall status ``record``. With no parsed metrics, overall
status is ``na``. Overall status reflects threshold evaluation; the "Pytest
stages" row shows whether another stage failed.
