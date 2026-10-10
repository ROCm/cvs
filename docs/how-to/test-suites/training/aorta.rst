.. meta::
  :description: Run the Aorta training benchmark with CVS to validate iteration time, compute ratio, and rank balance on AMD Instinct GPU clusters with ROCm.
  :keywords: CVS, Aorta, training, benchmark, AMD Instinct, ROCm, AMD, GPU, RCCL, distributed, PyTorch, Docker

**********************************************
Run the Aorta training benchmark
**********************************************

Aorta runs an RCCL/training workload in containers, collects PyTorch profiler traces, and
validates iteration time, compute ratio, overlap ratio and rank balance.

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
GPU count consistent with the selected Aorta YAML and profiling workload. Prefer writable
local/scratch storage when root-squashed NFS prevents the container from writing artifacts.

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
#. Launch the workload, poll every node's exit status, and scan the bounded kernel journal.
#. Collect fresh profiler traces and execution logs.
#. Run optional TraceLens/GEMM analysis.
#. Parse results and validate thresholds.
#. Generate the JSON report and tear down.

A failed stage gates dependent stages. Trace collection, parsing of surviving artifacts,
report generation and teardown remain possible after a distributed execution failure.
Optional analysis failures fall back to raw traces. Multi-node metrics always use the
collected raw traces, since head-node Excel reports cover only that node.

Inspect outputs
---------------

CVS downloads artifacts to ``output_dir/<run-id>/`` on the machine running CVS. Distributed
traces use ``combined_traces/node_<rank>/<original-output>/torch_profiler/``. The JSON report,
``aorta_benchmark_report.json``, records cluster configuration, aggregate performance,
per-rank summaries, execution status, validation status and collection errors. Benchmark and
RCCL logs are also downloaded. No shared filesystem with the CVS machine is required.

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
