.. meta::
  :description: Run Aorta single-node and distributed benchmarks through CVS
  :keywords: Aorta, ROCm, RCCL, benchmark, CVS

Aorta benchmark
===============

Aorta runs an RCCL/training workload in containers, collects PyTorch profiler traces, and
validates iteration time, compute ratio, overlap ratio and rank balance.

Prepare a configuration
-----------------------

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

See :doc:`the configuration reference </reference/configuration-files/training/aorta>` for
field descriptions and migration details.

Run the suite
-------------

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
