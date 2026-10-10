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

Threshold validation reports each configured threshold as a pytest sub-test
(``[threshold] (metric=<key>)``), and trace collection reports each node it collected
from (``[trace collection] (host=<node>)``). In the HTML report, expand the
``test_validate_thresholds`` or ``test_collect_traces`` row to see one row per sub-test.
A failed sub-test also fails its parent test, so dependent stages are gated and
``validation_passed`` is false.

Calibrate the sample thresholds for your hardware, node count and workload. Set
``enforce_thresholds: false`` to record metrics without threshold assertions.
