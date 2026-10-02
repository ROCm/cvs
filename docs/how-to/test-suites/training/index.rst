.. meta::
  :description: Run CVS distributed training test suites to benchmark multi-node GPU training throughput and scaling on AMD Instinct clusters before production jobs.
  :keywords: CVS, training, distributed training, AMD Instinct, ROCm, AMD, GPU, JAX, Megatron, TorchTitan, Aorta, MaxText

*****************************************************************
Run Cluster Validation Suite (CVS) distributed training test suites
*****************************************************************

Distributed training benchmarks validate multi-node GPU training before production jobs.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Suite
     - How to run
     - Config reference
   * - Aorta
     - :doc:`/how-to/test-suites/training/aorta`
     - :doc:`/reference/configuration-files/training/aorta`
   * - JAX MaxText
     - :doc:`/how-to/test-suites/training/jaxmaxtext`
     - :doc:`/reference/configuration-files/training/jaxmaxtext`
   * - Megatron
     - :doc:`/how-to/test-suites/training/megatron`
     - :doc:`/reference/configuration-files/training/megatron`
   * - TorchTitan
     - :doc:`/how-to/test-suites/training/torchtitan`
     - :doc:`/reference/configuration-files/training/torchtitan`

See also :doc:`/how-to/test-suites/index` for common ``cvs run`` flags and workflow.
