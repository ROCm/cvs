.. meta::
  :description: Run distributed training CVS test suites
  :keywords: CVS, JAX, Megatron, TorchTitan, Aorta, training

****************
Training tests
****************

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
     - :doc:`/how-to/test-suites/training/jax`
     - :doc:`/reference/configuration-files/training/jaxmaxtext`
   * - Megatron
     - :doc:`/how-to/test-suites/training/megatron`
     - :doc:`/reference/configuration-files/training/megatron`
   * - TorchTitan
     - :doc:`/how-to/test-suites/training/torchtitan`
     - :doc:`/reference/configuration-files/training/torchtitan`

See also :doc:`/how-to/run-tests/index` for common ``cvs run`` flags and workflow.
