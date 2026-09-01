.. meta::
  :description: Run inference and benchmarking CVS test suites
  :keywords: CVS, vLLM, ATOM, SGLang, xDiT, inference

*****************
Inference tests
*****************

LLM serving, disaggregated prefill/decode, and diffusion workloads for cluster-scale inference validation.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Suite
     - How to run
     - Config reference
   * - vLLM
     - :doc:`/how-to/test-suites/inference/vllm`
     - :doc:`/reference/configuration-files/inference/vllm`
   * - ATOM
     - :doc:`/how-to/test-suites/inference/atom`
     - :doc:`/reference/configuration-files/inference/atom`
   * - SGLang
     - :doc:`/how-to/test-suites/inference/sglang`
     - :doc:`/reference/configuration-files/inference/sglang`
   * - xDiT
     - :doc:`/how-to/test-suites/inference/xdit`
     - :doc:`/reference/configuration-files/inference/xdit`

See also :doc:`/how-to/run-tests/index` for common ``cvs run`` flags and workflow.
