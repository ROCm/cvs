.. meta::
  :description: Run CVS inference and benchmarking test suites for LLM serving and diffusion workloads on AMD Instinct GPU clusters with ROCm.
  :keywords: CVS, inference, benchmark, AMD Instinct, ROCm, AMD, GPU, vLLM, ATOM, SGLang, xDiT, LLM

***************************************************************************
Run Cluster Validation Suite (CVS) inference and benchmarking test suites
***************************************************************************

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

See also :doc:`/how-to/test-suites/index` for common ``cvs run`` flags and workflow.
