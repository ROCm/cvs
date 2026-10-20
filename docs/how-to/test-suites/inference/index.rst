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

.. _inference-common-prerequisites:

Common prerequisites
====================

These prerequisites apply to every inference suite. Each suite page lists only
its additional requirements.

On every GPU node:

- Docker, with the SSH user able to run it through passwordless ``sudo docker``
  or membership in the ``docker`` group.
- The host ROCm driver loaded, so ``/dev/kfd``, ``/dev/dri/*``, and
  ``/dev/infiniband/*`` when the workload uses RDMA are available for device
  passthrough.
- The suite container image loaded on the node or pullable from a reachable
  registry.
- Model weights on storage mounted at the same path on every participating node.
- A Hugging Face token file when the model requires gated access.

On the machine where you run ``cvs run``:

- CVS installed (see :doc:`/install/install`).
- SSH key access to every cluster node.

Test results
============

Sample pytest HTML report from an SGLang inference run.

.. important::

   The numbers in this sample report are **illustrative sample values, not final
   tuned results**. They are shown only to illustrate the report layout.

**HTML test report**

.. image:: /images/inference_pytest_report_sample.png
   :alt: Sample SGLang inference HTML test report
