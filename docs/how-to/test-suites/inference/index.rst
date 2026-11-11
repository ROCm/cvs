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

Sample output snapshots from inference and diffusion runs.

.. important::

   The numbers in these sample reports and screenshots (throughput, latency,
   and every other metric) are **illustrative sample values, not final tuned
   results**. They are shown only to illustrate the report layout.

Inference
---------

HTML test report
~~~~~~~~~~~~~~~~

.. image:: /images/inference_pytest_summary.png
   :alt: Inference pytest HTML report summary

Run Deck report
~~~~~~~~~~~~~~~

The Run Deck dashboard: run card, lifecycle timeline, sweep cells, cross-shape
comparison, and full results.

.. image:: /images/inference_rundeck.png
   :alt: Inference Run Deck run card and lifecycle timeline

.. image:: /images/inference_sweep_cells.png
   :alt: Inference Run Deck sweep cells

.. image:: /images/inference_cross_shape_comparison.png
   :alt: Inference Run Deck cross-shape comparison

.. image:: /images/inference_full_results.png
   :alt: Inference Run Deck full results

Run Deck interactive viewer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The interactive viewer: token throughput per GPU versus interactivity, and gate
margin versus concurrency.

.. image:: /images/inference_interactivity.png
   :alt: Inference Run Deck interactive viewer

.. image:: /images/inference_gate_margin.png
   :alt: Inference Run Deck gate margin versus concurrency

Diffusion
---------

HTML test report
~~~~~~~~~~~~~~~~

.. image:: /images/diffusion_pytest_summary.png
   :alt: Diffusion pytest HTML report summary

Run Deck report
~~~~~~~~~~~~~~~

The Run Deck dashboard: run card and full results.

.. image:: /images/diffusion_rundeck.png
   :alt: Diffusion Run Deck run card

.. image:: /images/diffusion_full_results.png
   :alt: Diffusion Run Deck full results

