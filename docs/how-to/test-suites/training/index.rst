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

.. _training-common-prerequisites:

Common prerequisites
====================

These prerequisites apply to every training suite. Each suite page lists only
its additional requirements.

On every GPU node:

- Docker, with the SSH user able to run it through passwordless ``sudo docker``
  or membership in the ``docker`` group.
- The host ROCm driver loaded, so ``/dev/kfd``, ``/dev/dri/*``, and
  ``/dev/infiniband/*`` when the workload uses RDMA are available for device
  passthrough.
- The suite container image loaded on the node or pullable from a reachable
  registry.

On the machine where you run ``cvs run``:

- CVS installed (see :doc:`/install/install`).
- SSH key access to every cluster node.

For a distributed run, RDMA interfaces must be configured and reachable on
every node.

Test results
============

Sample output snapshots from a training run.

.. important::

   The numbers in these sample reports and screenshots (throughput, loss, step
   time, and every other metric) are illustrative sample values, not final
   tuned results. They are shown only to illustrate the report layout.

HTML test report
----------------

**Per-metric subtests**

Each ``test_metric`` row expands to show every metric as its own pass, fail, or
skip verdict.

**Per-metric subtests - collapsed:**

.. image:: /images/training_html_report_subtests_collapsed.png
   :alt: Training per-metric subtests, collapsed

**Per-metric subtests - expanded:**

.. image:: /images/training_html_report_subtests_expanded.png
   :alt: Training per-metric subtests, expanded

**Metric results table:**

.. image:: /images/training_metric_results_table.png
   :alt: Training metric results table

Run Deck report
---------------

The Run Deck dashboard: run card, lifecycle timeline, sweep analytics charts,
gate matrix, per-sweep cells and full results.

.. image:: /images/training_rundeck_results_sample__runcard.png
   :alt: Training Run Deck run card

.. image:: /images/training_rundeck_results_sample__sweep-charts.png
   :alt: Training Run Deck results metrics by sweep charts

.. image:: /images/training_rundeck_results_sample__sweep-cells.png
   :alt: Training Run Deck sweep cells

Run Deck interactive viewer
---------------------------

The interactive viewer: per-step loss, learning-rate, grad-norm, throughput, and
token graphs, with background / layout / metric-selector controls.

.. image:: /images/training_viewer_graphs_sample__page1.png
   :alt: Training Run Deck interactive viewer graphs (page 1)

.. image:: /images/training_viewer_graphs_sample__page2.png
   :alt: Training Run Deck interactive viewer graphs (page 2)
