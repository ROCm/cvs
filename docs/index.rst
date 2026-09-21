.. meta::
  :description: CVS (Cluster Validation Suite) is a collection of test suites that validate AMD ROCm clusters, from single-node burn-in to distributed training and inference.
  :keywords: CVS, ROCm, AMD, cluster validation, GPU, AMD Instinct, test suites, burn-in, inference, training, RCCL, InfiniBand

********************************************
ROCm Cluster Validation Suite (CVS) documentation
********************************************

CVS is a collection of test suites that validate AMD ROCm clusters end to end, from single-node burn-in health tests to cluster-wide distributed training and inference.
CVS requires only SSH connectivity to the cluster nodes — no Slurm, Kubernetes, or scheduler needed.

The component public repository is located at `https://github.com/ROCm/cvs <https://github.com/ROCm/cvs>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Quickstart </install/quickstart>`
    * :doc:`Install CVS </install/install>`
    * :doc:`Upgrade CVS </install/upgrade>`

  .. grid-item-card:: How to

    * :doc:`Set up a cluster file <how-to/configure/cluster-config>`
    * :doc:`Set up test configs <how-to/configure/test-suite-config/index>`
    * :doc:`Run tests <how-to/test-suites/index>`
    * :doc:`Run cluster commands <how-to/execute-cluster-commands>`
    * :doc:`Copy files to cluster nodes <how-to/copy-to-cluster>`
    * :doc:`Monitor cluster health <how-to/monitor/index>`

  .. grid-item-card:: Reference

    * :doc:`Scalability and parallel SSH performance <reference/cvs-at-scale>`
    * :doc:`Cluster file <reference/cluster/cluster-file>`
    * :doc:`Run with containers <how-to/run-with-containers>`
    * :doc:`Passwordless SSH <reference/cluster/passwordless-ssh>`
    * :doc:`Configuration files <reference/configuration-files/index>`
    * :doc:`CLI reference <reference/cli/cvs-run>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the :doc:`License <license>` page.
