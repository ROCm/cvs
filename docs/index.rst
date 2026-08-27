.. meta::
  :description: CVS documentation
  :keywords: CVS, ROCm, documentation, test suites, validation

********************************************
ROCm Cluster Validation Suite (CVS) documentation
********************************************

CVS is a collection of test suites that validate AMD ROCm clusters end to end, from single-node burn-in health tests to cluster-wide distributed training and inferencing.
CVS requires only SSH connectivity to the cluster nodes — no Slurm, no Kubernetes, no scheduler needed.

The component public repository is located at `https://github.com/ROCm/cvs <https://github.com/ROCm/cvs>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Concepts

    * :doc:`What is CVS <concepts/what-is-cvs>`
    * :doc:`Scalability and performance <concepts/cvs-at-scale>`

  .. grid-item-card:: Getting started

    * :doc:`Quickstart </getting-started/quickstart>`
    * :doc:`Install </getting-started/install>`
    * :doc:`Upgrade </getting-started/upgrade>`

  .. grid-item-card:: How to

    * :doc:`Set up cluster file <how-to/configure/cluster-config>`
    * :doc:`Set up test configs <how-to/configure/test-suite-config/index>`
    * :doc:`Run tests <how-to/run-tests/index>`
    * :doc:`Run cluster commands <how-to/execute-cluster-commands>`
    * :doc:`Copy to cluster <how-to/copy-to-cluster>`
    * :doc:`Monitor cluster health <how-to/monitor/index>`

  .. grid-item-card:: Reference

    * :doc:`Cluster file <reference/cluster/cluster-file>`
    * :doc:`Run with containers <how-to/run-with-containers>`
    * :doc:`Passwordless SSH <reference/cluster/passwordless-ssh>`
    * :doc:`Configuration files <reference/configuration-files/index>`
    * :doc:`CLI reference <reference/cli/cvs-run>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the :doc:`Licensing page <license>`.
