.. meta::
  :description: CVS documentation
  :keywords: CVS, ROCm, documentation, test suites, validation

********************************************
Cluster Validation Suite (CVS) documentation
********************************************

CVS is a collection of test suites that validate AMD AI clusters end to end, from single-node burn-in health tests to cluster-wide distributed training and inferencing.
CVS requires only SSH connectivity to the cluster nodes — no Slurm, no Kubernetes, no scheduler needed.

The component public repository is located at `https://github.com/ROCm/cvs <https://github.com/ROCm/cvs>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Overview

    * :doc:`What is CVS <what-is-cvs>`

  .. grid-item-card:: Install

    * :doc:`Cluster Validation Suite installation </install/cvs-install>`

  .. grid-item-card:: How to

    * :doc:`Run tests <how-to/run-cvs-tests>`
    * :doc:`Run ad-hoc cluster-wide commands <how-to/execute-cluster-commands>`
    * :doc:`Copy files and directories to cluster nodes <how-to/copy-to-cluster>`
    * :doc:`Monitor the health of GPU clusters <how-to/run-cluster>`

  .. grid-item-card:: Reference

    * :doc:`Test configuration files <reference/configuration-files/configure-config>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the :doc:`Licensing page <license>`.
