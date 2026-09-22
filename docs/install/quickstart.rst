.. meta::
  :description: Get started with CVS on ROCm: install on AMD Instinct GPU clusters, configure SSH, and run your first cluster-wide command in minutes.
  :keywords: CVS, ROCm, quickstart, install, exec, AMD Instinct, GPU, AMD, SSH, cluster, Linux, getting started

*******************************************************************************
Cluster Validation Suite (CVS) quickstart: install and run your first cluster command
*******************************************************************************

This guide gets you from zero to your first cluster-wide ``cvs exec`` in about 15 minutes.

.. include:: /_includes/head-node.rst

Prerequisites
=============

The following prerequisites are required before you begin:

- Ubuntu-based Linux on the head node (see :doc:`/install/install` for supported versions).
- :doc:`Passwordless SSH </reference/cluster/passwordless-ssh>` from the head node to every worker.
- Python 3.10+ and ``python3-venv``.

Step 1: Install CVS
===================

Clone the repository and install:

.. code:: bash

  git clone https://github.com/ROCm/cvs
  cd cvs
  make install
  source .cvs_venv/bin/activate
  cvs --version
  cvs list

Step 2: Copy the cluster file
=============================

Every CVS command needs a ``cluster.json`` that lists your nodes and SSH credentials.

.. code:: bash

  mkdir -p ~/cvs_workspace
  cvs config copy cluster.json --output ~/cvs_workspace/cluster.json

Edit ``cluster.json`` and replace every ``<changeme>`` placeholder with your node hostnames, SSH user, and key path. CVS exits with an error if any placeholder remains unresolved. See :doc:`/how-to/configure/cluster-config` and :doc:`/reference/cluster/cluster-file`.

Step 3: Run cluster-wide commands
=================================

Use ``cvs exec`` to run a shell command on every node in parallel:

.. code:: bash

  cvs exec --cmd "hostname" --cluster_file ~/cvs_workspace/cluster.json

Successful output looks like this — one block per node:

.. code:: text

  [compute] Host: 10.0.0.2
  node01
  ---
  [compute] Host: 10.0.0.3
  node02
  ---

.. image:: /images/cvs-exec-hostname.png
   :alt: Terminal output of cvs exec hostname, with one hostname per compute node

You should see one hostname per node in your cluster. You can also set ``CLUSTER_FILE`` once and omit ``--cluster_file`` on later commands. See :doc:`/how-to/execute-cluster-commands` for ``--target``, ``--json``, and timeouts.

Validate success
================

If ``cvs exec`` returned a hostname from every node in your cluster file, CVS is installed and connected. Your cluster is ready to run tests.

If any node is missing from the output, check:

- Passwordless SSH works from the head node to that node: ``ssh <user>@<host> hostname``.
- The node's hostname or IP in ``cluster.json`` is correct and reachable.
- The SSH key path and user in ``cluster.json`` match what works in the manual SSH check above.

Run the GPU visibility check to confirm AMD GPUs are visible on all nodes before running any test suite:

.. code:: bash

  cvs exec --cmd "amd-smi list" \
    --cluster_file ~/cvs_workspace/cluster.json

Successful output from each node looks like this:

.. code:: text

  [compute] Host: 10.0.0.2
  GPU: 0
      BDF: 0000:03:00.0
      UUID: c30074a5-0000-1000-81ab-4f2e7c6d90b1
      KFD_ID: 42109
      NODE_ID: 2
      PARTITION_ID: 0
  <<truncated>>
  ---
  [compute] Host: 10.0.0.3
  GPU: 0
      BDF: 0000:23:00.0
      UUID: a10074a5-0000-1000-8042-6e1c8a9b52d0
      KFD_ID: 31758
      NODE_ID: 4
      PARTITION_ID: 0
  <<truncated>>
  ---

Every node should report its GPU devices. A node that returns no GPUs or an error indicates a driver or device access issue to resolve before testing.

What to do next
===============

CVS is installed and your cluster is connected. The next step is to run a test suite.

- :doc:`/how-to/test-suites/index` — choose a test suite and run it against the cluster. Start with :doc:`Preflight </how-to/test-suites/burn-in-diag/preflight>` to smoke-test all nodes, or jump straight to the suite that matches your validation goal.
- :doc:`/how-to/configure/test-suite-config/index` — every test suite requires a config file with cluster-specific values such as interface names, model paths, and thresholds. Copy a template, fill in your values, and pass it with ``--config_file``.
- :doc:`/how-to/configure/cluster-config` — if your cluster topology changes (new nodes, rack layout, container backend), update the cluster file here.
- :doc:`/install/uninstall` — uninstall or downgrade CVS on the head node.
