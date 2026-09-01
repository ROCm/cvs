.. meta::
  :description: Quickstart — install CVS and run your first cluster-wide command
  :keywords: CVS, quickstart, install, exec

**********
Quickstart
**********

This guide gets you from a fresh checkout to your first cluster-wide ``cvs exec`` in about 15 minutes.

Prerequisites
=============

- Ubuntu-based Linux on the head node (see :doc:`/getting-started/install` for supported versions)
- :doc:`Passwordless SSH </reference/cluster/passwordless-ssh>` from the head node to every worker
- Python 3.10+ and ``python3-venv``

Step 1: Install CVS
===================

From the repository root:

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

Edit ``cluster.json`` with your node hostnames, SSH user, and key path. See :doc:`/how-to/configure/cluster-config` and :doc:`/reference/cluster/cluster-file`.

Step 3: Run cluster-wide commands
=================================

Use ``cvs exec`` to run a shell command on every node in parallel:

.. code:: bash

  cvs exec --cmd "hostname" --cluster_file ~/cvs_workspace/cluster.json

Check GPU visibility on all nodes:

.. code:: bash

  cvs exec --cmd "amd-smi list" \
    --cluster_file ~/cvs_workspace/cluster.json

For GPU product name and memory (JSON):

.. code:: bash

  cvs exec --cmd "amd-smi static --json && amd-smi metric --json" \
    --cluster_file ~/cvs_workspace/cluster.json

You can also set ``CLUSTER_FILE`` once and omit ``--cluster_file`` on later commands. See :doc:`/how-to/execute-cluster-commands` for ``--target``, ``--json``, and timeouts.

Next steps
==========

- :doc:`/how-to/configure/cluster-config` — configure the cluster file
- :doc:`/how-to/configure/test-suite-config/index` — copy and edit test suite configs
- :doc:`/how-to/run-tests/index` — run tests (pick a test suite and run commands)
