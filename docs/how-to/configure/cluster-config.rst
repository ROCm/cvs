.. meta::
  :description: Configure the CVS cluster file (cluster.json)
  :keywords: CVS, configure, cluster, cluster.json, cvs config

*********************
Set up cluster file
*********************

The cluster file (``--cluster_file``) tells CVS how to reach your cluster nodes: SSH credentials, which hosts to use, and whether to run on bare metal or in containers. You pass it to ``cvs run``, ``cvs exec``, and ``cvs scp``.

Field-level schema: :doc:`/reference/cluster/cluster-file`.

Two ways to create a cluster file
=================================

You can create ``cluster.json`` in either of these ways:

#. **Copy a template and edit manually** — use ``cvs config copy`` to copy ``cluster.json`` (or ``cluster_container.json`` for container runs), then edit SSH credentials, the head node, and worker addresses. For smaller clusters this is convenient. See `Copy a cluster file template`_.

#. **Generate from a host list** — use ``cvs generate cluster_json`` with a hosts file or comma-separated host list to build ``cluster.json`` from your node inventory. Use this when you have many nodes or when you already maintain a host list. See `Generate a cluster JSON file`_.

Copy a cluster file template
============================

CVS ships cluster templates under ``cvs/input/``. Copy them with ``cvs config copy``:

.. code:: bash

  cvs config copy cluster.json --output ~/cvs_workspace/cluster.json
  cvs config copy cluster_container.json --output ~/cvs_workspace/cluster_container.json

Edit ``cluster.json`` for your environment: SSH user, private key path, head node, and worker node addresses.

Generate a cluster JSON file
============================

For many nodes, generate ``cluster.json`` from a hosts file or comma-separated host list:

**Option 1: hosts file**

.. code:: bash

  cvs generate cluster_json \
    --input_hosts_file /tmp/hosts.txt \
    --output_json_file ~/cvs_workspace/cluster.json \
    --username myuser --key_file ~/.ssh/id_rsa --head_node 192.168.1.10

The hosts file supports one IP or hostname per line, ranges like ``192.168.1.11-15``, and bracket notation like ``server[01-05]``.

**Option 2: comma-separated hosts**

.. code:: bash

  cvs generate cluster_json \
    --hosts "192.168.1.10,192.168.1.11-15,server[01-05]" \
    --output_json_file ~/cvs_workspace/cluster.json \
    --username myuser --key_file ~/.ssh/id_rsa --head_node 192.168.1.10

Passwordless SSH
================

Passwordless SSH from the head node to every worker is required. See :doc:`/reference/cluster/passwordless-ssh`.

Container backend
=================

For container-based suites, use ``cluster_container.json`` and see :doc:`/how-to/run-with-containers`.

Next step
=========

Copy and edit a test suite config, then run tests: :doc:`/how-to/configure/test-suite-config/index`.
