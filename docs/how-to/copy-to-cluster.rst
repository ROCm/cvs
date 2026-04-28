.. meta::
  :description: Copy files and directories to cluster nodes using CVS
  :keywords: CVS, cluster, copy, scp, files, directories, parallel

*******************************
Copy Files and Directories to Cluster Nodes
*******************************

CVS provides an ``scp`` command to copy files and directories to all nodes in the cluster simultaneously using parallel SCP operations. This is useful for distributing configuration files, scripts, data files, or software packages across all cluster nodes at once.

The ``scp`` command uses the same cluster configuration files as other CVS commands and supports both command-line arguments and environment variables for configuration.

Copy files
==========

To copy a file to all cluster nodes:

.. code:: bash

  cvs scp --file /path/to/local/file --cluster_file input/cluster_file/cluster.json

You can also use the ``CLUSTER_FILE`` environment variable:

.. code:: bash

  CLUSTER_FILE=input/cluster_file/cluster.json cvs scp --file /path/to/local/file

Copy directories
================

To copy a directory recursively to all cluster nodes:

.. code:: bash

  cvs scp --file /path/to/local/directory --recurse --cluster_file input/cluster_file/cluster.json

Command options
===============

The ``scp`` command supports these options:

- ``--file``: Local file or directory to copy to all nodes (required)
- ``--dest``: Remote destination path (optional, defaults to same path as source)
- ``--recurse``: Copy directories recursively (required for directories)
- ``--cluster_file``: Path to cluster configuration JSON file (optional if ``CLUSTER_FILE`` environment variable is set)
- ``--parallel``: Number of parallel SCP operations (default: 20)

Performance tuning
==================

The ``--parallel`` option controls how many SCP operations run simultaneously:

.. code:: bash

  # Use more parallel operations for faster copying (if network allows)
  cvs scp --file ./large_file --parallel 50 --cluster_file input/cluster_file/cluster.json

  # Use fewer parallel operations to reduce network load
  cvs scp --file ./sensitive_data --parallel 5 --cluster_file input/cluster_file/cluster.json

.. note::

  The default parallel setting is 20, which works well for most clusters. Increase this value if you have high network bandwidth and want faster transfers. Decrease it if you experience network congestion or timeouts.

File paths and permissions
==========================

Destination paths
-----------------

When ``--dest`` is not specified, files are copied to the same path on remote nodes:

.. code:: bash

  # This copies /home/user/file.txt to /home/user/file.txt on all nodes
  cvs scp --file /home/user/file.txt --cluster_file input/cluster_file/cluster.json

When ``--dest`` is specified, files are copied to that exact path:

.. code:: bash

  # This copies local file.txt to /tmp/remote_file.txt on all nodes
  cvs scp --file ./file.txt --dest /tmp/remote_file.txt --cluster_file input/cluster_file/cluster.json

Permissions and ownership
-------------------------

Files are copied with the permissions of the remote user specified in the cluster configuration. To set specific permissions after copying, combine with the ``exec`` command:

.. code:: bash

  # Copy a script and make it executable
  cvs scp --file ./setup.sh --dest /tmp/setup.sh --cluster_file input/cluster_file/cluster.json
  cvs exec --cmd "chmod +x /tmp/setup.sh" --cluster_file input/cluster_file/cluster.json

  # Copy configuration and set ownership
  cvs scp --file ./app.conf --dest /etc/app.conf --cluster_file input/cluster_file/cluster.json
  cvs exec --cmd "sudo chown root:root /etc/app.conf" --cluster_file input/cluster_file/cluster.json

Working with directories
========================

Directory structure
-------------------

When copying directories with ``--recurse``, the entire directory structure is preserved:

.. code:: bash

  # This will create /opt/myapp/ with all subdirectories and files
  cvs scp --file ./myapp --dest /opt/myapp --recurse --cluster_file input/cluster_file/cluster.json

To copy only the contents of a directory, adjust your source path:

.. code:: bash

  # Copy contents of ./config/ to /etc/myapp/
  cvs scp --file ./config/ --dest /etc/myapp/ --recurse --cluster_file input/cluster_file/cluster.json

Large directories
-----------------

For very large directories, consider:

1. Using tar compression locally first:

.. code:: bash

  # Compress first, then copy and extract
  tar czf myapp.tar.gz ./myapp/
  cvs scp --file myapp.tar.gz --dest /tmp/myapp.tar.gz --cluster_file input/cluster_file/cluster.json
  cvs exec --cmd "cd /opt && sudo tar xzf /tmp/myapp.tar.gz" --cluster_file input/cluster_file/cluster.json

2. Reducing parallelism to avoid overwhelming the network:

.. code:: bash

  cvs scp --file ./large_directory --dest /opt/data --recurse --parallel 5 --cluster_file input/cluster_file/cluster.json

Common use cases
================

RCCL testing without shared storage
------------------------------------

When cluster nodes don't have shared mount/storage, you need to copy MPI, RCCL-tests build artifacts, and environment scripts to all nodes. Use compressed archives for efficient transfer:

.. code:: bash

  # Copy environment scripts for RCCL tests (required when no shared storage)
  cvs scp --file ~/cvs/cvs/input/env_file/rccl/ainic_env_script.sh --dest /tmp/ainic_env_script.sh --cluster_file input/cluster_file/cluster.json

  # Create compressed archives including all components (build artifacts, .so files, headers)
  tar czf openmpi.tar.gz -C /opt openmpi
  tar czf rccl-tests.tar.gz -C /opt rccl-tests

  # Copy compressed archives to all nodes
  cvs scp --file openmpi.tar.gz --dest /tmp/openmpi.tar.gz --cluster_file input/cluster_file/cluster.json
  cvs scp --file rccl-tests.tar.gz --dest /tmp/rccl-tests.tar.gz --cluster_file input/cluster_file/cluster.json

  # Extract archives on all nodes
  cvs exec --cmd "sudo tar xzf /tmp/openmpi.tar.gz -C /opt" --cluster_file input/cluster_file/cluster.json
  cvs exec --cmd "sudo tar xzf /tmp/rccl-tests.tar.gz -C /opt" --cluster_file input/cluster_file/cluster.json

  # Clean up temporary archives
  cvs exec --cmd "rm /tmp/*.tar.gz" --cluster_file input/cluster_file/cluster.json