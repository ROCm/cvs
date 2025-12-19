.. meta::
  :description: Execute arbitrary commands on cluster nodes using CVS
  :keywords: CVS, cluster, commands, exec, SSH, arbitrary

*******************************
Execute Arbitrary Commands on Cluster Nodes
*******************************

CVS provides an ``exec`` command to execute arbitrary shell commands on all nodes in the cluster simultaneously using parallel SSH. This is useful for gathering system information, running diagnostics, or performing administrative tasks across all cluster nodes at once.

The ``exec`` command uses the same cluster configuration files as other CVS commands and supports both command-line arguments and environment variables for configuration.

Execute commands
================

To execute a command on all cluster nodes:

.. code:: bash

  cvs exec --cmd "hostname" --cluster_file input/cluster_file/cluster.json

You can also use the ``CLUSTER_FILE`` environment variable:

.. code:: bash

  CLUSTER_FILE=input/cluster_file/cluster.json cvs exec --cmd "hostname"

Command options
===============

The ``exec`` command supports these options:

- ``--cmd``: The shell command to execute on all nodes (required)
- ``--cluster_file``: Path to cluster configuration JSON file (optional if ``CLUSTER_FILE`` environment variable is set)

Examples
========

Here are some useful examples of commands you can execute across your cluster:

System information
------------------

.. code:: bash

  # Check system uptime on all nodes
  cvs exec --cmd "uptime" --cluster_file input/cluster_file/cluster.json

  # Get OS and kernel information
  cvs exec --cmd "uname -a" --cluster_file input/cluster_file/cluster.json

  # Check available memory
  cvs exec --cmd "free -h" --cluster_file input/cluster_file/cluster.json

GPU information
---------------

.. code:: bash

  # Get GPU information using rocm-smi
  cvs exec --cmd "rocm-smi --showid --showproductname --showuniqueid" --cluster_file input/cluster_file/cluster.json

  # Get GPU temperature and usage
  cvs exec --cmd "rocm-smi --showtemp --showuse" --cluster_file input/cluster_file/cluster.json

  # Get GPU memory information
  cvs exec --cmd "rocm-smi --showmeminfo vram" --cluster_file input/cluster_file/cluster.json

Network information
-------------------

.. code:: bash

  # List RDMA devices
  cvs exec --cmd "rdma res" --cluster_file input/cluster_file/cluster.json

  # Check network interfaces
  cvs exec --cmd "ip addr show" --cluster_file input/cluster_file/cluster.json

  # Check network connectivity
  cvs exec --cmd "ping -c 3 8.8.8.8" --cluster_file input/cluster_file/cluster.json

  # Show routing table
  cvs exec --cmd "ip route show" --cluster_file input/cluster_file/cluster.json

Storage and disk information
----------------------------

.. code:: bash

  # Check disk usage
  cvs exec --cmd "df -h" --cluster_file input/cluster_file/cluster.json

  # Check mounted filesystems
  cvs exec --cmd "mount | grep -E '(ext4|xfs|nfs)'" --cluster_file input/cluster_file/cluster.json

  # Check disk I/O statistics
  cvs exec --cmd "iostat -x 1 3" --cluster_file input/cluster_file/cluster.json

Process and service information
-------------------------------

.. code:: bash

  # Check running processes
  cvs exec --cmd "ps aux | head -20" --cluster_file input/cluster_file/cluster.json

  # Check system load
  cvs exec --cmd "top -b -n 1 | head -10" --cluster_file input/cluster_file/cluster.json

  # Check systemd services status
  cvs exec --cmd "systemctl status rocm" --cluster_file input/cluster_file/cluster.json

Output format
=============

The command output is displayed with each node's hostname followed by the command output, making it easy to identify which output came from which node:

.. code:: text

  Host: node01
  Linux node01 5.15.0-91-generic #101-Ubuntu SMP Tue Nov 14 13:30:08 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
  ---
  Host: node02
  Linux node02 5.15.0-91-generic #101-Ubuntu SMP Tue Nov 14 13:30:08 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
  ---

This format helps you quickly identify which node produced which output when troubleshooting or gathering information across the cluster.

Troubleshooting
===============

If you encounter connection issues:

- Verify your SSH credentials in the cluster configuration file
- Ensure SSH key-based authentication is properly set up
- Check network connectivity to all cluster nodes
- Verify the cluster file format and paths

If commands fail on specific nodes:

- Check if the command exists on those nodes
- Verify user permissions for executing the command
- Check if required packages or tools are installed on all nodes

.. tip::

  Use simple commands first (like ``hostname`` or ``uptime``) to verify connectivity before running more complex commands.