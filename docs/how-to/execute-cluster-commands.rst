.. meta::
  :description: Run ad-hoc commands across all nodes and switches in a CVS cluster
  :keywords: CVS, cluster, commands, exec, SSH, ad-hoc, cluster-wide

*********************************
Run ad-hoc cluster-wide commands
*********************************

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
- ``--target``: Scope of execution — ``computes`` (default), ``switches``, or ``all``
- ``--timeout``: Per-node command output timeout in seconds (default: ``30``). Controls how long to wait for each host's stdout after the SSH connection is established.
- ``--connect-timeout``: Per-node SSH connection timeout in seconds (default: ``15``). Unreachable hosts fail fast after this many seconds regardless of ``--timeout``.
- ``--json``: Emit results as a single JSON object instead of human-readable text. Useful for scripting and piping to ``jq``.
- ``--verbose`` / ``-v``: Show internal SSH diagnostics (``SocketDisconnectError``, ``AuthenticationError``, pruning messages). Suppressed by default to keep output clean.

Target scope
============

By default ``cvs exec`` runs on every host in ``node_dict`` (compute nodes). Use ``--target`` to change the scope:

.. list-table::
   :widths: 2 6
   :header-rows: 1

   * - ``--target``
     - Hosts targeted
   * - ``computes`` (default)
     - All entries in ``node_dict``
   * - ``switches``
     - All ``switch_trays`` listed in the ``racks`` block (requires a rack-aware cluster file)
   * - ``all``
     - Both compute nodes and switch trays

.. code:: bash

  # Compute nodes only (default — identical to old behaviour)
  cvs exec --cmd "hostname" --cluster_file input/cluster_file/cluster.json

  # Switch trays only
  cvs exec --cmd "show version" --cluster_file input/cluster_file/cluster_rack.json --target switches

  # Both compute nodes and switch trays
  cvs exec --cmd "date" --cluster_file input/cluster_file/cluster_rack.json --target all

Rack-aware cluster file
=======================

``--target switches`` and ``--target all`` require a ``racks`` block in the cluster file that lists the switch tray IPs and (optionally) per-rack SSH credentials. A reference template is provided at ``input/cluster_file/cluster_rack.json``.

If ``--target switches`` is used with a plain cluster file (no ``racks`` block), CVS prints a warning and exits without executing any command.

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

Each result is prefixed with the target type (``compute`` or ``switch``) and the host IP or hostname, making it easy to identify which output came from which node:

.. code:: text

  [compute] Host: 10.0.0.2
  Linux node01 5.15.0-91-generic #101-Ubuntu SMP Tue Nov 14 13:30:08 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
  ---
  [compute] Host: 10.0.0.3
  Linux node02 5.15.0-91-generic #101-Ubuntu SMP Tue Nov 14 13:30:08 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux
  ---

When ``--target all`` is used, switch tray output follows the compute output with a ``[switch]`` prefix:

.. code:: text

  [switch] Host: 192.168.1.1
  SONiC Software Version: SONiC.HEAD.xxx
  ---

This format helps you quickly distinguish compute and switch output when troubleshooting or gathering information across the cluster.

JSON output
===========

Pass ``--json`` to receive a single structured JSON object on stdout. All SSH diagnostics are automatically suppressed so the output is pipe-safe.

.. code:: bash

  cvs exec --cmd "hostname" --cluster_file cluster.json --json

Output schema:

.. code:: json

  {
    "command": "hostname",
    "read_timeout": 30,
    "connect_timeout": 15,
    "output": {
      "10.0.0.2": "node01\n",
      "10.0.0.3": "node02\n",
      "10.0.0.4": "ABORT: Host Unreachable Error"
    }
  }

- ``output`` is a flat ``host → string`` map. Error and timeout strings appear as values for unreachable hosts.
- For ``--target all``, compute and switch hosts are merged into the same ``output`` dict.

Pipe directly to ``jq`` for filtering:

.. code:: bash

  # Print only the output dict
  cvs exec --cmd "hostname" --json | jq '.output'

  # Get a single host's output (use bracket notation — IPs contain dots)
  cvs exec --cmd "hostname" --json | jq -r '.output["10.0.0.2"]'

  # List only hosts that succeeded (no error strings)
  cvs exec --cmd "hostname" --json | jq '[.output | to_entries[] | select(.value | test("ABORT|Error") | not) | .key]'

Timeout behaviour
=================

``--timeout`` and ``--connect-timeout`` control two distinct phases of the per-node SSH operation:

.. list-table::
   :widths: 3 3 4
   :header-rows: 1

   * - Flag
     - Phase controlled
     - Default
   * - ``--connect-timeout``
     - TCP/SSH handshake per host
     - ``15`` s
   * - ``--timeout``
     - Command output reading after connection
     - ``30`` s

For short diagnostic commands, set both to the same value to ensure unreachable hosts fail within that window:

.. code:: bash

  cvs exec --cmd "uptime" --timeout 10 --connect-timeout 10

For long-running operations (benchmarks, firmware updates), keep ``--connect-timeout`` small and raise ``--timeout`` only:

.. code:: bash

  cvs exec --cmd "./run_benchmark.sh" --timeout 600 --connect-timeout 10

.. note::

  When TCP packets to target hosts are silently dropped (e.g. no VPN/sshuttle tunnel), the effective per-host timeout is governed by ``--timeout`` (the socket-level IO timeout), not ``--connect-timeout``. Set ``--timeout`` equal to ``--connect-timeout`` for the fastest failure in that scenario.

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

If ``--target switches`` produces a warning and no output:

- Verify that the cluster file contains a ``racks`` block with ``switch_trays`` entries
- Use ``input/cluster_file/cluster_rack.json`` as a reference template

If SSH connections hang or time out:

- Reduce ``--connect-timeout`` (default ``15`` s) to fail unreachable hosts faster
- Reduce ``--timeout`` (default ``30`` s) if commands should complete quickly
- Confirm network reachability to all target hosts before running (e.g. via sshuttle or VPN)

If the output contains unexpected SSH diagnostic messages:

- These are suppressed by default; they appear only when ``--verbose`` / ``-v`` is passed
- If they still appear, check that the root logger level has not been overridden elsewhere

.. tip::

  Use ``--json | jq '.output'`` to get a clean, machine-readable summary. Combine with ``--verbose`` only when debugging connection issues.
