.. meta::
  :description: Manage SSH host keys for cluster nodes using CVS
  :keywords: CVS, cluster, SSH, keys, sshkeyscan, host keys, known_hosts

*******************************
Manage SSH Host Keys for Cluster Nodes
*******************************

CVS provides an ``sshkeyscan`` command to scan and update SSH host keys for all nodes in the cluster. This is essential for maintaining proper SSH connectivity and avoiding host verification warnings when connecting to cluster nodes, especially in distributed computing environments like MPI jobs.

The ``sshkeyscan`` command uses the same cluster configuration files as other CVS commands and supports both local and remote execution modes with parallel processing for efficiency.

Overview
========

SSH host key management is critical for cluster operations because:

- **Security**: Ensures you're connecting to legitimate hosts
- **Automation**: Prevents interactive prompts that break automated scripts
- **MPI/Distributed Jobs**: Head nodes need to SSH to compute nodes without prompts
- **Consistency**: Maintains uniform SSH configuration across the cluster

The ``sshkeyscan`` command automates the process of collecting SSH host keys from all cluster nodes and adding them to your ``known_hosts`` file.

Basic usage
===========

Scan and add SSH keys for all cluster nodes:

.. code:: bash

  cvs sshkeyscan --cluster_file input/cluster_file/cluster.json

Use the ``CLUSTER_FILE`` environment variable:

.. code:: bash

  CLUSTER_FILE=input/cluster_file/cluster.json cvs sshkeyscan

Execution modes
===============

Local execution (default)
--------------------------

In local mode, ssh-keyscan runs from your current machine:

.. code:: bash

  cvs sshkeyscan --cluster_file cluster.json --at local

This mode is useful when:
- Setting up SSH keys from a management station
- You have direct network access to all cluster nodes
- Running initial cluster setup

Remote execution (head node)
-----------------------------

In head node mode, ssh-keyscan runs from the cluster's head node:

.. code:: bash

  cvs sshkeyscan --cluster_file cluster.json --at head

This mode is essential when:
- Setting up SSH keys for MPI jobs
- The head node has different network access than your local machine
- Preparing the cluster for distributed computing workloads

Command options
===============

The ``sshkeyscan`` command supports these options:

Core options
------------

- ``--cluster_file``: Path to cluster configuration JSON file (optional if ``CLUSTER_FILE`` environment variable is set)
- ``--at {local,head}``: Execution location - 'local' (current node, default) or 'head' (cluster head node)

SSH configuration
-----------------

- ``--known_hosts``: Path to known_hosts file (default: ``~/.ssh/known_hosts``)
- ``--remove-existing``: Remove existing host keys before scanning (clean slate approach)

Performance and safety
----------------------

- ``--parallel``: Number of parallel ssh-keyscan processes (default: 20)
- ``--dry-run``: Show what would be done without making changes

Examples
========

Basic SSH key management
------------------------

.. code:: bash

  # Scan all cluster nodes and add keys locally
  cvs sshkeyscan --cluster_file input/cluster_file/cluster.json

  # Scan from head node (for MPI job preparation)
  cvs sshkeyscan --cluster_file input/cluster_file/cluster.json --at head

  # Preview operations without making changes
  cvs sshkeyscan --cluster_file input/cluster_file/cluster.json --dry-run

Advanced usage
--------------

.. code:: bash

  # Clean existing keys and rescan (fresh start)
  cvs sshkeyscan --cluster_file cluster.json --at head --remove-existing

  # High-performance scanning with increased parallelism
  cvs sshkeyscan --cluster_file cluster.json --parallel 50

  # Use custom known_hosts file
  cvs sshkeyscan --cluster_file cluster.json --known_hosts /tmp/cluster_known_hosts

  # Environment variable with head node execution
  CLUSTER_FILE=cluster.json cvs sshkeyscan --at head --remove-existing

Workflow integration
--------------------

.. code:: bash

  # Complete cluster setup workflow
  # 1. Generate cluster configuration
  cvs generate cluster_json --hosts "node[01-10]" --output_json_file cluster.json --username myuser --key_file ~/.ssh/id_rsa

  # 2. Set up SSH keys for local management
  cvs sshkeyscan --cluster_file cluster.json --at local

  # 3. Set up SSH keys on head node for MPI jobs
  cvs sshkeyscan --cluster_file cluster.json --at head

  # 4. Verify connectivity
  cvs exec --cmd "hostname" --cluster_file cluster.json

Performance considerations
=========================

Parallel processing
-------------------

The ``--parallel`` option controls how many ssh-keyscan processes run simultaneously:

.. code:: bash

  # Conservative (good for slower networks)
  cvs sshkeyscan --cluster_file cluster.json --parallel 10

  # Default (balanced)
  cvs sshkeyscan --cluster_file cluster.json --parallel 20

  # Aggressive (fast networks, powerful head node)
  cvs sshkeyscan --cluster_file cluster.json --parallel 100

**Guidelines**:
- Start with the default (20) and adjust based on performance
- Higher parallelism may overwhelm slower networks or head nodes
- Monitor network and CPU usage during large-scale scans

Timeout handling
----------------

The command includes built-in timeout handling:
- SSH connections timeout after 30 seconds
- Failed connections are reported but don't stop the scan
- Retry failed nodes by running the command again

Output format
=============

The command provides detailed progress information:

.. code:: text

  SSH Key Scan Configuration:
    Cluster file: cluster.json
    Execution location: head
    Head node: 192.168.1.100
    Known hosts file: ~/.ssh/known_hosts
    Number of hosts: 10
    Parallel processes: 20
    Remove existing keys: False
    Dry run: False

  Scanning SSH host keys...
  node01: SUCCESS - Added 2 host key(s) (on remote)
  node02: SUCCESS - Added 2 host key(s) (on remote)
  node03: FAILED - Connection timeout
  node04: SUCCESS - Added 2 host key(s) (on remote)
  ...

  SSH Key Scan Summary:
    Total hosts: 10
    Successful: 9
    Failed: 1
    SSH host keys updated on: head node
    Known hosts file: ~/.ssh/known_hosts

Security considerations
=======================

Key verification
----------------

While ``sshkeyscan`` automates key collection, be aware that:

- Keys are accepted without manual verification
- Use this command only on trusted networks
- Consider manual verification for high-security environments

Best practices
--------------

1. **Run from secure locations**: Use the command from trusted management stations
2. **Regular updates**: Re-run when nodes are rebuilt or keys change
3. **Backup known_hosts**: Keep backups of your known_hosts files
4. **Use dry-run**: Always test with ``--dry-run`` in production environments
5. **Monitor failures**: Investigate and resolve connection failures promptly

Troubleshooting
===============

Connection issues
-----------------

If nodes fail to respond:

.. code:: bash

  # Check basic connectivity first
  cvs exec --cmd "hostname" --cluster_file cluster.json

  # Verify SSH access manually
  ssh -i your_key_file username@failing_node

  # Check network connectivity
  ping failing_node_ip

Common solutions:
- Verify SSH credentials in cluster configuration
- Ensure SSH service is running on target nodes
- Check firewall rules and network connectivity
- Verify SSH key permissions (600 for private keys)

Permission issues
-----------------

If you get permission errors:

.. code:: bash

  # Check known_hosts file permissions
  ls -la ~/.ssh/known_hosts

  # Ensure SSH directory exists and has correct permissions
  mkdir -p ~/.ssh
  chmod 700 ~/.ssh
  touch ~/.ssh/known_hosts
  chmod 600 ~/.ssh/known_hosts

Head node execution issues
--------------------------

If ``--at head`` fails:

1. **Verify head node configuration**: Check ``head_node_dict.mgmt_ip`` in cluster file
2. **Check SSH credentials**: Ensure ``username`` and ``priv_key_file`` are correct
3. **Test head node connectivity**: SSH to head node manually first
4. **Verify head node SSH setup**: Head node needs SSH client tools installed

Performance issues
------------------

If scanning is slow:

1. **Reduce parallelism**: Lower ``--parallel`` value for slower networks
2. **Check network bandwidth**: High parallelism may saturate network links
3. **Monitor head node resources**: CPU/memory constraints may limit performance
4. **Use local execution**: May be faster if you have better connectivity than head node

.. tip::

  Use ``--dry-run`` to test configuration and connectivity before running actual scans, especially in production environments.

.. warning::

  The ``--remove-existing`` flag will delete existing host keys before scanning. Use with caution and ensure you have backups if needed.