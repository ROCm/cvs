.. meta::
  :description: Run CVS test suites as a Slurm or SPUR job step: scheduler detection, one task per node, optional cluster file, and the generated cluster_agents.json.
  :keywords: CVS, Slurm, SPUR, srun, scheduler, job step, cluster_agents.json, ROCm, AMD Instinct

*********************************************************
Run Cluster Validation Suite (CVS) under Slurm or SPUR
*********************************************************

CVS supports two ways to run test suites:

- **SSH mode** is the default. Launch ``cvs run`` from a head node with a
  ``--cluster_file``; parallel SSH carries commands to the cluster nodes.
- **Scheduler-managed mode** launches ``cvs run`` itself as a Slurm or SPUR
  **job step** with one task per node. Rank 0 runs pytest. Every other rank
  serves an HTTP agent that runs suite commands on its node. Suites using the
  orchestrator do not need SSH between nodes.

Managed mode applies to ``cvs run``. ``cvs exec``, ``cvs scp`` and ``cvs
monitor`` use SSH and cluster configuration. ``cvs list`` reads the local
suite catalog; it does not launch scheduler agents.

When a run is scheduler-managed
===============================

Both conditions must hold:

- CVS detects a scheduler other than ``bare_metal``.
- The process runs in a job step: ``SLURM_JOB_ID``, ``SLURM_STEP_ID`` and
  ``SLURM_PROCID`` are all set. SPUR exports these ``SLURM_*`` names too.

A bare ``salloc`` shell, or an ``sbatch`` script that calls ``cvs run``
without ``srun``, normally has a job ID but no step ID. CVS then uses SSH
mode and requires ``--cluster_file``. Some sites create an interactive or
batch step automatically; launch ``cvs run`` with ``srun`` or ``spur run``
to ensure one task is started on each node.

CVS detects the scheduler in this order:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Priority
     - Detection
   * - 1
     - ``CVS_SCHEDULER``: ``spur``, ``slurm`` or ``bare_metal``. Case is ignored
       and surrounding whitespace is trimmed. Any other value raises
       ``Unknown scheduler type '<value>' in CVS_SCHEDULER, expected one of: ['spur', 'slurm', 'bare_metal']``.
   * - 2
     - ``SPUR_JOB_ID`` set: SPUR.
   * - 3
     - ``SLURM_JOB_ID`` set: Slurm.
   * - 4
     - ``spur version`` succeeds: SPUR; otherwise ``scontrol version``
       succeeds: Slurm. Each probe has a five-second timeout.
   * - 5
     - Otherwise ``bare_metal``.

Set ``CVS_SCHEDULER=bare_metal`` to force SSH mode, even inside a job step.
Set ``CVS_SCHEDULER=spur`` or ``CVS_SCHEDULER=slurm`` to override detection;
the override does not replace the job-step requirement. The type also selects
``spur run`` versus ``srun`` for nested RCCL launches. Under SPUR, pairwise
and incremental RCCL tests are skipped.

Requirements
============

- **One CVS task per node.** Pass ``--ntasks-per-node 1`` explicitly, including
  when starting a step inside an allocation that specified more tasks per node.
  CVS expands ``SPUR_NODES`` or ``SLURM_NODELIST`` and requires its host count
  to equal ``SLURM_NTASKS``.
- **Start a job step with ``--mpi=none``.** Use ``srun`` or ``spur run`` for
  ``cvs run``. Suites such as RCCL start their own nested
  ``--overlap --mpi=pmix`` steps when needed.
- **Use a shared workspace.** Pass ``--workspace`` or set ``CVS_WORKSPACE``
  to writable storage visible at the same path on every node. This is
  required for every managed run, with or without containers: ranks
  rendezvous in ``<run_dir>/agent/``.
- **Make the same CVS installation and inputs available on every node.**
  The virtual environment, ``--config_file``, and referenced threshold or
  environment files must have the same paths on every node, usually on a
  shared home directory. Build the virtual environment with a supported
  Python interpreter that also runs on the compute nodes.
- **Allow agent connections.** Every node must reach the other nodes on
  arbitrary TCP ports. Agents bind an ephemeral port on ``0.0.0.0``.
- **Request the GPUs for the outer step.** For example, use
  ``--gpus-per-node <n> --exclusive`` if your cluster defines GPU resources.
  Agent commands run inside that step and see its device limits on clusters
  that constrain devices with cgroups.

Quickstart
==========

This example runs two short checks in ``host_configs_cvs`` on two nodes.
The suite uses the orchestrator, so it exercises the HTTP agents. Copy its
configuration to shared storage and replace every ``<changeme>`` value with
the values for your nodes:

.. code:: bash

  cvs config copy platform/host_config.json --output /shared/<user>/cvs/host_config.json

Replace ``<account>``, ``<partition>``, ``<user>`` and ``<cvs-checkout>``
below. Omit ``-A <account>`` if your site does not require an account.
These host checks do not need GPUs. For GPU-intensive suites, add
``--gpus-per-node <n>`` to the outer step if your site defines GPU resources.

**Slurm, one command:**

.. code:: bash

  srun -A <account> -p <partition> -N 2 --ntasks-per-node 1 --exclusive \
    -t 00:30:00 --mpi=none \
    bash -lc 'source <cvs-checkout>/.cvs_venv/bin/activate &&
      cvs run host_configs_cvs test_check_os_release test_check_kernel_version \
        --config_file /shared/<user>/cvs/host_config.json \
        --workspace /shared/<user>/cvs'

**Slurm, interactive allocation:**

.. code:: bash

  salloc -A <account> -p <partition> -N 2 --ntasks-per-node 1 --exclusive -t 00:30:00
  srun --ntasks-per-node 1 --mpi=none \
    bash -lc 'source <cvs-checkout>/.cvs_venv/bin/activate &&
      cvs run host_configs_cvs test_check_os_release test_check_kernel_version \
        --config_file /shared/<user>/cvs/host_config.json \
        --workspace /shared/<user>/cvs'

Running ``cvs run`` directly in the allocation shell does not start a task
on each node and is normally not a managed run. Always use the ``srun`` line.

**Slurm, batch job:** Save this script and submit it with ``sbatch``.

.. code:: bash

  #!/bin/bash
  #SBATCH -N 2
  #SBATCH --ntasks-per-node=1
  #SBATCH --exclusive

  srun --mpi=none \
    bash -lc 'source <cvs-checkout>/.cvs_venv/bin/activate &&
      cvs run host_configs_cvs test_check_os_release test_check_kernel_version \
        --config_file /shared/<user>/cvs/host_config.json \
        --workspace /shared/<user>/cvs'

Add ``#SBATCH -A <account>``, ``#SBATCH -p <partition>`` and a time limit
if your site requires them. The ``srun`` line is essential: calling ``cvs
run`` directly from the batch script does not reliably start a CVS task on
each node.

**SPUR:**

.. code:: bash

  spur run -A <account> -p <partition> -N 2 --ntasks-per-node 1 --exclusive \
    -t 00:30:00 --mpi=none \
    bash -lc 'source <cvs-checkout>/.cvs_venv/bin/activate &&
      cvs run host_configs_cvs test_check_os_release test_check_kernel_version \
        --config_file /shared/<user>/cvs/host_config.json \
        --workspace /shared/<user>/cvs'

Set ``SPUR_CONTROLLER_ADDR`` if your SPUR deployment requires it.

No ``--cluster_file`` is needed in these examples. The HTML report and log
land in ``/shared/<user>/cvs/cvs_runs/$SLURM_JOB_ID/host_configs_cvs.html``
and ``host_configs_cvs.log``. See :ref:`scheduler-run-layout` for the other
artifacts.

What happens during a managed run
=================================

#. Every rank resolves the same directory,
   ``<workspace>/cvs_runs/<SLURM_JOB_ID>/``.
#. Rank 0 writes a bearer token and its host and port to ``<run_dir>/agent/``,
   starts its HTTP agent and refreshes a heartbeat every 10 seconds.
#. Other ranks wait up to 60 seconds for the rendezvous files, start their
   agents and register with rank 0.
#. Rank 0 waits up to 60 seconds for registrations, then writes
   ``cluster_agents.json``. If any host is missing, it exits before pytest
   with ``agents did not register for scheduler hosts: [...]``.
#. Rank 0 runs pytest with ``--cluster_file=<run_dir>/cluster_agents.json``.
   Orchestrator-based suites send commands to the agents over HTTP.
#. When pytest ends, rank 0 writes ``agent/rank0.done`` and workers exit 0.
   If rank 0's heartbeat is stale for three intervals, workers exit 1.

Rank 0's exit code is pytest's exit code.

.. _scheduler-cluster-agents:

The generated cluster_agents.json
=================================

Rank 0 writes ``<workspace>/cvs_runs/<job_id>/cluster_agents.json``. With
no ``--cluster_file``, a two-node job looks like this (ports vary by job):

.. code:: json

  {
    "orchestrator": "baremetal",
    "username": "{user-id}",
    "priv_key_file": "/home/{user-id}/.ssh/id_rsa",
    "node_dict": {
      "node01": {"vpc_ip": "node01", "agent_port": 41235},
      "node02": {"vpc_ip": "node02", "agent_port": 38811}
    },
    "head_node_dict": {"mgmt_ip": "node01"},
    "agent_token_file": "<workspace>/cvs_runs/<job_id>/agent/secret"
  }

You can also pass ``--cluster_file`` to supply metadata:

- Top-level keys other than ``node_dict`` and ``head_node_dict`` are kept,
  such as ``orchestrator: container`` with its ``container`` block,
  ``username``, ``priv_key_file`` and ``env_vars``.
- ``node_dict`` may list only hosts in this job. Other hosts cause
  ``cluster file contains nodes outside this scheduler job: [...]``.
  Each node's metadata must be an object.
- Per-node metadata, such as a fabric ``vpc_ip``, overrides the default
  ``vpc_ip`` (the scheduler hostname). CVS adds each node's ``agent_port``.
- ``head_node_dict`` is replaced by the job's first host. The output node
  keys and their order always come from the scheduler node list.

Do not hand-edit this file or reuse it in another job. Its ports and token
are specific to one job.

.. _scheduler-run-layout:

Run directory layout
====================

.. code:: text

  <workspace>/cvs_runs/<job_id>/
    cluster_agents.json
    agent/
      secret                    # mode 0600
      rank0.host
      rank0.port
      rank0.alive
      rank0.heartbeat_interval
      rank0.done
    <test-file-stem>.html
    <test-file-stem>.log
    exec_output/                # when a suite requests file output

Every ``cvs run`` in the same allocation shares this directory. Re-running
the same suite overwrites its ``<stem>.html`` and ``<stem>.log`` files. Pass
distinct ``--html`` and ``--log-file`` paths to keep both. A repeated run
can read stale rank-0 rendezvous files before rank 0 clears them, causing
agent registration to fail. Use a distinct ``--workspace`` for each repeat
to avoid that race; each workspace gets its own run directory and artifacts.
In test configs, ``{run_dir}`` resolves to the run directory.

Limitations
===========

- ``cvs exec``, ``cvs scp`` and ``cvs monitor`` still use SSH. ``cvs list``
  reads the local catalog and does not start scheduler agents.
- Suites that construct their own parallel-SSH handle still connect over
  SSH in a managed run. They need passwordless SSH and correct ``username``
  and ``priv_key_file`` values, supplied through ``--cluster_file`` if the
  defaults do not match. These include ``csp_qual_agfhc``,
  ``ib_perf_bw_test``, ``install_ibperf_tools``, ``install_ompi`` and the
  ``megatron_llama3_1_*`` suites.
- Under SPUR, ``test_rccl_pairwise`` and ``test_rccl_incremental`` are
  skipped. See :ref:`rccl-run-managed`.
- The container backend skips its in-container ``sshd`` setup under a
  scheduler.

Troubleshooting
===============

The last two messages below appear at the end of Python tracebacks. The
other listed errors are printed as ``Error:`` messages by ``cvs run``.

.. list-table::
   :header-rows: 1
   :widths: 43 27 30

   * - Message
     - Cause
     - Fix
   * - ``Error: --cluster_file is required outside a scheduler-managed run``
     - No job step, or ``CVS_SCHEDULER=bare_metal``.
     - Launch with ``srun`` / ``spur run``, or pass ``--cluster_file``.
   * - ``Error: A scheduler-managed run must set an explicit workspace on shared storage: pass --workspace or set CVS_WORKSPACE. ...``
     - No ``--workspace`` or ``CVS_WORKSPACE``.
     - Set one to writable shared storage.
   * - ``Error: Could not create the run directory ...``
     - Workspace is unmounted, read-only or full.
     - Use writable shared storage.
   * - ``Error: managed CVS requires one task per node: scheduler expanded N hosts but SLURM_NTASKS is M``
     - More than one task per node.
     - Pass ``--ntasks-per-node 1``.
   * - ``Error: managed CVS run requires SPUR_NODES or SLURM_NODELIST``
     - Scheduler node list is absent.
     - Launch as a job step.
   * - ``Error: cluster file contains nodes outside this scheduler job: [...]``
     - ``--cluster_file`` lists a host outside the job.
     - Keep only job hosts in ``node_dict``.
   * - ``Error: agents did not register for scheduler hosts: [...]``
     - A worker did not start, ports are blocked, or workspace is not shared.
     - Check ports, shared storage and worker stderr.
   * - ``TimeoutError: rank-0 rendezvous did not appear``
     - Workspace is not shared or rank 0 failed.
     - Check shared storage and rank 0's output.
   * - ``ValueError: Unknown scheduler type '<value>' in CVS_SCHEDULER, expected one of: ['spur', 'slurm', 'bare_metal']``
     - Invalid ``CVS_SCHEDULER`` value.
     - Use ``spur``, ``slurm`` or ``bare_metal``.

Related
=======

- :doc:`/reference/cli/cvs-run`
- :doc:`/reference/cluster/cluster-file`
- :doc:`/how-to/test-suites/network/rccl`
- :doc:`/how-to/test-suites/training/aorta`
- :doc:`/how-to/test-suites/inference/atom`
