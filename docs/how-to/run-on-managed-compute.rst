.. meta::
  :description: Run CVS test suites on managed compute. Slurm and SPUR are supported, with one task per node, a shared workspace, and an optional cluster file.
  :keywords: CVS, managed compute, Slurm, SPUR, srun, job step, ROCm, AMD Instinct

*****************************************************
Run Cluster Validation Suite (CVS) on managed compute
*****************************************************

CVS supports two ways to run test suites:

- **SSH mode** is the default. Launch ``cvs run`` from a head node with a
  ``--cluster_file``; parallel SSH carries commands to the cluster nodes.
- **Managed compute** launches ``cvs run`` as a Slurm or SPUR job step with
  one task per node. Slurm and SPUR are the supported platforms. Rank 0 runs
  pytest and writes the report. Every rank runs an HTTP agent that executes
  suite commands on its node. Suites using the orchestrator do not need SSH
  between nodes.

Managed mode applies to ``cvs run``. ``cvs exec``, ``cvs scp`` and ``cvs
monitor`` use SSH and cluster configuration. ``cvs list`` reads the local
suite catalog; it does not launch scheduler agents.

When a run is scheduler-managed
===============================

Both conditions must hold:

- CVS detects SPUR or Slurm. SPUR is detected before Slurm. Set
  ``CVS_SCHEDULER`` to ``spur``, ``slurm``, or ``bare_metal`` to override
  detection. ``bare_metal`` forces SSH mode even inside a job step. ``spur``
  or ``slurm`` still requires a job step.
- The process runs in a job step: ``SLURM_JOB_ID``, ``SLURM_STEP_ID`` and
  ``SLURM_PROCID`` are all set. SPUR exports these ``SLURM_*`` names too.

A bare ``salloc`` shell, or an ``sbatch`` script that calls ``cvs run``
without ``srun``, normally has a job ID but no step ID. CVS then uses SSH
mode and requires ``--cluster_file``. Some sites create an interactive or
batch step automatically; launch ``cvs run`` with ``srun`` or ``spur run``
to ensure one task is started on each node.

Set ``SPUR_CONTROLLER_ADDR`` when the SPUR deployment requires it.

Requirements
============

- Start ``cvs run`` with ``srun`` or ``spur run``, using ``--ntasks-per-node 1``
  and ``--mpi=none``.
- Pass ``--workspace`` or set ``CVS_WORKSPACE`` to writable storage visible at
  the same path on every node.
- Install CVS and place the configuration, threshold, and environment files at
  the same paths on every node.
- Allow network connections in both directions between rank 0 and every worker.
- Request the node and GPU count required by the suite.
- If the login node cannot create a virtual environment, update the checkout
  there and run ``make install`` inside a GPU-node job step.

Quickstart
==========

Run any suite with one task per node and a shared workspace. ``--cluster_file``
is optional.

**SPUR:**

.. code:: bash

  spur run -A <account> -p <partition> \
    -N <nodes> --ntasks-per-node 1 --gpus-per-node <gpus> \
    --exclusive -t <time-limit> --mpi=none \
    bash -lc 'source <cvs-checkout>/.cvs_venv/bin/activate &&
      cvs run <suite> \
        --config_file <shared-config.json> \
        --workspace <shared-workspace>'

**Slurm:**

.. code:: bash

  srun -A <account> -p <partition> \
    -N <nodes> --ntasks-per-node 1 --gpus-per-node <gpus> \
    --exclusive -t <time-limit> --mpi=none \
    bash -lc 'source <cvs-checkout>/.cvs_venv/bin/activate &&
      cvs run <suite> \
        --config_file <shared-config.json> \
        --workspace <shared-workspace>'

Omit ``-A <account>`` if your site does not require an account. Omit
``--gpus-per-node`` for a suite that does not use GPUs.

This example runs two short checks in ``host_configs_cvs`` on two nodes.
The suite uses the orchestrator, so it exercises the HTTP agents. Copy its
configuration to shared storage and replace every ``<changeme>`` value with
the values for your nodes:

.. code:: bash

  cvs config copy platform/host_config.json --output /shared/<user>/cvs/host_config.json

Replace ``<account>``, ``<partition>``, ``<user>`` and ``<cvs-checkout>``
below.

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

**SPUR, same host checks:**

.. code:: bash

  spur run -A <account> -p <partition> -N 2 --ntasks-per-node 1 --exclusive \
    -t 00:30:00 --mpi=none \
    bash -lc 'source <cvs-checkout>/.cvs_venv/bin/activate &&
      cvs run host_configs_cvs test_check_os_release test_check_kernel_version \
        --config_file /shared/<user>/cvs/host_config.json \
        --workspace /shared/<user>/cvs'

.. _scheduler-run-layout:

Results
=======

Only rank 0 writes the report. The run ID is the scheduler job ID, so the
artifacts are under ``<workspace>/cvs_runs/<job-id>/``. See
:ref:`pytest-html-report-and-bundle` for the files in that directory.

Use a distinct ``--workspace`` for each repeat. A new run in the same
workspace can read the previous run's agent rendezvous files, and agent
registration then fails.

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
- :doc:`/how-to/test-suites/index`
