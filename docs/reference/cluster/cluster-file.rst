.. meta::
  :description: Reference for the CVS cluster file JSON format, including SSH credentials, node topology, baremetal and container backend selection, and Docker runtime options.
  :keywords: CVS, cluster file, JSON, baremetal, container, Docker, orchestrator, SSH, ROCm, RDMA, node topology

********************************************************************************
Cluster Validation Suite (CVS) cluster file: configuration and backend selection
********************************************************************************

In SSH mode, each ``cvs run`` invocation is pointed at a cluster file using ``--cluster_file <path>``. The cluster file declares the SSH credentials, the node list, and an ``orchestrator`` value. Which suites follow that value is in the scope table below. CVS ships two starter templates:

.. note::

  Under a Slurm or SPUR job step ``--cluster_file`` is optional. See :doc:`/how-to/run-on-managed-compute`.

CVS ships two starter templates for the cluster file.

.. list-table::
   :widths: 3 2 6
   :header-rows: 1

   * - Template
     - Backend
     - Use when
   * - ``cluster.json``
     - ``baremetal`` (default)
     - ROCm, RVS, RCCL, and other workload binaries are installed on the host filesystem.
   * - ``cluster_container.json``
     - ``container``
     - A suite that runs its workload in a per-host container. The host needs Docker, GPUs, and SSH. ``per_run`` (the default lifetime) removes that container when the run ends.

Copy the template that matches the suite you are running, edit the placeholders, and pass it to ``cvs run``. ``cvs exec`` and ``cvs scp`` read the same file for hosts and SSH credentials and ignore ``orchestrator``.

For SSH-mode runs, configure :doc:`/reference/cluster/passwordless-ssh` before running CVS against the cluster.

Prerequisites on each cluster node
==================================

Every cluster node must meet the following requirements before using the
container backend.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Requirement
     - Details
   * - Docker installed
     - The SSH user needs passwordless ``sudo docker`` or direct Docker
       access (for example, membership in the ``docker`` group). CVS
       probes ``sudo -n true`` once per orchestrator and caches the result.
   * - Host ROCm driver loaded
     - ``/dev/kfd`` and ``/dev/dri/*`` must be present for GPU passthrough.
       ``/dev/infiniband/*`` is also required when the suite uses RDMA.
   * - SSH directory accessible
     - Required only when the suite starts in-container sshd (multinode
       MPI suites). CVS mounts ``/home/<cvs-user>/.ssh`` at ``/host_ssh``
       inside the container and copies it to ``/root/.ssh``. Single-node
       suites and managed-compute runs skip sshd setup entirely and do not
       need this directory or ``openssh-server`` in the image.
   * - Container image available
     - The image must be pre-loaded on each node, provided as ``image_tar``
       in the cluster file, or pullable from a reachable registry. Multinode
       suites that call sshd setup require ``/usr/sbin/sshd`` in the image.
       The image must also provide the binaries the suite runs (for example
       ``/opt/rocm/bin/rvs`` for ``rvs_cvs``).


Backends
========

.. include:: /_includes/orchestrator-scope.rst

Baremetal
---------

Baremetal is the default when a suite honors ``orchestrator`` and you leave the key unset. Commands go over SSH and run on the host filesystem.

Container
---------

When a suite runs a container, CVS reaches each node and starts one container there. In SSH mode that reachability is SSH. On a Slurm or SPUR job step it is the per-node HTTP agent (see :doc:`/how-to/run-on-managed-compute`). Later workload commands are ``docker exec`` into that container (``docker exec <name> bash -c <cmd>``). ``per_run``, the default ``container.lifetime``, starts the container at setup and removes it at teardown.

``docker exec`` does not use port 2224. CVS starts ``sshd`` inside the container on port 2224 only so ``mpirun`` can SSH to peer containers (``-p 2224``). That step copies ``/host_ssh`` to ``/root/.ssh`` and runs ``/usr/sbin/sshd -p2224``. It is skipped on a single node, and on a Slurm or SPUR job step. The health suites, RCCL, ``host_configs_cvs``, and ``install_ompi`` use the shared fixture, which calls this step. ``preflight_checks`` has its own fixture that does the same. ATOM calls it. vLLM skips it. Aorta, SGLang, xDiT, ``megatron_single``, ``megatron_distributed``, TorchTitan, JAX MaxText, and MORI launch a container without calling it.

Cluster file shape
==================

Both templates share the same top-level shape.

.. note::

  ``{user-id}`` is expanded to the user running CVS (``USER``, then ``LOGNAME``, then ``USERNAME``, then ``root``) when a suite loads the cluster file, and when you run ``cvs monitor check_cluster_health``. ``cvs exec`` and ``cvs scp`` leave the characters ``{user-id}`` as the SSH user and key path. Put a literal username in a file you pass to those two commands. The placeholders ``{xx.xx.xx.xx|hostname-N}`` are examples for real IPs or hostnames.

.. dropdown:: ``cluster_container.json``

  .. code:: json

    {
        "orchestrator": "container",
        "username": "{user-id}",
        "priv_key_file": "/home/{user-id}/.ssh/id_rsa",

        "head_node_dict": {
            "mgmt_ip": "{xx.xx.xx.xx|hostname-1}"
        },

        "env_vars": {},

        "node_dict": {
            "{xx.xx.xx.xx|hostname-1}": {
                "bmc_ip": "NA",
                "vpc_ip": "{xx.xx.xx.xx|hostname-1}"
            },
            "{xx.xx.xx.xx|hostname-2}": {
                "bmc_ip": "NA",
                "vpc_ip": "{xx.xx.xx.xx|hostname-2}"
            }
        },

        "container": {
            "lifetime": "per_run",
            "image": "rocm/cvs:latest",
            "name": "cvs_container",
            "runtime": {
                "name": "docker",
                "args": {
                    "network": "host",
                    "ipc": "host",
                    "privileged": true
                }
            }
        }
    }

Top-level parameters
--------------------

The following table describes every key accepted at the top level of the cluster file.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameter
     - Default value
     - Description
   * - ``orchestrator``
     - ``baremetal``
     - ``baremetal`` or ``container``. Suites in the scope table either follow it, force ``container``, or ignore it.
   * - ``username``
     - ``{user-id}``
     - SSH username for all hosts. Expanded for ``cvs run`` and ``cvs monitor check_cluster_health``. Left literal by ``cvs exec`` and ``cvs scp``.
   * - ``priv_key_file``
     - ``/home/{user-id}/.ssh/id_rsa``
     - Absolute path to the SSH private key used for every host. The same ``{user-id}`` rule as ``username``.
   * - ``head_node_dict.mgmt_ip``
     - (optional)
     - Stored and schema-checked. The orchestrator does not SSH here. See "Which host is the head" below.
   * - ``env_vars``
     - ``{}``
     - Shell ``export`` prefix on host SSH commands, including ``cvs exec`` and the ``docker`` CLI calls CVS makes over SSH. This is the host shell. It is not passed as ``docker -e``. ``cvs scp`` does not read it. See ``container.env`` below.
   * - ``node_dict``
     - (required)
     - Cluster member nodes keyed by public IP or hostname. Each value is ``{"bmc_ip": "NA", "vpc_ip": "..."}``. ``vpc_ip`` is the address reachable from other nodes; set it equal to the public IP if there is no separate VPC. In scheduler-managed runs CVS adds ``agent_port`` to each node.
   * - ``agent_token_file``
     - (generated)
     - Written by CVS in scheduler-managed runs only; path to the agent bearer token. Do not set by hand.
   * - ``container``
     - ``{}``
     - Container backend configuration. Required when a suite builds a container orchestrator. See the next section.

Which host is the head
----------------------

CVS uses three different hosts. Setting one does not change the others.

.. list-table::
   :widths: 3 7
   :header-rows: 1

   * - Host
     - What it is
   * - Launch host
     - The machine where you run the ``cvs`` process. It can sit outside ``node_dict``. ``{user-id}`` and the default container name use this process's user. The SSH-key mount is ``/home/<that user>/.ssh`` on each cluster node, not the cluster ``username``.
   * - Orchestrator head
     - The first ``node_dict`` key (or, when ``node_dict`` is a list, the first entry's ``mgmt_ip``). ``exec_on_head``, the MPI hostfile, and the sudo answer below all use this host.
   * - ``head_node_dict.mgmt_ip``
     - Optional field stored on the config. The orchestrator does not select its SSH head from it. Some suites overwrite it so it matches the host they already chose as the first ``node_dict`` key.

CVS's own privileged commands — the Docker CLI calls (``docker run``, ``exec``, ``rm``, ``ps``, ``load``, ``pull``, ``login``) and the MPI hostfile write — probe ``sudo -n true`` once per orchestrator. The cached answer is the orchestrator head's result, and CVS prefixes every later privileged command on every host with ``sudo -n `` or with nothing. If the nodes disagree, CVS logs a warning and still uses the head's answer. No cluster-file key controls this.

Container block
===============

The ``container`` block configures the container backend. It is consumed by the ``ContainerOrchestrator`` defined in `cvs/core/orchestrators/container.py <https://github.com/ROCm/cvs/blob/main/cvs/core/orchestrators/container.py>`_.

The following table describes each key in the ``container`` block.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameter
     - Default value
     - Description
   * - ``lifetime``
     - ``per_run``
     - Container lifecycle policy: ``no_launch``, ``per_run``, or ``persistent``. See the truth table below.
   * - ``image``
     - (required)
     - Image tag. Setup fails when this is missing. If any host does not already have ``image``, CVS runs ``docker pull`` on every host (after ``image_tar`` or a registry login when you set those).
   * - ``name``
     - ``<launch user>_<sanitized image>``
     - Optional. Omit it and CVS builds the name from the user running ``cvs`` plus the image, with every character outside ``[A-Za-z0-9_]`` replaced by ``_``. Set it when two runs share a host, and when ``lifetime`` is ``persistent``.
   * - ``env``
     - ``GPUS=8``, ``MULTINODE=true``, plus your keys
     - Environment inside the container (``docker -e``). Your keys override those two defaults. This is not cluster ``env_vars``. ``runtime.args.env`` is applied after this map, so the same name there wins.
   * - ``image_tar``
     - (unset)
     - Path on each node. When set and any host lacks ``image``, CVS runs ``docker load`` on every host and skips registry login.
   * - ``runtime.name``
     - ``docker``
     - Container runtime. ``docker`` is the working runtime. ``enroot`` is accepted and its setup returns failure.
   * - ``runtime.args``
     - ``{}``
     - Backend-specific runtime arguments (see the next section).

Docker ``runtime.args`` reference
=================================

When ``runtime.name`` is ``docker``, the keys below configure the underlying ``docker run`` command. Defaults are merged from ``DEFAULT_CONTAINER_ARGS`` in `cvs/core/orchestrators/container.py <https://github.com/ROCm/cvs/blob/main/cvs/core/orchestrators/container.py>`_:

- List arguments (``volumes``, ``devices``, ``cap_add``, ``security_opt``, ``group_add``, ``ulimit``) **append to** the baked-in defaults.
- Scalar arguments (``network``, ``ipc``, ``privileged``) **override** the default when set, otherwise inherit it.
- ``env`` is a map of extra ``docker -e`` flags, applied after ``container.env``.
- ``registry`` is not a ``docker run`` flag. When ``image_tar`` is unset and ``registry`` is set, CVS runs ``docker login`` on every node before the pull. ``username`` and ``password_file`` are required; ``password_file`` is a path on each node; ``server`` is optional.
- An empty ``args: {}`` already yields a working RDMA-ready container. The keys below are only needed to extend or override.

The following table lists the supported ``runtime.args`` keys and their defaults.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Argument
     - Default value
     - Description
   * - ``network``
     - ``host``
     - ``--network`` mode. ``host`` for clusters sharing the host network stack.
   * - ``ipc``
     - ``host``
     - ``--ipc`` mode. ``host`` enables cross-process IPC required for RDMA.
   * - ``privileged``
     - ``true``
     - ``--privileged``. Required for device passthrough and RDMA.
   * - ``volumes``
     - Appended
     - List of ``host:container[:ro]`` mounts. CVS also mounts ``/home/<user running cvs>/.ssh`` at ``/host_ssh``. That user is the launch-host user, not the cluster ``username``.
   * - ``devices``
     - ``["/dev/kfd", "/dev/dri"]`` (appended)
     - Device passthroughs. CVS also passes through each device node under ``/dev/infiniband``, discovered on each host at launch; a host without ``/dev/infiniband`` gets none. Listed ``/dev/infiniband`` entries are left to that discovery.
   * - ``cap_add``
     - ``["SYS_PTRACE", "IPC_LOCK", "SYS_ADMIN"]`` (appended)
     - Linux capabilities.
   * - ``security_opt``
     - ``["seccomp=unconfined", "apparmor=unconfined"]`` (appended)
     - Security profile relaxations needed for RDMA and ptrace.
   * - ``group_add``
     - ``["video"]`` (appended)
     - Supplementary groups inside the container.
   * - ``ulimit``
     - ``["memlock=-1"]`` (appended)
     - Per-process resource limits. ``memlock=-1`` is required for RDMA.
   * - ``env``
     - (unset)
     - Extra ``-e`` flags. The same name overrides ``container.env``.
   * - ``shm_size``
     - (unset)
     - ``--shm-size``.
   * - ``user``
     - (unset)
     - ``--user`` inside the container.
   * - ``registry``
     - (unset)
     - ``username``, ``password_file``, and optional ``server`` for ``docker login`` before a pull. Used when ``image_tar`` is unset.

``lifetime`` truth table
========================

``setup_containers`` and ``teardown_containers`` branch on ``container.lifetime``. The behavior below is pinned by the per-lifetime unit tests in `cvs/core/orchestrators/unittests/test_container.py <https://github.com/ROCm/cvs/blob/main/cvs/core/orchestrators/unittests/test_container.py>`_.

The following table shows what each ``lifetime`` value does during setup and teardown.

.. list-table::
   :widths: 2 4 4
   :header-rows: 1

   * - ``lifetime``
     - ``setup_containers``
     - ``teardown_containers``
   * - ``no_launch``
     - Verify a container with the configured name is already running on every host; set ``container_id``. Never starts anything.
     - No-op. CVS does not own a container it did not launch.
   * - ``per_run`` (default)
     - Start a fresh container on every host (force-removing any stale same-named container first).
     - Force-remove the container CVS started.
   * - ``persistent``
     - Attach if the container is already running on every host. Start fresh only if it is running on no host. Running on some hosts but not all is a hard error (CVS will not force-remove the still-running hosts and destroy their overlay). Idempotent across runs.
     - No-op. The container is left running for the next run; remove it yourself when done.

.. note::

  With ``lifetime: persistent``, pin ``container.name``. The default name is the user running ``cvs``, then ``_``, then the image with characters outside ``[A-Za-z0-9_]`` replaced by ``_``. That name changes when you change the image tag, and the previous container's overlay is left behind.

Related resources
=================

- :doc:`/how-to/run-with-containers` — walkthrough for a container cluster file.
- :doc:`/how-to/test-suites/burn-in-diag/health` — health suites that follow the cluster ``orchestrator`` key.
- `cvs/input/cluster_file/README.md <https://github.com/ROCm/cvs/blob/main/cvs/input/cluster_file/README.md>`_ — in-tree reference next to the templates.
