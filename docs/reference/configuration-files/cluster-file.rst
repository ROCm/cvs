.. meta::
  :description: Configure the CVS cluster file and select the execution backend
  :keywords: cluster, cluster_file, baremetal, container, docker, orchestrator, cvs

***********************
Cluster file
***********************

Each ``cvs run`` invocation is pointed at a cluster file via ``--cluster_file <path>``. The cluster file declares the SSH credentials, the node list, and the **execution backend** that runs the workload. CVS ships two starter templates:

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
     - The workload runs inside a long-lived per-host container; the host only needs Docker, GPUs, and SSH.

Copy the template that matches your cluster shape, edit the placeholders, and pass it to ``cvs run`` and ``cvs exec``. The choice of backend is made entirely in the cluster file. The ``cvs run`` invocation does not change.

Backends
========

.. note::

  **Scope today.** Of the test suites shipped with CVS, only ``rvs_cvs`` consumes the orchestrator and honors the ``orchestrator`` key in the cluster file. All other ``cvs run`` test suites and the ``cvs exec`` CLI run on the host regardless of the ``orchestrator`` value. Migrating additional suites to the orchestrator is tracked separately. Custom Python scripts can use the ``OrchestratorFactory`` API directly as an escape hatch.

Baremetal
---------

Baremetal is the default. Workload commands are sent over SSH to each node and executed directly on the host filesystem. This is the path that has always been used by CVS and is consumed by every existing test suite.

Container
---------

Container mode routes workload commands through a container runtime on each node. CVS uses host SSH to the node, then ``docker exec`` into a long-lived per-host container. Inside the container, an SSH daemon listens on port ``2224`` so that MPI-style workloads can fan out using the in-container SSH transport.

The container backend is currently consumed by ``rvs_cvs`` only. The container lifecycle (start, verify, sshd setup, teardown) runs from the test fixture in `cvs/tests/health/rvs_cvs.py <https://github.com/ROCm/cvs/blob/main/cvs/tests/health/rvs_cvs.py>`_.

Cluster file shape
==================

Both templates share the same top-level shape. The ``container`` block and the ``orchestrator`` key are only meaningful in container mode.

.. note::

  In the cluster file, ``{user-id}`` resolves to the current login user at runtime. Replace it with a literal username if you want to pin it. The placeholders ``{xx.xx.xx.xx|hostname-N}`` are example placeholders for real IPs or hostnames.

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
            "enabled": true,
            "launch": true,
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

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameter
     - Default value
     - Description
   * - ``orchestrator``
     - ``baremetal``
     - Execution backend. Set to ``container`` to route workload commands through the container runtime. Honored today only by ``rvs_cvs``; see the Backends section above.
   * - ``username``
     - ``{user-id}``
     - SSH username for all hosts. ``{user-id}`` resolves to the current login user at runtime.
   * - ``priv_key_file``
     - ``/home/{user-id}/.ssh/id_rsa``
     - Absolute path to the SSH private key used for every host.
   * - ``head_node_dict.mgmt_ip``
     - (required)
     - Head node management IP or hostname. **Must equal one of the keys in ``node_dict``.**
   * - ``env_vars``
     - ``{}``
     - Custom environment variables exported on every host before each command. Honored by the legacy parallel-SSH path; ``cvs exec`` does not export this block.
   * - ``node_dict``
     - (required)
     - Cluster member nodes keyed by public IP or hostname. Each value is ``{"bmc_ip": "NA", "vpc_ip": "..."}``. ``vpc_ip`` is the address reachable from other nodes; set it equal to the public IP if there is no separate VPC.
   * - ``container``
     - ``{}``
     - Container backend configuration. Required when ``orchestrator`` is ``container``. See the next section.

Container block
===============

The ``container`` block configures the container backend. It is consumed by the ``ContainerOrchestrator`` defined in `cvs/core/orchestrators/container.py <https://github.com/ROCm/cvs/blob/main/cvs/core/orchestrators/container.py>`_.

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Configuration parameter
     - Default value
     - Description
   * - ``enabled``
     - ``false``
     - Master switch. Must be ``true`` for the container backend to do anything. When ``false``, ``setup_containers`` and ``teardown_containers`` short-circuit to no-ops.
   * - ``launch``
     - ``false``
     - When ``true``, CVS starts the long-running containers on every host as part of test setup. When ``false``, CVS only verifies that containers with the configured name are already running on every host and never stops them.
   * - ``image``
     - (required)
     - Image with the test dependencies (for example ``rvs``) pre-installed and an ``sshd`` you can start on port ``2224``. Must be present locally on each node or pullable from a reachable registry.
   * - ``name``
     - (required)
     - Container name on each host. For parallel runs, make this per-iteration unique (for example ``cvs_iter_<run_id>``).
   * - ``runtime.name``
     - ``docker``
     - Container runtime. Today only ``docker`` is implemented. ``enroot`` is registered as a stub and is not yet functional.
   * - ``runtime.args``
     - ``{}``
     - Backend-specific runtime arguments (see the next section).

Docker ``runtime.args`` reference
=================================

When ``runtime.name`` is ``docker``, the keys below configure the underlying ``docker run`` command. Defaults are merged from ``DEFAULT_CONTAINER_ARGS`` in `cvs/core/orchestrators/container.py <https://github.com/ROCm/cvs/blob/main/cvs/core/orchestrators/container.py>`_:

- List arguments (``volumes``, ``devices``, ``cap_add``, ``security_opt``, ``group_add``, ``ulimit``) **append to** the baked-in defaults.
- Scalar arguments (``network``, ``ipc``, ``privileged``) **override** the default when set, otherwise inherit it.
- An empty ``args: {}`` already yields a working RDMA-ready container. The keys below are only needed to extend or override.

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
     - List of ``host:container[:ro]`` mounts. The container always also receives ``/home/$user:/workspace`` and ``/home/$user/.ssh:/host_ssh`` injected by the orchestrator.
   * - ``devices``
     - ``["/dev/kfd", "/dev/dri", "/dev/infiniband"]`` (appended)
     - Device passthroughs. Per-host ``/dev/infiniband/*`` is also discovered at runtime.
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

``launch`` truth table
======================

The ``launch`` flag interacts with ``enabled`` and the runtime-tracked ``container_id`` to decide whether ``teardown_containers`` actually stops anything. The behavior below is pinned by the unit test ``test_teardown_containers_short_circuits_when_launch_true`` in `cvs/core/orchestrators/unittests/test_container.py <https://github.com/ROCm/cvs/blob/main/cvs/core/orchestrators/unittests/test_container.py>`_.

.. list-table::
   :widths: 2 2 2 5
   :header-rows: 1

   * - ``enabled``
     - ``launch``
     - ``container_id``
     - Behavior of ``teardown_containers``
   * - ``false``
     - any
     - any
     - Short-circuit (no-op, returns ``True``).
   * - ``true``
     - ``true``
     - any
     - Short-circuit. Containers are treated as externally managed; CVS does not stop them.
   * - ``true``
     - ``false``
     - unset
     - Short-circuit. Nothing was started.
   * - ``true``
     - ``false``
     - set
     - Calls the runtime's ``teardown_containers`` to stop and remove the container.

Prerequisites on each cluster node
==================================

To use the container backend, every cluster node must have:

- **Docker installed** with passwordless ``sudo docker`` for the SSH user.
- **Host driver loaded** so ``/dev/kfd``, ``/dev/dri/*``, and ``/dev/infiniband/*`` (when RDMA is in scope) are present for passthrough.
- **SSH user home directory accessible**. The orchestrator mounts ``~/.ssh`` as ``/host_ssh`` and copies keys into ``/root/.ssh`` inside the container so that the in-container ``sshd`` on port ``2224`` can authenticate.
- **Container image** either pre-loaded on every node (``docker load``) or pullable from a reachable registry. The image must contain ``openssh-server`` (for the in-container ``sshd``) and the workload binaries the suite invokes (for example ``/opt/rocm/bin/rvs``).

See also
========

- :doc:`/how-to/run-with-containers` - task-oriented walkthrough for running ``rvs_cvs`` in container mode.
- `cvs/input/cluster_file/README.md <https://github.com/ROCm/cvs/blob/main/cvs/input/cluster_file/README.md>`_ - in-tree reference next to the templates.
