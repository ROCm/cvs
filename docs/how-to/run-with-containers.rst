.. meta::
  :description: Run CVS test suites against a per-host container backend
  :keywords: CVS, container, docker, rvs, orchestrator, run

****************************************
Run CVS tests with the container backend
****************************************

CVS can route workload commands through a long-lived per-host container instead of running them directly on the host filesystem. This is useful when you want to validate the same image you ship to production, keep the host footprint minimal (Docker, GPU driver, and SSH only), or pin the test environment byte-for-byte.

.. note::

  **Scope today.** Of the test suites shipped with CVS, only ``rvs_cvs`` consumes the orchestrator and honors the ``orchestrator`` key in the cluster file. Other ``cvs run`` test suites and the ``cvs exec`` CLI run on the host regardless of the ``orchestrator`` value. See the Backends section in :doc:`/reference/configuration-files/cluster-file` for the full picture.

Prerequisites
=============

On every cluster node:

- **Docker** installed, with passwordless ``sudo docker`` for the SSH user.
- **Host driver** loaded so ``/dev/kfd``, ``/dev/dri/*``, and ``/dev/infiniband/*`` (when RDMA is in scope) are present for passthrough.
- **SSH user home directory** reachable. The orchestrator mounts ``~/.ssh`` as ``/host_ssh`` inside the container so that the in-container ``sshd`` on port ``2224`` can authenticate.
- **Container image** either pre-loaded on every node (``docker load``) or pullable from a reachable registry. The image must contain ``openssh-server`` and the workload binaries the suite invokes (for example ``/opt/rocm/bin/rvs``).

On the head node where you launch ``cvs run``:

- CVS installed (see :doc:`/install/cvs-install`).
- SSH key-based access to every cluster node as the SSH user.

Step 1: Copy the cluster template
=================================

CVS ships a ``cluster_container.json`` template alongside the baremetal ``cluster.json``. Copy it to a working location:

.. code:: bash

  cvs copy-config cluster_container.json --output /tmp/cvs/input/cluster_file/cluster_container.json

You can list every available template with:

.. code:: bash

  cvs copy-config --list

Step 2: Edit the placeholders
=============================

Open the copied file and edit:

- ``{user-id}``: your SSH user (or leave it for runtime resolution).
- ``priv_key_file``: absolute path to your SSH private key.
- ``head_node_dict.mgmt_ip`` and the keys of ``node_dict``: real IPs or hostnames of your cluster nodes. ``mgmt_ip`` must equal one of the keys in ``node_dict``.
- ``container.image``: an image present on every node or pullable from a reachable registry. The image must include ``openssh-server`` and the workload binary (for example ``rvs``).
- ``container.name``: container name on each host. For parallel runs make this per-iteration unique (for example ``cvs_iter_<run_id>``).
- ``container.launch``: see the lifecycle note below.

For the full schema and runtime argument reference, see :doc:`/reference/configuration-files/cluster-file`.

Step 3: Run a test suite
========================

Run ``rvs_cvs`` the same way you run any CVS test suite, but with the container cluster file:

.. code:: bash

  cvs run rvs_cvs \
      --cluster_file /tmp/cvs/input/cluster_file/cluster_container.json \
      --config_file cvs/input/config_file/health/mi300_health_config.json \
      --html=/var/www/html/cvs/rvs.html \
      --self-contained-html \
      --capture=tee-sys \
      --log-file=/tmp/rvs.log \
      -vvv -s

What happens during the run:

- The ``orch`` fixture in ``rvs_cvs`` reads ``orchestrator: container`` from the cluster file and constructs a ``ContainerOrchestrator``.
- If ``container.launch`` is ``true``, CVS removes any container with the same name on each host, runs the configured image with the merged ``runtime.args``, and starts an in-container ``sshd`` on port ``2224``.
- If ``container.launch`` is ``false``, CVS verifies that a container with the configured name is already running on every host and reuses it.
- All ``rvs`` invocations are routed through the container via ``docker exec`` and the in-container ``sshd``.

Step 4: Verify the run actually used the container
==================================================

Confirm that the workload ran inside the container, not on the host. From any node in the cluster, list running containers with the configured name:

.. code:: bash

  ssh <user>@<node> sudo docker ps --filter name=^cvs_container$

You should see one running container per node with the configured name and image. You can fan this check out across every node from the head node:

.. code:: bash

  cvs exec --cluster_file /tmp/cvs/input/cluster_file/cluster_container.json \
      --cmd "sudo docker ps --filter name=^cvs_container$ --format '{{.Names}} {{.Image}}'"

.. note::

  ``cvs exec`` always runs commands on the host regardless of the ``orchestrator`` value. That's exactly what you want for verification: it lets you observe the host's view of which containers are running.

Lifecycle and teardown
======================

The ``container.launch`` flag controls who owns the container lifecycle:

- ``launch: true`` - CVS starts the long-running container as part of test setup. Teardown is a no-op; CVS does not stop the container so that you can inspect it after the run. Stop and remove it manually when you are done.
- ``launch: false`` - CVS does not start anything. It verifies that a container with the configured name is already running on every host and reuses it. Teardown is a no-op.

When ``container.enabled`` is ``false``, both setup and teardown short-circuit to no-ops. See the launch truth table in :doc:`/reference/configuration-files/cluster-file` for the full state matrix.

To stop and remove a container started with ``launch: true``, run on every node:

.. code:: bash

  cvs exec --cluster_file /tmp/cvs/input/cluster_file/cluster_container.json \
      --cmd "sudo docker rm -f cvs_container"

Replace ``cvs_container`` with the actual ``container.name`` from your cluster file.

Common pitfalls
===============

- **Forgot ``enabled: true``.** The container block is present but the orchestrator silently no-ops every lifecycle call. The first ``orch.exec`` call then errors with "no container running".
- **Image without ``openssh-server``.** ``setup_sshd`` cannot start ``sshd`` on port ``2224`` and ``orch.exec`` fails to connect. Make sure the image installs ``openssh-server`` and exposes the binary at ``/usr/sbin/sshd``.
- **Image without the workload binary.** ``cvs run rvs_cvs`` invokes ``rvs`` inside the container; if the image lacks ``/opt/rocm/bin/rvs`` the test fails with a "command not found" style error.
- **``launch: true`` plus a stale container with the same name.** On rerun, ``docker run`` would refuse to start because the name is taken. CVS's launch path issues ``docker rm -f`` before ``docker run`` to handle this; if you start a container outside CVS with the same name, stop it first.
- **Port ``2224`` already bound on the host.** With ``network: host`` the in-container ``sshd`` binds to ``2224`` in the host's network namespace. If something else on the host already listens on ``2224`` the bind fails. Stop the conflicting service.
- **Picked the wrong cluster template.** Use ``cluster_container.json`` (with ``orchestrator: container``) for container mode. Using ``cluster.json`` runs the suite on the host even when the image you wanted to test is sitting on every node.

See also
========

- :doc:`/reference/configuration-files/cluster-file` - cluster file schema, container block reference, and ``runtime.args`` table.
- `cvs/input/cluster_file/README.md <https://github.com/ROCm/cvs/blob/main/cvs/input/cluster_file/README.md>`_ - in-tree reference next to the templates.
- :doc:`/how-to/run-cvs-tests` - general test running guide.
- :doc:`/how-to/execute-cluster-commands` - ``cvs exec`` documentation.
