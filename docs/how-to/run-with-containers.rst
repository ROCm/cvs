.. meta::
  :description: Run CVS test suites against a per-host Docker container backend on AMD Instinct GPU clusters to validate production images with minimal host footprint.
  :keywords: CVS, ROCm, container, Docker, RVS, orchestrator, AMD Instinct, GPU, AMD, SSH, Linux, RDMA, InfiniBand

***************************************************************************************
Run Cluster Validation Suite (CVS) test suites with a per-host Docker container backend
***************************************************************************************

Some CVS suites run workload commands inside a per-host container. Use that path when you want the test to see the image you ship, or when the host should keep only Docker, the GPU driver, and SSH. ``container.lifetime`` defaults to ``per_run``: CVS starts the container at setup and removes it at teardown.

.. include:: /_includes/orchestrator-scope.rst

Prerequisites
=============

On every cluster node:

- **Docker** installed. The SSH user needs passwordless ``sudo docker`` or direct Docker access (for example membership in the ``docker`` group). CVS probes ``sudo -n true`` once per orchestrator and caches the first ``node_dict`` key's answer. See :doc:`/reference/cluster/cluster-file`.
- **Host driver** loaded so ``/dev/kfd``, ``/dev/dri/*``, and ``/dev/infiniband/*`` (when RDMA is in scope) are present for passthrough.
- **Launch-user SSH directory** at ``/home/<user running cvs>/.ssh`` when the suite starts in-container sshd. CVS mounts that path at ``/host_ssh``.
- **Container image** already loaded, given as ``image_tar`` on each node, or pullable. A suite that starts sshd on more than one node needs ``/usr/sbin/sshd``. The image also needs the binaries that suite runs.

On the machine where you launch ``cvs run``:

- CVS installed (see :doc:`/install/install`).
- SSH key-based access to every cluster node as the cluster ``username``.

That machine is the launch host. It can be outside ``node_dict``. It is not ``head_node_dict.mgmt_ip``, and it is not automatically the orchestrator head. The orchestrator head is the first ``node_dict`` key.

Step 1: Copy the cluster template
=================================

CVS ships a ``cluster_container.json`` template alongside the baremetal ``cluster.json``. Copy it to a working location:

.. code:: bash

  cvs config copy cluster_container.json --output ~/cvs_workspace/cluster_container.json

You can browse every available template directory with:

.. code:: bash

  cvs config list-dirs

Step 2: Edit the placeholders
=============================

Replace every ``<changeme>`` placeholder in the copied file. A suite that loads the cluster file exits if any remain. Key fields to set:

- ``username`` and ``priv_key_file``. ``{user-id}`` expands to the user running ``cvs run``. ``cvs exec`` and ``cvs scp`` leave that text literal, so use a real username in a file you pass to those commands.
- ``node_dict`` keys: real IPs or hostnames. The first key is the orchestrator head.
- ``container.image``: required. CVS pulls it on hosts that do not have it, after ``image_tar`` or ``runtime.args.registry`` when you set those.
- ``container.name``: optional. The default is ``<user running cvs>_<image>``, with characters outside ``[A-Za-z0-9_]`` replaced by ``_``. Pin it for ``lifetime: persistent`` and when two runs share a host.
- ``container.lifetime``: ``no_launch``, ``per_run``, or ``persistent``. See the lifecycle note below.
- ``container.env``: variables inside the container (``docker -e``). Cluster ``env_vars`` is a host-shell ``export`` prefix, including on the ``docker`` commands themselves, and is not copied into the container.

For the full schema and runtime argument reference, see :doc:`/reference/cluster/cluster-file`.

Step 3: Run a suite that follows the cluster key
================================================

``rvs_cvs`` follows ``orchestrator`` in the cluster file. Run it the same way you run any CVS test suite, with the container cluster file:

.. code:: bash

  cvs run rvs_cvs \
      --cluster_file ~/cvs_workspace/cluster_container.json \
      --config_file ~/cvs_workspace/health/health_config.json \
      --capture=tee-sys -vvv -s

The same cluster file applies to the other health suites on :doc:`/how-to/test-suites/burn-in-diag/health` (AGFHC, TransferBench, and their installers). Suites that force a container, or that stay on the host, are listed in the scope table above.

What happens during an ``rvs_cvs`` run with ``orchestrator: container``:

- The shared fixture reads the cluster file and builds a container orchestrator.
- With ``lifetime: per_run``, CVS removes any container with the same name on each host and starts the image with the merged ``runtime.args``. The container's command is ``sleep infinity``. Workload commands are ``docker exec`` into it.
- With ``lifetime: persistent``, CVS attaches to a container already running on every host, or starts one if none is running.
- With ``lifetime: no_launch``, CVS checks that a container with the resolved name is already running on every host and reuses it.
- On more than one node, that fixture also starts sshd on port 2224. A single node skips sshd. ``rvs`` itself is ``docker exec``, not an SSH connection to port 2224.

Step 4: Verify container execution
===================================

Confirm that the workload ran inside the container. From any node, list running containers with the configured name:

.. code:: bash

  ssh <user>@<node> sudo docker ps --filter name=^cvs_container$

You should see one running container per node with the configured name and image while the run is in ``per_run`` setup, or after the run when lifetime is ``persistent`` or ``no_launch``. You can fan this check out from the launch host:

.. code:: bash

  cvs exec --cluster_file ~/cvs_workspace/cluster_container.json \
      --cmd "sudo docker ps --filter name=^cvs_container$ --format '{{.Names}} {{.Image}}'"

``cvs exec`` SSHes to each host and ignores ``orchestrator``, so this shows the host's ``docker ps``. A ``per_run`` container is already removed when the suite has finished teardown. Check during the run, or use ``persistent`` / ``no_launch`` if you need the container afterwards.

``docker exec`` and port 2224
==============================

Ordinary container commands are ``docker exec <name> bash -c <cmd>`` over the host connection. Port 2224 is a second channel. CVS starts ``/usr/sbin/sshd -p2224`` inside the container so ``mpirun`` can reach peer ranks. Setup copies ``/host_ssh`` to ``/root/.ssh`` first.

CVS skips that setup when the cluster has one node, and on a Slurm or SPUR job step. The health suites, RCCL, ``host_configs_cvs``, ``install_ompi``, ``preflight_checks``, and ATOM call it. vLLM skips it. Aorta, SGLang, xDiT, ``megatron_single``, ``megatron_distributed``, TorchTitan, JAX MaxText, and MORI do not call it. With ``network: host``, port 2224 is in the host network namespace on the suites that do start sshd.

Lifecycle and teardown
======================

``container.lifetime`` decides who owns the container:

- ``per_run`` (default) — CVS starts a fresh container at setup and force-removes it at teardown. Writes to the container overlay are gone when the run ends.
- ``persistent`` — CVS attaches when the container is already running on every host, or starts it when it is running on none. Teardown leaves it in place, so the overlay survives the run.

  - You can run ``cvs run install_rvs`` and then ``cvs run rvs_cvs`` as separate invocations and keep the install.
  - If the container is running on some hosts and missing on others, CVS stops. A relaunch would remove the containers that are still up. Remove ``container.name`` on every host and rerun, or start it on the missing hosts.
  - Pin ``container.name``. The default name changes with the image tag.
- ``no_launch`` — CVS starts nothing. It checks that the resolved name is already running on every host. Teardown leaves it in place.

The full setup/teardown matrix is in :doc:`/reference/cluster/cluster-file`.

To remove a container CVS left running (``persistent`` or ``no_launch``):

.. code:: bash

  cvs exec --cluster_file ~/cvs_workspace/cluster_container.json \
      --cmd "sudo docker rm -f cvs_container"

Replace ``cvs_container`` with your ``container.name``. ``cvs exec`` runs that on the host.

Common pitfalls
===============

- **Multi-node suite that starts sshd, image without ``/usr/sbin/sshd``.** sshd setup fails. Single-node runs skip that step. Suites that never call it do not need sshd in the image.
- **Image without the workload binary.** ``rvs_cvs`` runs ``rvs`` inside the container. A missing ``/opt/rocm/bin/rvs`` fails the test.
- **``lifetime: persistent`` without a pinned ``name``.** The default name follows the image tag. A tag change leaves the previous container in place and starts a new one.
- **Port 2224 already in use on the host.** Suites that start sshd with ``network: host`` bind 2224 in the host network namespace. Stop the other listener.
- **Cluster file says ``baremetal`` for a suite that follows the key.** The suite runs on the host. Suites that force ``container`` still start a container; see the scope table.
- **``{user-id}`` passed to ``cvs exec``.** The SSH user is the literal string ``{user-id}``.

Related resources
=================

- :doc:`/reference/cluster/cluster-file` — cluster file schema, container block, and ``runtime.args`` table.
- :doc:`/how-to/test-suites/burn-in-diag/health` — AGFHC, TransferBench, and RVS, which follow the cluster key.
- `cvs/input/cluster_file/README.md <https://github.com/ROCm/cvs/blob/main/cvs/input/cluster_file/README.md>`_ — in-tree reference next to the templates.
- :doc:`/how-to/test-suites/index` — how to invoke a suite.
- :doc:`/how-to/execute-cluster-commands` — ``cvs exec``.
