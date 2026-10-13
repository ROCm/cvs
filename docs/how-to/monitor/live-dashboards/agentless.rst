.. meta::
  :description: Run live AMD Instinct GPU cluster dashboards using the CVS Cluster Monitor over SSH without installing exporters or agents on GPU nodes.
  :keywords: CVS, ROCm, cluster-mon, dashboard, SSH, agentless, AMD Instinct, GPU, AMD, RDMA, Docker, InfiniBand, monitoring

***********************************************************************************
Agentless SSH live dashboards with the Cluster Validation Suite (CVS) Cluster Monitor
***********************************************************************************

The **CVS Cluster Monitor** (``cvs/monitors/cluster-mon/``) is a live dashboard that polls the cluster over SSH. It does not install exporters on GPU nodes — collectors run ``amd-smi``, RDMA, and log queries remotely on a configurable interval.

Use this when you want real-time GPU/NIC views, heatmaps, topology, and logs without deploying Prometheus exporters on every host.

Features
========

This approach provides the following features.

- Real-time GPU metrics: utilization, temperature, power, memory, PCIe, ECC, XGMI
- Network: RDMA statistics, LLDP topology, NIC firmware and driver info
- Logs: filtered ``dmesg``, journal errors, userspace failures, custom grep search
- TCP reachability probes before SSH (fast skip of unreachable nodes)
- Jump host support and web-based configuration (nodes, ``~/.ssh`` key path, polling interval)

Prerequisites
=============

The following prerequisites are required.

- Docker and Docker Compose v2 on the monitoring host
- SSH access to cluster nodes (direct or via jump host)
- ``amd-smi`` on GPU nodes; RDMA tools optional for network views

Docker Compose also starts a **Redis** sidecar for metric snapshots and events (no separate Redis install on the host).

Quick start (Docker)
====================

1. From your CVS checkout, go to ``cvs/monitors/cluster-mon/``.

2. Prepare configuration under ``config/``:

   .. code:: bash

     cp config/cluster.yaml.example config/cluster.yaml
     cp config/nodes.txt.example config/nodes.txt

   Edit ``cluster.yaml`` and ``nodes.txt`` (one host per line). Key paths are ``~/.ssh/<filename>`` on the monitoring host. With a jump host, ``ssh.key_file`` is unused and ``jump_host.key_file`` is the jump-host login key. ``jump_host.node_key_file`` stays on the jump host and is fetched over SFTP. See ``cvs/monitors/cluster-mon/README.md`` for the full schema.

   The private key is already in that account's ``~/.ssh``. The dashboard does not upload key files.

3. Build and deploy (recommended):

   .. code:: bash

     ./full-rebuild.sh

   Run the script as the account that owns ``~/.ssh``, not under ``sudo``. It creates ``config/cluster.yaml`` from the example when that file is missing, then reads the owner of ``config/cluster.yaml``:

   - ``CLUSTER_MON_UID`` and ``CLUSTER_MON_GID`` from ``stat``
   - ``CLUSTER_MON_HOME`` from that uid's passwd entry

   The container runs as that uid and gid. ``$CLUSTER_MON_HOME/.ssh`` is bind-mounted read-only at the same path, and ``HOME`` is set so ``~/.ssh/id_ed25519`` opens that directory. The script then builds the image, runs ``docker compose up -d``, and reloads configuration.

   Alternatively, after editing ``config/`` manually, set the three variables and start Compose yourself:

   .. code:: bash

     export CLUSTER_MON_UID=$(stat -c '%u' config/cluster.yaml)
     export CLUSTER_MON_GID=$(stat -c '%g' config/cluster.yaml)
     export CLUSTER_MON_HOME=$(getent passwd "$CLUSTER_MON_UID" | cut -d: -f6)
     sudo --preserve-env=CLUSTER_MON_UID,CLUSTER_MON_GID,CLUSTER_MON_HOME docker compose up -d --build

   ``sudo`` drops exported variables unless they are listed in ``--preserve-env``. Compose refuses to start if any of the three is missing.

4. Open the dashboard at ``http://<monitor-host>:8005`` (host port **8005** maps to the app inside the container on port 8001).

5. Optional — change settings without rebuilding: open the **Configuration** tab, edit nodes, the ``~/.ssh`` key path, or jump-host settings, then **Save Configuration and Start Monitoring**.

Verify deployment
=================

.. code:: bash

  sudo docker compose logs -f
  curl http://<monitor-host>:8005/health

The health endpoint reports collection status (for example ``ssh_manager``, ``collecting``, connected clients).

Operational notes
=================

Keep the following in mind when operating this feature.

- Default metrics interval: 60 seconds (``polling.interval`` in ``cluster.yaml``). For large fleets (50+ nodes), consider 120 seconds.
- Host reachability is re-probed every 5 minutes; SSH clients refresh when nodes come back online
- Stop: ``sudo docker compose down``. That keeps the Redis volume. SSH keys stay in ``~/.ssh`` on the host. Do not pass ``-v`` unless you intend to wipe Redis history.
- Restart after a source change: ``./full-rebuild.sh``
- LLDP packages can be installed cluster-wide from the **Configuration** tab

Other deployment options
========================

``cvs/monitors/cluster-mon/DEPLOYMENT.md`` covers bare-metal (Go backend + React UI), an Nginx reverse proxy, systemd, resource limits, upgrades, and troubleshooting. Prefer ``docker compose`` and host port **8005**. The process inside the container listens on **8001**.

For Prometheus/Grafana-based monitoring with exporters installed on nodes, see :doc:`/how-to/monitor/live-dashboards/exporters`.
