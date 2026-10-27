.. meta::
  :description: Run live AMD Instinct GPU cluster dashboards using the CVS Cluster Monitor over SSH without installing exporters or agents on GPU nodes.
  :keywords: CVS, ROCm, cluster-mon, dashboard, SSH, agentless, AMD Instinct, GPU, AMD, RDMA, Docker, InfiniBand, monitoring

*****************************************************************************************
Run agentless SSH live dashboards with the Cluster Validation Suite (CVS) Cluster Monitor
*****************************************************************************************

The CVS Cluster Monitor (`cvs/monitors/cluster-mon/ <https://github.com/ROCm/cvs/tree/main/cvs/monitors/cluster-mon/>`_) is a live dashboard that polls the cluster over SSH. It does not install exporters on GPU nodes — collectors run ``amd-smi``, RDMA, and log queries remotely on a configurable interval.

Use this when you want real-time GPU/NIC views, heatmaps, topology, and logs without deploying Prometheus exporters on every host.

Features
========

This approach provides the following features.

- Real-time GPU metrics: utilization, temperature, power, memory, PCIe, ECC, XGMI
- Network: RDMA statistics, LLDP topology, NIC firmware and driver info
- Logs: filtered ``dmesg``, journal errors, userspace failures, custom grep search
- TCP reachability probes before SSH (fast skip of unreachable nodes)
- Jump host support and web-based configuration (nodes, SSH keys, polling interval)

Prerequisites
=============

The following prerequisites are required.

- Docker Engine 20.10 or later with Compose v2 on the monitoring host
- SSH access to cluster nodes (direct or through a jump host)
- ``amd-smi`` on GPU nodes (AMD MI300/MI325 or later); RDMA tools optional for network views

Docker Compose also starts a **Redis** sidecar for metric snapshots and events (no separate Redis install on the host).

Quick start (Docker)
====================

1. From your CVS checkout, go to `cvs/monitors/cluster-mon/ <https://github.com/ROCm/cvs/tree/main/cvs/monitors/cluster-mon/>`_.

2. Prepare configuration under ``config/``:

   .. code:: bash

     cp config/cluster.yaml.example config/cluster.yaml
     cp config/nodes.txt.example config/nodes.txt

   Edit ``cluster.yaml`` (SSH user, key path, polling interval, optional jump host) and ``nodes.txt`` (one host per line). See `cvs/monitors/cluster-mon/README.md <https://github.com/ROCm/cvs/blob/main/cvs/monitors/cluster-mon/README.md>`_ for the full schema.

3. Build and deploy (recommended):

   .. code:: bash

     ./full-rebuild.sh

   The script builds the image, runs ``docker-compose up -d``, seeds config from the examples if missing, and triggers a config reload so monitoring starts automatically.

   Alternatively, after editing ``config/`` manually:

   .. code:: bash

     docker-compose up -d --build

4. Open the dashboard at ``http://<monitor-host>:8005`` (host port **8005** maps to the app inside the container on port 8001).

5. (Optional) Change settings without rebuilding: open the **Configuration** tab, upload SSH keys if needed, edit nodes or jump-host settings, then **Save Configuration and Start Monitoring**.

Verify deployment
=================

.. code:: bash

  docker-compose logs -f
  curl http://<monitor-host>:8005/health

The health endpoint reports collection status (for example ``ssh_manager``, ``collecting``, connected clients).

Operational notes
=================

Keep the following in mind when operating this feature.

- Default metrics interval: 60 seconds (``polling.interval`` in ``cluster.yaml`` or ``POLLING__INTERVAL`` env var). For large fleets (50+ nodes), consider 120 seconds.
- Host reachability is re-probed every 5 minutes; SSH clients refresh when nodes come back online.
- Stop or restart: ``docker-compose down`` or ``docker-compose restart``.
- LLDP packages can be installed cluster-wide from the **Configuration** tab.

Other deployment options
========================

`cvs/monitors/cluster-mon/DEPLOYMENT.md <https://github.com/ROCm/cvs/blob/main/cvs/monitors/cluster-mon/DEPLOYMENT.md>`_ covers bare-metal (Python backend + React frontend), Nginx reverse proxy, systemd, resource limits, upgrades, and troubleshooting. 

.. note:: 
  
  You should use ``docker-compose`` and port **8005** on the host — some older examples in that file reference port 8001 on the host.

To monitor with Prometheus and Grafana using exporters installed on nodes, see :doc:`/how-to/monitor/live-dashboards/exporters`.
