.. meta::
  :description: Deploy Prometheus exporters on AMD Instinct GPU cluster nodes for live fleet dashboards with Grafana, Loki, and Slurm metrics using CVS.
  :keywords: CVS, ROCm, metrics_exp, Prometheus, Grafana, exporters, AMD Instinct, GPU, AMD, Loki, InfiniBand, RDMA, monitoring

*****************************************************************************************
Deploy live GPU fleet dashboards with Cluster Validation Suite (CVS) Prometheus exporters
*****************************************************************************************

The GPU Fleet and Control Plane Monitor (``cvs/monitors/metrics_exp/``) deploys exporters on cluster nodes and stores time-series metrics in Prometheus, with Grafana dashboards and Loki log aggregation.

Use this for long-running fleet visibility at scale (hundreds to 1000+ nodes), Slurm or Kubernetes control-plane metrics, and retention-based trending.

Architecture
============

The stack is composed of the following components.

- **Fleet Monitor UI** (port 30080): Manage monitoring servers, GPU node groups, and control-plane groups; install exporters over SSH
- **Monitoring server**: Prometheus (30090), Grafana (30030), Loki (30100)
- **GPU nodes**: AMD Device Metrics Exporter (5000), node_exporter (9100), RDMA exporter (9417), user-activity exporter (9420), Promtail
- **Slurm head/login**: Slurm exporter (9418), node_exporter, Promtail
- **Kubernetes control plane**: K8s CP exporter (9419), node_exporter, Promtail

Prerequisites
=============

The following prerequisites are required.

- Docker Engine 23.0 or later with Compose v2 (``docker compose``) on the Fleet Monitor or monitoring server host
- SSH from the Fleet Monitor server to GPU and control-plane nodes (optional jump host)
- GPU nodes: ROCm driver (``/dev/kfd`` and ``/dev/dri`` present), Docker or
  Podman (for the AMD Device Metrics Exporter container), Python 3 (for the
  RDMA and user-activity exporters)
- Control-plane nodes (Slurm or Kubernetes): Python 3 (for the Slurm and
  Kubernetes control-plane exporters)

Deploy the monitoring stack
===========================

1. Copy the example environment file and edit it to set passwords and retention settings:

   .. code:: bash

     cp .env.example .env

2. Build and start the stack from ``cvs/monitors/metrics_exp/``:

   .. code:: bash

     docker compose build --no-cache
     docker compose up -d

3. Open Fleet Monitor at ``http://<server>:30080``.

4. Go to **Monitoring Servers**, add a server, and select **Install Stack** to deploy Prometheus, Grafana, Loki, and the pre-built dashboards.

5. Go to **Node Groups**, add your GPU nodes, upload the SSH key, select **Verify Connectivity**, then select **Install** to deploy the exporters on every node in the group.

6. (Optional) Go to **Control Node Groups**, add your Slurm or Kubernetes control-plane nodes, and select **Install**.

7. Open Grafana at ``http://<monitoring-server>:30030`` and sign in with the default credentials (username ``admin``, password ``admin``).

Dashboards
==========

Pre-provisioned folders include GPU Fleet Monitoring (utilization, thermal/power, health, RDMA, logs) and Control Plane Monitoring (Slurm and Kubernetes). See `cvs/monitors/metrics_exp/README.md <https://github.com/ROCm/cvs/blob/main/cvs/monitors/metrics_exp/README.md>`_ for metrics catalog, storage planning, firewall ports, and debugging.

For a lightweight SSH-only dashboard without node agents, see :doc:`/how-to/monitor/live-dashboards/agentless`.
