.. meta::
  :description: Live GPU fleet dashboards with Prometheus exporters
  :keywords: CVS, metrics_exp, Prometheus, Grafana, exporters

**********************
Agent (Exporters)
**********************

The **GPU Fleet & Control Plane Monitor** (``cvs/monitors/metrics_exp/``) deploys exporters on cluster nodes and stores time-series metrics in Prometheus, with Grafana dashboards and Loki log aggregation.

Use this for long-running fleet visibility at scale (hundreds to 1000+ nodes), Slurm or Kubernetes control-plane metrics, and retention-based trending.

Architecture
============

- **Fleet Monitor UI** (port 30080): manage monitoring servers, GPU node groups, and control-plane groups; install exporters over SSH
- **Monitoring server**: Prometheus (30090), Grafana (30030), Loki (30100)
- **GPU nodes**: AMD Device Metrics Exporter (5000), node_exporter (9100), RDMA exporter (9417), user-activity exporter (9420), Promtail
- **Slurm head/login**: Slurm exporter (9418), node_exporter, Promtail
- **Kubernetes control plane**: K8s CP exporter (9419), node_exporter, Promtail

Prerequisites
=============

- Docker Compose on the Fleet Monitor / monitoring server host
- SSH from the Fleet Monitor server to GPU and control-plane nodes (optional jump host)
- GPU nodes: ROCm driver, Docker/Podman for the AMD metrics container

Quick start
===========

1. Deploy the stack from ``cvs/monitors/metrics_exp/``:

   .. code:: bash

     cp .env.example .env
     # Edit passwords and retention settings
     docker compose build --no-cache
     docker compose up -d

2. Open **Fleet Monitor** at ``http://<server>:30080``.

3. **Monitoring Servers** → add server → **Install Stack** (Prometheus, Grafana, Loki, pre-built dashboards).

4. **Node Groups** → add GPU nodes → upload SSH key → **Verify Connectivity** → **Install** (deploys exporters on all nodes in the group).

5. **Control Node Groups** (optional) → add Slurm or Kubernetes control plane nodes → **Install**.

6. Open Grafana at ``http://<monitoring-server>:30030`` (default ``admin`` / ``admin``).

Dashboards
==========

Pre-provisioned folders include **GPU Fleet Monitoring** (utilization, thermal/power, health, RDMA, logs) and **Control Plane Monitoring** (Slurm and Kubernetes). See ``cvs/monitors/metrics_exp/README.md`` for metrics catalog, storage planning, firewall ports, and debugging.

For a lightweight SSH-only dashboard without node agents, see :doc:`/how-to/monitor/live-dashboards/agentless`.
