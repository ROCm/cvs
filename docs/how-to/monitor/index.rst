.. meta::
  :description: Monitor AMD Instinct GPU cluster health with CVS using health reports, agentless SSH dashboards, or Prometheus exporter-based live dashboards.
  :keywords: CVS, ROCm, health, monitor, cluster, AMD Instinct, GPU, AMD, Prometheus, Grafana, SSH, dashboard

************************************************************************************
Monitor AMD Instinct GPU cluster health and metrics with Cluster Validation Suite (CVS)
************************************************************************************

CVS provides three monitoring paths:

- :doc:`Health reports </how-to/monitor/health-reports/index>` — run ``cvs monitor check_cluster_health`` for a point-in-time HTML health report over SSH (no agents on nodes).
- :doc:`Live dashboards </how-to/monitor/live-dashboards/index>` — real-time and historical metrics while workloads run:

  - :doc:`Agentless (SSH) </how-to/monitor/live-dashboards/agentless>` — CVS Cluster Monitor dashboard (``cvs/monitors/cluster-mon/``)
  - :doc:`Agent (Exporters) </how-to/monitor/live-dashboards/exporters>` — Prometheus/Grafana fleet monitor (``cvs/monitors/metrics_exp/``)

Choose health reports for preflight validation or triage snapshots. Choose live dashboards for ongoing visibility during training or inference campaigns.
