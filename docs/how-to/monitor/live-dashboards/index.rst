.. meta::
  :description: Live GPU cluster dashboards with CVS
  :keywords: CVS, monitoring, dashboard, Prometheus, SSH

****************
Live dashboards
****************

Live dashboards keep GPU and network metrics visible while workloads run—real-time views and historical trends in Grafana or the CVS Cluster Monitor UI. CVS ships two approaches under ``cvs/monitors/``:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Approach
     - How it works
     - Best for
   * - :doc:`Agentless (SSH) </how-to/monitor/live-dashboards/agentless>`
     - Dashboard polls nodes over SSH (``amd-smi``, RDMA, logs); no exporters on workers
     - Quick dashboard, minimal node footprint, MI300/MI325 clusters with SSH access
   * - :doc:`Agent (Exporters) </how-to/monitor/live-dashboards/exporters>`
     - Installs Prometheus exporters on nodes; Fleet Monitor UI + Grafana + Loki
     - Large fleets, historical retention, Slurm/K8s control-plane views

Both support jump hosts and parallel SSH. Choose agentless when you cannot install services on compute nodes; choose exporters when you need Prometheus retention, alerting, and multi-week trends.

For a one-time HTML health report (no dashboard), use :doc:`/how-to/monitor/health-reports/index`.
