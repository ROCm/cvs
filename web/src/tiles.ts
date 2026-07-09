import { Activity, FlaskConical, Gauge, type LucideIcon } from "lucide-react";

export interface TileMeta {
  id: string;
  path: string;
  name: string;
  description: string;
  icon: LucideIcon;
}

export const TILES: TileMeta[] = [
  {
    id: "cvs",
    path: "/cvs",
    name: "Test Execution",
    description: "Run CVS test suites with dynamic forms and live logs.",
    icon: FlaskConical,
  },
  {
    id: "cluster",
    path: "/cluster",
    name: "Cluster Monitor",
    description: "Live, agentless GPU/NIC health and operational actions.",
    icon: Activity,
  },
  {
    id: "fleet",
    path: "/fleet",
    name: "Fleet Metrics",
    description: "Prometheus/Grafana/Loki fleet dashboards and trends.",
    icon: Gauge,
  },
];
