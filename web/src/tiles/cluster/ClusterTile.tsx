import { NavLink, Route, Routes } from "react-router-dom";
import { Cpu, FileText, HardDrive, Loader2, LayoutDashboard, Network, Pause, Play, RefreshCw, Server, Share2 } from "lucide-react";
import { useCluster } from "./ClusterContext";
import { fmt } from "./format";
import DashboardPage from "./DashboardPage";
import NodeGrid from "./NodeGrid";
import GpuMetrics from "./GpuMetrics";
import NicMetrics from "./NicMetrics";
import GpuSoftware from "./GpuSoftware";
import NicSoftware from "./NicSoftware";
import LogsPage from "./LogsPage";
import TopologyPage from "./TopologyPage";

const TABS = [
  { to: "", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "nodes", label: "Nodes", icon: Server, end: false },
  { to: "gpu", label: "GPU Metrics", icon: Cpu, end: false },
  { to: "nic", label: "NIC Metrics", icon: Network, end: false },
  { to: "gpu-sw", label: "GPU SW", icon: HardDrive, end: false },
  { to: "nic-sw", label: "NIC SW", icon: HardDrive, end: false },
  { to: "logs", label: "Logs", icon: FileText, end: false },
  { to: "topology", label: "Topology", icon: Share2, end: false },
];

function Header() {
  const { live, setLive, connected, refreshing, reload, snapshot, nodes } = useCluster();
  const reachable = nodes?.nodes.filter((n) => n.probed && n.reachable).length ?? 0;
  const total = nodes?.total ?? 0;

  return (
    <div className="mb-6">
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-semibold">Cluster Monitor</h1>
        <div className="ml-auto flex items-center gap-2">
          <button
            type="button"
            onClick={() => setLive(!live)}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium ${
              live
                ? "bg-primary text-primary-foreground hover:bg-primary/90"
                : "border border-border hover:border-primary hover:text-primary"
            }`}
          >
            {live ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            {live ? "Stop" : "Start"}
          </button>
          {live && (
            <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
              <span
                className={`h-2 w-2 rounded-full ${
                  connected ? "bg-green-500" : "animate-pulse bg-amber-500"
                }`}
              />
              {connected ? "streaming" : "connecting…"}
            </span>
          )}
          <button
            type="button"
            onClick={() => void reload()}
            disabled={refreshing}
            className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm hover:border-primary hover:text-primary disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>
      </div>

      <nav className="flex items-center gap-1 border-b border-border">
        {TABS.map((t) => (
          <NavLink
            key={t.to}
            to={t.to}
            end={t.end}
            className={({ isActive }) =>
              `inline-flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition-colors ${
                isActive
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              }`
            }
          >
            <t.icon className="h-4 w-4" />
            {t.label}
          </NavLink>
        ))}
        <span className="ml-auto py-2 text-xs text-muted-foreground">
          {nodes ? `${reachable}/${total} reachable` : ""}
          {snapshot?.collected_at ? ` · metrics ${fmt(snapshot.collected_at)}` : ""}
        </span>
      </nav>
    </div>
  );
}

function ProbeOverlay() {
  const { probeReachable, probeTotal } = useCluster();
  const label = probeTotal > 0
    ? `Connecting to inventory… (${probeReachable}/${probeTotal} nodes)`
    : "Connecting to inventory…";
  return (
    <div className="flex min-h-[40vh] flex-col items-center justify-center gap-4 rounded-xl border border-border bg-card p-10 text-center">
      <Loader2 className="h-8 w-8 animate-spin text-primary font-semibold" />
      <p className="text-base font-medium text-foreground">{label}</p>
      <p className="text-sm text-muted-foreground">
        The SSH pool is probing nodes. Cluster data will appear once the initial sweep completes.
      </p>
    </div>
  );
}

function ClusterContent() {
  const { probeReady } = useCluster();
  return (
    <>
      <Header />
      {probeReady ? (
        <Routes>
          <Route index element={<DashboardPage />} />
          <Route path="nodes" element={<NodeGrid />} />
          <Route path="gpu" element={<GpuMetrics />} />
          <Route path="nic" element={<NicMetrics />} />
          <Route path="gpu-sw" element={<GpuSoftware />} />
          <Route path="nic-sw" element={<NicSoftware />} />
          <Route path="logs" element={<LogsPage />} />
          <Route path="topology" element={<TopologyPage />} />
        </Routes>
      ) : (
        <ProbeOverlay />
      )}
    </>
  );
}

export default function ClusterTile() {
  return (
    <div className="mx-auto max-w-6xl">
      <ClusterContent />
    </div>
  );
}
