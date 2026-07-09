import { useEffect, useState, useCallback } from "react";
import { X } from "lucide-react";
import {
  getClusterStatus,
  getNodeDetail,
  type ClusterStatus,
  type NodeDetail,
} from "@/shared/api";
import { useCluster } from "./ClusterContext";

// ── Color helpers ──────────────────────────────────────────────────────────

function utilColor(pct: number | null): string {
  if (pct === null) return "#9ca3af"; // gray-400
  if (pct >= 90) return "#ef4444";   // red-500
  if (pct >= 80) return "#f97316";   // orange-500
  if (pct >= 60) return "#eab308";   // yellow-500
  if (pct >= 30) return "#22c55e";   // green-500
  return "#86efac";                  // green-300
}

function memColor(pct: number | null): string {
  if (pct === null) return "#9ca3af";
  if (pct >= 95) return "#a855f7";   // purple-500
  if (pct >= 85) return "#6366f1";   // indigo-500
  if (pct >= 70) return "#3b82f6";   // blue-500
  if (pct >= 40) return "#93c5fd";   // blue-300
  return "#dbeafe";                  // blue-100
}

function tempColor(c: number | null): string {
  if (c === null) return "#9ca3af";
  if (c >= 85) return "#ef4444";
  if (c >= 75) return "#f97316";
  if (c >= 65) return "#eab308";
  if (c >= 40) return "#22c55e";
  return "#93c5fd";
}

// ── Heatmap ────────────────────────────────────────────────────────────────

interface HeatmapCell {
  value: number | null;
  color: string;
  tooltip: string;
}

interface HeatmapRow {
  label: string;  // node hostname
  cells: HeatmapCell[];
}

function Heatmap({
  title,
  rows,
  unit,
}: {
  title: string;
  rows: HeatmapRow[];
  unit: string;
}) {
  if (rows.length === 0) return null;
  const maxCols = Math.max(...rows.map((r) => r.cells.length));

  return (
    <div>
      <p className="mb-1.5 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
        {title}
      </p>
      <div className="overflow-x-auto rounded-lg border border-border bg-card p-3">
        <div className="space-y-1">
          {rows.map((row) => (
            <div key={row.label} className="flex items-center gap-1">
              <span
                className="w-28 flex-shrink-0 truncate font-mono text-xs text-muted-foreground"
                title={row.label}
              >
                {row.label}
              </span>
              <div className="flex gap-0.5">
                {Array.from({ length: maxCols }, (_, i) => {
                  const cell = row.cells[i];
                  if (!cell) {
                    return <div key={i} className="h-5 w-5 rounded-sm bg-border/30" />;
                  }
                  return (
                    <div
                      key={i}
                      className="h-5 w-5 flex-shrink-0 rounded-sm cursor-default"
                      style={{ backgroundColor: cell.color }}
                      title={cell.tooltip}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="h-3 w-3 rounded-sm bg-gray-400" />
            <span>No data</span>
          </div>
          <span>·</span>
          <span>{unit}</span>
        </div>
      </div>
    </div>
  );
}

// ── KPI card ───────────────────────────────────────────────────────────────

function KpiCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: "green" | "red";
}) {
  return (
    <div className="rounded-lg border border-border bg-card px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p
        className={`mt-0.5 text-xl font-semibold tabular-nums ${
          accent === "green"
            ? "text-green-600 dark:text-green-400"
            : accent === "red"
              ? "text-destructive"
              : ""
        }`}
      >
        {value}
      </p>
      {sub && <p className="mt-0.5 text-xs text-muted-foreground">{sub}</p>}
    </div>
  );
}

// ── Node Details Modal ─────────────────────────────────────────────────────

function NodeDetailsModal({
  host,
  onClose,
}: {
  host: string;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<NodeDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    setErr(null);
    getNodeDetail(host)
      .then(setDetail)
      .catch((e) => setErr(String(e)));
  }, [host]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-2xl max-h-[80vh] overflow-y-auto rounded-xl border border-border bg-background p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="font-mono text-base font-semibold">{host}</h2>
          <button
            onClick={onClose}
            className="rounded-md p-1 hover:bg-muted"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {!detail && !err && (
          <p className="text-sm text-muted-foreground">Loading…</p>
        )}
        {err && <p className="text-sm text-destructive">{err}</p>}

        {detail && detail.error && (
          <p className="text-sm text-destructive">{detail.error}</p>
        )}

        {detail && !detail.error && (
          <div className="space-y-4">
            {/* GPU table */}
            {detail.gpus && detail.gpus.length > 0 && (
              <div>
                <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  GPUs
                </p>
                <div className="overflow-x-auto rounded-lg border border-border">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/40">
                      <tr>
                        {["GPU", "Util%", "VRAM", "Hotspot°C", "Power W"].map((h) => (
                          <th key={h} className="px-3 py-2 text-left font-semibold text-muted-foreground">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="font-mono">
                      {detail.gpus.map((g) => (
                        <tr key={g.index} className="border-t border-border/50">
                          <td className="px-3 py-1.5">{g.index}</td>
                          <td className="px-3 py-1.5">{g.utilization_pct.toFixed(0)}%</td>
                          <td className="px-3 py-1.5">
                            {(g.mem_used_mb / 1024).toFixed(1)}/{(g.mem_total_mb / 1024).toFixed(0)} GB
                            <span className="ml-1 text-muted-foreground">
                              ({g.mem_used_pct.toFixed(0)}%)
                            </span>
                          </td>
                          <td
                            className={`px-3 py-1.5 ${g.temp_hotspot_c >= 85 ? "text-destructive" : g.temp_hotspot_c >= 75 ? "text-yellow-500" : ""}`}
                          >
                            {(g.temp_hotspot_c || g.temp_edge_c).toFixed(0)}
                          </td>
                          <td className="px-3 py-1.5">{g.power_w.toFixed(0)} W</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* RDMA Links table */}
            {detail.rdma_links && detail.rdma_links.length > 0 && (
              <div>
                <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  RDMA Links
                </p>
                <div className="overflow-x-auto rounded-lg border border-border">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/40">
                      <tr>
                        {["Device", "State", "Phys State", "Netdev"].map((h) => (
                          <th key={h} className="px-3 py-2 text-left font-semibold text-muted-foreground">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="font-mono">
                      {detail.rdma_links.map((l) => {
                        const ok = l.state === "ACTIVE";
                        return (
                          <tr key={l.device} className="border-t border-border/50">
                            <td className="px-3 py-1.5">{l.device}</td>
                            <td
                              className={`px-3 py-1.5 ${ok ? "text-green-600 dark:text-green-400" : "text-destructive"}`}
                            >
                              {l.state}
                            </td>
                            <td className="px-3 py-1.5">{l.physical_state}</td>
                            <td className="px-3 py-1.5">{l.netdev}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Dashboard node card ────────────────────────────────────────────────────

function NodeCard({
  host,
  gpus,
  hasError,
  onClick,
}: {
  host: string;
  gpus: { util: number; temp: number }[];
  hasError: boolean;
  onClick: () => void;
}) {
  const avgUtil =
    gpus.length > 0 ? gpus.reduce((s, g) => s + g.util, 0) / gpus.length : null;
  const maxTemp =
    gpus.length > 0 ? Math.max(...gpus.map((g) => g.temp)) : null;

  const statusColor = hasError
    ? "bg-destructive"
    : maxTemp !== null && maxTemp >= 85
      ? "bg-orange-500"
      : "bg-green-500";

  return (
    <button
      onClick={onClick}
      className="w-full rounded-lg border border-border bg-card p-3 text-left hover:border-primary/50 hover:shadow-sm transition-all"
    >
      <div className="mb-2 flex items-center gap-2">
        <span className={`h-2 w-2 flex-shrink-0 rounded-full ${statusColor}`} />
        <span className="flex-1 truncate font-mono text-xs font-medium" title={host}>
          {host}
        </span>
        <span className="text-xs text-muted-foreground">{gpus.length} GPU</span>
      </div>
      {avgUtil !== null && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Util</span>
            <span className="font-mono">{avgUtil.toFixed(0)}%</span>
          </div>
          <div className="h-1 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.min(avgUtil, 100)}%`,
                backgroundColor: utilColor(avgUtil),
              }}
            />
          </div>
          {maxTemp !== null && (
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Max temp</span>
              <span
                className={`font-mono ${maxTemp >= 85 ? "text-destructive" : maxTemp >= 75 ? "text-yellow-500" : ""}`}
              >
                {maxTemp.toFixed(0)}°C
              </span>
            </div>
          )}
        </div>
      )}
    </button>
  );
}

// ── Main Dashboard ─────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { snapshot, live } = useCluster();
  const [status, setStatus] = useState<ClusterStatus | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  // Fetch the aggregate status whenever the snapshot changes (lightweight call).
  useEffect(() => {
    getClusterStatus()
      .then(setStatus)
      .catch(() => undefined);
  }, [snapshot]);

  const gpu = snapshot?.gpu ?? [];

  // Build heatmap rows per metric
  const utilRows: HeatmapRow[] = gpu.map((n) => ({
    label: n.host,
    cells: (n.gpus ?? []).map((g) => ({
      value: g.utilization_pct,
      color: utilColor(g.utilization_pct),
      tooltip: `GPU ${g.index}: ${g.utilization_pct.toFixed(0)}% util, ${(g.mem_used_mb / 1024).toFixed(0)}/${(g.mem_total_mb / 1024).toFixed(0)} GB VRAM`,
    })),
  }));

  const memRows: HeatmapRow[] = gpu.map((n) => ({
    label: n.host,
    cells: (n.gpus ?? []).map((g) => ({
      value: g.mem_used_pct,
      color: memColor(g.mem_used_pct),
      tooltip: `GPU ${g.index}: ${g.mem_used_pct.toFixed(0)}% VRAM (${(g.mem_used_mb / 1024).toFixed(1)}/${(g.mem_total_mb / 1024).toFixed(0)} GB)`,
    })),
  }));

  const tempRows: HeatmapRow[] = gpu.map((n) => ({
    label: n.host,
    cells: (n.gpus ?? []).map((g) => ({
      value: g.temp_hotspot_c || g.temp_edge_c,
      color: tempColor(g.temp_hotspot_c || g.temp_edge_c),
      tooltip: `GPU ${g.index}: hotspot ${g.temp_hotspot_c.toFixed(0)}°C, edge ${g.temp_edge_c.toFixed(0)}°C`,
    })),
  }));

  const closeModal = useCallback(() => setSelectedNode(null), []);

  return (
    <div className="space-y-6">
      {/* KPI summary bar */}
      {status && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          <KpiCard label="Total Nodes" value={String(status.total_nodes)} />
          <KpiCard
            label="Healthy Nodes"
            value={status.metrics_available ? String(status.healthy_nodes) : "—"}
            sub={
              status.metrics_available && status.unhealthy_nodes > 0
                ? `${status.unhealthy_nodes} unhealthy`
                : undefined
            }
            accent={
              status.metrics_available && status.healthy_nodes === status.total_nodes
                ? "green"
                : undefined
            }
          />
          <KpiCard
            label="Unreachable"
            value={status.metrics_available ? String(status.unreachable_nodes) : "—"}
            accent={
              status.metrics_available && status.unreachable_nodes > 0 ? "red" : undefined
            }
          />
          <KpiCard
            label="Total GPUs"
            value={status.metrics_available ? String(status.total_gpus) : "—"}
          />
          <KpiCard
            label="Avg GPU Util"
            value={
              status.metrics_available && status.total_gpus > 0
                ? `${status.avg_gpu_util_pct.toFixed(1)}%`
                : "—"
            }
            sub={
              status.metrics_available && status.total_gpus > 0
                ? `${status.avg_gpu_temp_c.toFixed(0)}°C avg temp`
                : undefined
            }
          />
        </div>
      )}

      {gpu.length === 0 ? (
        <div className="rounded-xl border border-border bg-card p-8 text-center space-y-2">
          {!live ? (
            <>
              <p className="text-foreground font-medium">No metrics collected yet</p>
              <p className="text-foreground text-sm">
                Click <span className="text-primary font-semibold">Start</span> to begin live monitoring.
              </p>
            </>
          ) : (
            <>
              <p className="text-foreground font-medium">Waiting for first collection…</p>
              <p className="text-muted-foreground text-sm">
                GPU + NIC are swept in parallel across all nodes. First results arrive within the polling interval (default 60 s).
              </p>
            </>
          )}
        </div>
      ) : (
        <>
          {/* Heatmaps */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <Heatmap
              title="GPU Utilization"
              rows={utilRows}
              unit="green=low, yellow=60%, orange=80%, red=90%+"
            />
            <Heatmap
              title="GPU VRAM Usage"
              rows={memRows}
              unit="blue-100=low → blue-500 → purple=95%+"
            />
            <Heatmap
              title="GPU Temperature"
              rows={tempRows}
              unit="blue<40°C, green<65°C, yellow<75°C, orange<80°C, red≥85°C"
            />
          </div>

          {/* Node grid */}
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Nodes — click for details
            </p>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
              {gpu.map((n) => (
                <NodeCard
                  key={n.host}
                  host={n.host}
                  gpus={(n.gpus ?? []).map((g) => ({
                    util: g.utilization_pct,
                    temp: g.temp_hotspot_c || g.temp_edge_c,
                  }))}
                  hasError={!!n.error}
                  onClick={() => setSelectedNode(n.host)}
                />
              ))}
            </div>
          </div>
        </>
      )}

      {/* Node details modal */}
      {selectedNode && (
        <NodeDetailsModal host={selectedNode} onClose={closeModal} />
      )}
    </div>
  );
}
