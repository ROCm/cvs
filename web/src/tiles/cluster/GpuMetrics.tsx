import { useMemo } from "react";
import { type ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/shared/DataTable";
import { useCluster } from "./ClusterContext";

// ── Row types ──────────────────────────────────────────────────────────────

interface PerfRow {
  node: string;
  gpu: number;
  util_pct: number;
  mem_used_gb: number;
  mem_total_gb: number;
  mem_used_pct: number;
  temp_edge_c: number;
  temp_hotspot_c: number;
  power_w: number;
}

interface PCIeRow {
  node: string;
  gpu: number;
  width: string;
  speed: string;
  bandwidth: string;
  replay: number;
  l0_rec: number;
  nak_sent: number;
  nak_rcv: number;
}

interface ECCRow {
  node: string;
  gpu: number;
  correctable: number;
  uncorrectable: number;
  deferred: number;
}

interface XGMIRow {
  node: string;
  gpu: number;
  status: string;
  error_count: number;
}

// ── Column definitions ─────────────────────────────────────────────────────

function errCell(v: number) {
  return <span className={v > 0 ? "text-destructive font-semibold" : ""}>{v.toLocaleString()}</span>;
}

const perfColumns: ColumnDef<PerfRow, unknown>[] = [
  { accessorKey: "node", header: "Node" },
  { accessorKey: "gpu", header: "GPU" },
  {
    accessorKey: "util_pct",
    header: "Util%",
    cell: ({ getValue }) => `${(getValue() as number).toFixed(0)}%`,
  },
  {
    id: "mem",
    header: "VRAM",
    accessorFn: (r) => r.mem_used_gb,
    cell: ({ row }) =>
      `${row.original.mem_used_gb.toFixed(1)} / ${row.original.mem_total_gb.toFixed(0)} GB`,
  },
  {
    accessorKey: "mem_used_pct",
    header: "VRAM%",
    cell: ({ getValue }) => `${(getValue() as number).toFixed(0)}%`,
  },
  {
    accessorKey: "temp_hotspot_c",
    header: "Hotspot°C",
    cell: ({ getValue }) => {
      const v = getValue() as number;
      const cls = v >= 85 ? "text-destructive" : v >= 75 ? "text-yellow-500" : "";
      return <span className={cls}>{v.toFixed(0)}</span>;
    },
  },
  {
    accessorKey: "temp_edge_c",
    header: "Edge°C",
    cell: ({ getValue }) => (getValue() as number).toFixed(0),
  },
  {
    accessorKey: "power_w",
    header: "Power W",
    cell: ({ getValue }) => `${(getValue() as number).toFixed(0)} W`,
  },
];

const pcieColumns: ColumnDef<PCIeRow, unknown>[] = [
  { accessorKey: "node", header: "Node" },
  { accessorKey: "gpu", header: "GPU" },
  { accessorKey: "width", header: "Width" },
  { accessorKey: "speed", header: "Speed" },
  { accessorKey: "bandwidth", header: "Bandwidth" },
  {
    accessorKey: "replay",
    header: "Replay",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
  {
    accessorKey: "l0_rec",
    header: "L0→Rec",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
  {
    accessorKey: "nak_sent",
    header: "NAK Sent",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
  {
    accessorKey: "nak_rcv",
    header: "NAK Rcv",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
];

const eccColumns: ColumnDef<ECCRow, unknown>[] = [
  { accessorKey: "node", header: "Node" },
  { accessorKey: "gpu", header: "GPU" },
  {
    accessorKey: "correctable",
    header: "Correctable",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
  {
    accessorKey: "uncorrectable",
    header: "Uncorrectable",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
  {
    accessorKey: "deferred",
    header: "Deferred",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
];

const xgmiColumns: ColumnDef<XGMIRow, unknown>[] = [
  { accessorKey: "node", header: "Node" },
  { accessorKey: "gpu", header: "GPU" },
  { accessorKey: "status", header: "Status" },
  {
    accessorKey: "error_count",
    header: "Error Count",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
];

// ── KPI card ───────────────────────────────────────────────────────────────

function KpiCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-card px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-0.5 text-lg font-semibold tabular-nums">{value}</p>
    </div>
  );
}

// ── Section header ─────────────────────────────────────────────────────────

function SectionTitle({ title }: { title: string }) {
  return (
    <h3 className="mb-2 text-sm font-semibold text-foreground">{title}</h3>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export default function GpuMetrics() {
  const { snapshot, live } = useCluster();
  const gpu = snapshot?.gpu ?? [];

  // Flatten to per-GPU rows
  const perfRows = useMemo<PerfRow[]>(() =>
    gpu.flatMap((n) =>
      (n.gpus ?? []).map((g) => ({
        node: n.host,
        gpu: g.index,
        util_pct: g.utilization_pct,
        mem_used_gb: g.mem_used_mb / 1024,
        mem_total_gb: g.mem_total_mb / 1024,
        mem_used_pct: g.mem_used_pct,
        temp_edge_c: g.temp_edge_c,
        temp_hotspot_c: g.temp_hotspot_c || g.temp_edge_c,
        power_w: g.power_w,
      })),
    ), [gpu]);

  const pcieRows = useMemo<PCIeRow[]>(() =>
    gpu.flatMap((n) =>
      (n.gpus ?? [])
        .filter((g) => g.pcie)
        .map((g) => ({
          node: n.host,
          gpu: g.index,
          width: g.pcie!.width,
          speed: g.pcie!.speed,
          bandwidth: g.pcie!.bandwidth,
          replay: g.pcie!.replay_count,
          l0_rec: g.pcie!.l0_to_recovery_count,
          nak_sent: g.pcie!.nak_sent_count,
          nak_rcv: g.pcie!.nak_received_count,
        })),
    ), [gpu]);

  const eccRows = useMemo<ECCRow[]>(() =>
    gpu.flatMap((n) =>
      (n.gpus ?? [])
        .filter((g) => g.ecc)
        .map((g) => ({
          node: n.host,
          gpu: g.index,
          correctable: g.ecc!.correctable,
          uncorrectable: g.ecc!.uncorrectable,
          deferred: g.ecc!.deferred,
        })),
    ), [gpu]);

  const xgmiRows = useMemo<XGMIRow[]>(() =>
    gpu.flatMap((n) =>
      (n.gpus ?? [])
        .filter((g) => g.xgmi)
        .map((g) => ({
          node: n.host,
          gpu: g.index,
          status: g.xgmi!.status ?? "",
          error_count: g.xgmi!.error_count,
        })),
    ), [gpu]);

  // KPI aggregates
  const totalGPUs = perfRows.length;
  const avgUtil =
    totalGPUs > 0
      ? (perfRows.reduce((s, r) => s + r.util_pct, 0) / totalGPUs).toFixed(1)
      : "—";
  const avgTemp =
    totalGPUs > 0
      ? (perfRows.reduce((s, r) => s + r.temp_hotspot_c, 0) / totalGPUs).toFixed(1)
      : "—";
  const totalECC = eccRows.reduce(
    (s, r) => s + r.correctable + r.uncorrectable + r.deferred,
    0,
  );

  if (gpu.length === 0) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center space-y-2">
        {!live ? (
          <>
            <p className="text-foreground font-medium">No GPU metrics yet</p>
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
    );
  }

  return (
    <div className="space-y-6">
      {/* KPI cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <KpiCard label="Total GPUs" value={String(totalGPUs)} />
        <KpiCard label="Avg Util" value={totalGPUs > 0 ? `${avgUtil}%` : "—"} />
        <KpiCard label="Avg Hotspot" value={totalGPUs > 0 ? `${avgTemp}°C` : "—"} />
        <KpiCard
          label="Total ECC Errors"
          value={totalECC > 0 ? totalECC.toLocaleString() : "0"}
        />
      </div>

      {/* GPU Performance */}
      <div>
        <SectionTitle title="GPU Performance" />
        <DataTable
          data={perfRows}
          columns={perfColumns}
          searchPlaceholder="Filter by node or GPU…"
        />
      </div>

      {/* PCIe */}
      {pcieRows.length > 0 && (
        <div>
          <SectionTitle title="PCIe" />
          <DataTable
            data={pcieRows}
            columns={pcieColumns}
            searchPlaceholder="Filter by node or GPU…"
            hideZeroCols
          />
        </div>
      )}

      {/* ECC / RAS */}
      {eccRows.length > 0 && (
        <div>
          <SectionTitle title="ECC / RAS Errors" />
          <DataTable
            data={eccRows}
            columns={eccColumns}
            searchPlaceholder="Filter by node or GPU…"
            hideZeroCols
          />
        </div>
      )}

      {/* XGMI */}
      {xgmiRows.length > 0 && (
        <div>
          <SectionTitle title="XGMI" />
          <DataTable
            data={xgmiRows}
            columns={xgmiColumns}
            searchPlaceholder="Filter by node or GPU…"
            hideZeroCols
          />
        </div>
      )}
    </div>
  );
}
