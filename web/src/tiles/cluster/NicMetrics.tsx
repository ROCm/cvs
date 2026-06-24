import { useMemo } from "react";
import { type ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/shared/DataTable";
import { useCluster } from "./ClusterContext";
import { humanBytes } from "./format";

// ── Row types ──────────────────────────────────────────────────────────────

interface LinkRow {
  node: string;
  device: string;
  state: string;
  physical_state: string;
  netdev: string;
}

interface StatRow extends Record<string, unknown> {
  node: string;
  device: string;
}

interface EthRow {
  node: string;
  iface: string;
  rx_bytes: number;
  tx_bytes: number;
  rx_errors: number;
  rx_dropped: number;
  tx_errors: number;
  tx_dropped: number;
  rx_packets: number;
  tx_packets: number;
}

interface ResRow {
  node: string;
  device: string;
  pd: number;
  cq: number;
  qp: number;
  cm_id: number;
  mr: number;
  ctx: number;
  srq: number;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function StateCell({ state }: { state: string }) {
  const ok = state === "ACTIVE" || state === "LINK_UP" || state === "UP";
  const warn = state === "POLLING" || state === "CONNECTING";
  return (
    <span
      className={ok ? "text-green-600 dark:text-green-400" : warn ? "text-yellow-500" : "text-destructive"}
    >
      {state || "—"}
    </span>
  );
}

function errCell(v: number) {
  return <span className={v > 0 ? "text-destructive font-semibold" : ""}>{v.toLocaleString()}</span>;
}

function KpiCard({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="rounded-lg border border-border bg-card px-4 py-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className={`mt-0.5 text-lg font-semibold tabular-nums${accent ? " text-destructive" : ""}`}>
        {value}
      </p>
    </div>
  );
}

function SectionTitle({ title }: { title: string }) {
  return <h3 className="mb-2 text-sm font-semibold text-foreground">{title}</h3>;
}

// ── Static column defs ─────────────────────────────────────────────────────

const linkColumns: ColumnDef<LinkRow, unknown>[] = [
  { accessorKey: "node", header: "Node" },
  { accessorKey: "device", header: "Device" },
  {
    accessorKey: "state",
    header: "State",
    cell: ({ getValue }) => <StateCell state={getValue() as string} />,
  },
  { accessorKey: "physical_state", header: "Phys State" },
  { accessorKey: "netdev", header: "Netdev" },
];

const ethColumns: ColumnDef<EthRow, unknown>[] = [
  { accessorKey: "node", header: "Node" },
  { accessorKey: "iface", header: "Interface" },
  {
    accessorKey: "rx_bytes",
    header: "RX",
    cell: ({ getValue }) => humanBytes(getValue() as number),
  },
  {
    accessorKey: "tx_bytes",
    header: "TX",
    cell: ({ getValue }) => humanBytes(getValue() as number),
  },
  {
    accessorKey: "rx_packets",
    header: "RX Pkts",
    cell: ({ getValue }) => (getValue() as number).toLocaleString(),
  },
  {
    accessorKey: "tx_packets",
    header: "TX Pkts",
    cell: ({ getValue }) => (getValue() as number).toLocaleString(),
  },
  {
    accessorKey: "rx_errors",
    header: "RX Err",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
  {
    accessorKey: "rx_dropped",
    header: "RX Drop",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
  {
    accessorKey: "tx_errors",
    header: "TX Err",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
  {
    accessorKey: "tx_dropped",
    header: "TX Drop",
    cell: ({ getValue }) => errCell(getValue() as number),
  },
];

const resColumns: ColumnDef<ResRow, unknown>[] = [
  { accessorKey: "node", header: "Node" },
  { accessorKey: "device", header: "Device" },
  { accessorKey: "pd", header: "PD", cell: ({ getValue }) => (getValue() as number).toLocaleString() },
  { accessorKey: "cq", header: "CQ", cell: ({ getValue }) => (getValue() as number).toLocaleString() },
  { accessorKey: "qp", header: "QP", cell: ({ getValue }) => (getValue() as number).toLocaleString() },
  { accessorKey: "cm_id", header: "CM_ID", cell: ({ getValue }) => (getValue() as number).toLocaleString() },
  { accessorKey: "mr", header: "MR", cell: ({ getValue }) => (getValue() as number).toLocaleString() },
  { accessorKey: "ctx", header: "CTX", cell: ({ getValue }) => (getValue() as number).toLocaleString() },
  { accessorKey: "srq", header: "SRQ", cell: ({ getValue }) => (getValue() as number).toLocaleString() },
];

// ── Main component ─────────────────────────────────────────────────────────

export default function NicMetrics() {
  const { snapshot, live } = useCluster();
  const nic = snapshot?.nic ?? [];

  // RDMA Links flat rows
  const linkRows = useMemo<LinkRow[]>(() =>
    nic.flatMap((n) =>
      (n.rdma_links ?? []).map((l) => ({
        node: n.host,
        device: l.device,
        state: l.state,
        physical_state: l.physical_state,
        netdev: l.netdev,
      })),
    ), [nic]);

  // RDMA Stats flat rows + dynamic column keys
  const { statRows, statKeys } = useMemo(() => {
    const keySet = new Set<string>();
    const rows: StatRow[] = nic.flatMap((n) =>
      (n.rdma_stats ?? []).map((d) => {
        Object.keys(d.stats).forEach((k) => keySet.add(k));
        return { node: n.host, device: d.device, ...d.stats } as StatRow;
      }),
    );
    return { statRows: rows, statKeys: Array.from(keySet).sort() };
  }, [nic]);

  // Build dynamic RDMA stat columns
  const statColumns = useMemo<ColumnDef<StatRow, unknown>[]>(() => {
    const isErr = (k: string) =>
      k.includes("error") || k.includes("discard") || k.includes("drop") || k.includes("cqe");
    return [
      { accessorKey: "node", header: "Node" },
      { accessorKey: "device", header: "Device" },
      ...statKeys.map((k) => ({
        accessorKey: k,
        header: k,
        cell: ({ getValue }: { getValue: () => unknown }) => {
          const v = (getValue() ?? 0) as number;
          return isErr(k) ? errCell(v) : <span>{v.toLocaleString()}</span>;
        },
      })),
    ];
  }, [statKeys]);

  // Ethtool flat rows
  const ethRows = useMemo<EthRow[]>(() =>
    nic.flatMap((n) =>
      (n.eth_stats ?? []).map((s) => ({
        node: n.host,
        iface: s.iface,
        rx_bytes: s.stats.rx_bytes ?? 0,
        tx_bytes: s.stats.tx_bytes ?? 0,
        rx_packets: s.stats.rx_packets ?? 0,
        tx_packets: s.stats.tx_packets ?? 0,
        rx_errors: s.stats.rx_errors ?? 0,
        rx_dropped: s.stats.rx_dropped ?? 0,
        tx_errors: s.stats.tx_errors ?? 0,
        tx_dropped: s.stats.tx_dropped ?? 0,
      })),
    ), [nic]);

  // RDMA Resources flat rows
  const resRows = useMemo<ResRow[]>(() =>
    nic.flatMap((n) =>
      (n.rdma_resources ?? []).map((r) => ({
        node: n.host,
        device: r.device,
        pd: r.values.pd ?? 0,
        cq: r.values.cq ?? 0,
        qp: r.values.qp ?? 0,
        cm_id: r.values.cm_id ?? 0,
        mr: r.values.mr ?? 0,
        ctx: r.values.ctx ?? 0,
        srq: r.values.srq ?? 0,
      })),
    ), [nic]);

  // KPI aggregates
  const totalLinks = linkRows.length;
  const activeLinks = linkRows.filter((r) => r.state === "ACTIVE").length;
  const downLinks = linkRows.filter((r) => r.state === "DOWN").length;
  const activeQPs = resRows.reduce((s, r) => s + r.qp, 0);

  if (nic.length === 0) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center space-y-2">
        {!live ? (
          <>
            <p className="text-foreground font-medium">No NIC metrics yet</p>
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
        <KpiCard label="RDMA Links" value={String(totalLinks)} />
        <KpiCard label="Active" value={String(activeLinks)} />
        <KpiCard label="Down" value={String(downLinks)} accent={downLinks > 0} />
        <KpiCard label="Active QPs" value={activeQPs.toLocaleString()} />
      </div>

      {/* RDMA Links */}
      {linkRows.length > 0 && (
        <div>
          <SectionTitle title="RDMA Links" />
          <DataTable
            data={linkRows}
            columns={linkColumns}
            searchPlaceholder="Filter by node, device, or state…"
          />
        </div>
      )}

      {/* RDMA Statistics */}
      {statRows.length > 0 && (
        <div>
          <SectionTitle title="RDMA Statistics" />
          <DataTable
            data={statRows}
            columns={statColumns}
            searchPlaceholder="Filter by node or device…"
            hideZeroCols
          />
        </div>
      )}

      {/* Ethtool Statistics */}
      {ethRows.length > 0 && (
        <div>
          <SectionTitle title="Ethtool Statistics" />
          <DataTable
            data={ethRows}
            columns={ethColumns}
            searchPlaceholder="Filter by node or interface…"
            hideZeroCols
          />
        </div>
      )}

      {/* RDMA Resources */}
      {resRows.length > 0 && (
        <div>
          <SectionTitle title="RDMA Resources" />
          <DataTable
            data={resRows}
            columns={resColumns}
            searchPlaceholder="Filter by node or device…"
          />
        </div>
      )}
    </div>
  );
}
