import { useEffect, useMemo, useState } from "react";
import {
  getTopologyLLDP,
  type LLDPNeighbor,
  type TopologySnapshot,
} from "../../shared/api";
import { DataTable } from "../../shared/DataTable";
import { type ColumnDef } from "@tanstack/react-table";

// ─── Types ───────────────────────────────────────────────────────────────────

interface FlatRow extends LLDPNeighbor {
  host: string;
}

// ─── SVG Graph ───────────────────────────────────────────────────────────────

interface GraphNode {
  id: string;
  label: string;
  isSwitch: boolean;
  x: number;
  y: number;
}

interface GraphEdge {
  fromId: string;
  toId: string;
  iface: string;
  port: string;
}

function buildGraph(snap: TopologySnapshot): { nodes: GraphNode[]; edges: GraphEdge[] } {
  // Collect unique chassis (switches) and cluster nodes
  const switchSet = new Set<string>();
  const edgeList: { hostId: string; switchId: string; iface: string; port: string }[] = [];

  for (const nd of snap.nodes) {
    if (!nd.neighbors || nd.neighbors.length === 0) continue;
    for (const nb of nd.neighbors) {
      if (!nb.chassis) continue;
      switchSet.add(nb.chassis);
      edgeList.push({ hostId: nd.host, switchId: nb.chassis, iface: nb.interface, port: nb.port });
    }
  }

  const switches = Array.from(switchSet).sort();
  const clusterNodes = snap.nodes
    .filter((n) => (n.neighbors ?? []).length > 0)
    .map((n) => n.host)
    .sort();

  const SVG_W = 1100;
  const SWITCH_Y = 60;
  const NODE_AREA_TOP = 180;
  const NODE_AREA_H = 380;

  const switchSpacing = switches.length > 1 ? (SVG_W - 100) / (switches.length - 1) : 0;

  const graphNodes: GraphNode[] = [];

  switches.forEach((sw, i) => {
    graphNodes.push({
      id: `sw:${sw}`,
      label: sw,
      isSwitch: true,
      x: switches.length === 1 ? SVG_W / 2 : 50 + i * switchSpacing,
      y: SWITCH_Y,
    });
  });

  // Group cluster nodes by their primary switch (first neighbor)
  const nodesBySwitch: Map<string, string[]> = new Map();
  for (const sw of switches) nodesBySwitch.set(sw, []);
  const unassigned: string[] = [];

  for (const host of clusterNodes) {
    const nd = snap.nodes.find((n) => n.host === host);
    const primary = nd?.neighbors?.[0]?.chassis;
    if (primary && nodesBySwitch.has(primary)) {
      nodesBySwitch.get(primary)!.push(host);
    } else {
      unassigned.push(host);
    }
  }

  // Layout: group each switch's nodes in a column below the switch
  let col = 0;
  const totalCols = clusterNodes.length;
  const colSpacing = totalCols > 1 ? (SVG_W - 40) / Math.max(totalCols - 1, 1) : 0;

  const assignRows = (hosts: string[]) => {
    hosts.forEach((host) => {
      const x = totalCols <= 1 ? SVG_W / 2 : 20 + col * colSpacing;
      const y = NODE_AREA_TOP + (col % 2 === 0 ? 0 : 30); // slight stagger
      graphNodes.push({ id: `host:${host}`, label: host, isSwitch: false, x, y: Math.min(y, NODE_AREA_TOP + NODE_AREA_H - 20) });
      col++;
    });
  };

  for (const sw of switches) assignRows(nodesBySwitch.get(sw) ?? []);
  assignRows(unassigned);

  const edges: GraphEdge[] = edgeList.map((e) => ({
    fromId: `host:${e.hostId}`,
    toId: `sw:${e.switchId}`,
    iface: e.iface,
    port: e.port,
  }));

  return { nodes: graphNodes, edges };
}

interface TooltipState {
  x: number;
  y: number;
  content: string[];
}

function TopologyGraph({ snap }: { snap: TopologySnapshot }) {
  const { nodes, edges } = useMemo(() => buildGraph(snap), [snap]);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  if (nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-500 text-sm">
        No LLDP graph data — collect LLDP data first.
      </div>
    );
  }

  const nodeMap = new Map(nodes.map((n) => [n.id, n]));

  return (
    <div className="relative overflow-x-auto">
      <svg
        viewBox="0 0 1100 560"
        className="w-full min-w-[700px] border border-slate-700 rounded-lg bg-slate-900"
        onMouseLeave={() => setTooltip(null)}
      >
        {/* Legend */}
        <g transform="translate(10,10)">
          <rect width={12} height={12} rx={2} fill="#6366f1" />
          <text x={16} y={10} fill="#94a3b8" fontSize={10}>Switch / Neighbor</text>
          <rect x={120} width={12} height={12} rx={6} fill="#22c55e" />
          <text x={136} y={10} fill="#94a3b8" fontSize={10}>Cluster node</text>
        </g>

        {/* Edges */}
        {edges.map((e, i) => {
          const from = nodeMap.get(e.fromId);
          const to = nodeMap.get(e.toId);
          if (!from || !to) return null;
          return (
            <line
              key={i}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke="#334155"
              strokeWidth={1}
            />
          );
        })}

        {/* Nodes */}
        {nodes.map((n) => {
          if (n.isSwitch) {
            return (
              <g
                key={n.id}
                transform={`translate(${n.x},${n.y})`}
                style={{ cursor: "pointer" }}
                onMouseEnter={(e) => {
                  const svgRect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                  setTooltip({
                    x: e.clientX - svgRect.left + 8,
                    y: e.clientY - svgRect.top - 20,
                    content: ["Switch / Neighbor", n.label],
                  });
                }}
                onMouseLeave={() => setTooltip(null)}
              >
                <rect x={-36} y={-12} width={72} height={24} rx={4} fill="#4f46e5" opacity={0.9} />
                <text textAnchor="middle" y={4} fill="#fff" fontSize={9} fontWeight={600}>
                  {n.label.length > 14 ? n.label.slice(0, 12) + "…" : n.label}
                </text>
              </g>
            );
          }
          return (
            <g
              key={n.id}
              transform={`translate(${n.x},${n.y})`}
              style={{ cursor: "pointer" }}
              onMouseEnter={(e) => {
                const svgRect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                const nd = snap.nodes.find((nd) => nd.host === n.label);
                const lines = ["Cluster Node", n.label];
                nd?.neighbors?.forEach((nb) => lines.push(`↳ ${nb.interface} → ${nb.chassis}:${nb.port}`));
                setTooltip({ x: e.clientX - svgRect.left + 8, y: e.clientY - svgRect.top - 20, content: lines });
              }}
              onMouseLeave={() => setTooltip(null)}
            >
              <circle r={5} fill="#22c55e" />
            </g>
          );
        })}

        {/* Switch labels below the rect */}
        {nodes
          .filter((n) => n.isSwitch)
          .map((n) => (
            <text
              key={`lbl-${n.id}`}
              x={n.x}
              y={n.y + 22}
              textAnchor="middle"
              fill="#6366f1"
              fontSize={8}
              opacity={0.7}
            >
              {n.label.length > 18 ? n.label.slice(0, 16) + "…" : n.label}
            </text>
          ))}
      </svg>

      {tooltip && (
        <div
          className="absolute pointer-events-none z-10 bg-slate-800 border border-slate-600 rounded shadow-lg px-3 py-2 text-xs text-slate-200 whitespace-pre"
          style={{ left: tooltip.x, top: tooltip.y, maxWidth: 300 }}
        >
          {tooltip.content.map((line, i) => (
            <div key={i} className={i === 0 ? "font-semibold text-slate-400 text-[10px] uppercase mb-1" : ""}>
              {line}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Table ────────────────────────────────────────────────────────────────────

const columns: ColumnDef<FlatRow, unknown>[] = [
  { accessorKey: "host", header: "Node", size: 160 },
  { accessorKey: "interface", header: "Interface", size: 120 },
  { accessorKey: "chassis", header: "Neighbor", size: 200 },
  { accessorKey: "port", header: "Neighbor Port", size: 150 },
  {
    accessorKey: "description",
    header: "Description",
    size: 200,
    cell: (info) => (info.getValue() as string) || "—",
  },
  {
    accessorKey: "mgmt_ip",
    header: "Mgmt IP",
    size: 130,
    cell: (info) => (info.getValue() as string) || "—",
  },
];

// ─── Main Page ────────────────────────────────────────────────────────────────

type ViewMode = "table" | "graph";

export default function TopologyPage() {
  const [snap, setSnap] = useState<TopologySnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<ViewMode>("table");
  const [refreshing, setRefreshing] = useState(false);

  const load = async (refresh = false) => {
    if (refresh) setRefreshing(true);
    else setLoading(true);
    setError(null);
    try {
      const data = await getTopologyLLDP();
      setSnap(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  // Build flat rows + KPI counts
  const { rows, totalNeighbors, uniqueSwitches, nodesWithLLDP } = useMemo(() => {
    if (!snap) return { rows: [], totalNeighbors: 0, uniqueSwitches: 0, nodesWithLLDP: 0 };
    const r: FlatRow[] = [];
    const switches = new Set<string>();
    let withData = 0;
    for (const nd of snap.nodes) {
      if ((nd.neighbors ?? []).length > 0) withData++;
      for (const nb of nd.neighbors ?? []) {
        r.push({ host: nd.host, ...nb });
        if (nb.chassis) switches.add(nb.chassis);
      }
    }
    return { rows: r, totalNeighbors: r.length, uniqueSwitches: switches.size, nodesWithLLDP: withData };
  }, [snap]);

  const hasData = rows.length > 0;
  const collectedAt = snap?.collected_at ? new Date(snap.collected_at).toLocaleString() : "—";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-slate-100">Network Topology</h2>
          <p className="text-xs text-slate-400 mt-0.5">LLDP neighbor discovery · {collectedAt}</p>
        </div>
        <div className="flex items-center gap-2">
          {/* View toggle */}
          <div className="flex rounded-lg overflow-hidden border border-slate-700 text-sm">
            {(["table", "graph"] as ViewMode[]).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-3 py-1.5 capitalize transition-colors ${
                  view === v
                    ? "bg-indigo-600 text-white"
                    : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                }`}
              >
                {v === "table" ? "⊞ Table" : "⬡ Graph"}
              </button>
            ))}
          </div>
          <button
            onClick={() => load(true)}
            disabled={refreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg border border-slate-600 transition-colors disabled:opacity-50"
          >
            <svg className={`w-3.5 h-3.5 ${refreshing ? "animate-spin" : ""}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h5M20 20v-5h-5M4 9a9 9 0 0114.6-3.9M20 15a9 9 0 01-14.6 3.9" />
            </svg>
            {refreshing ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
        {[
          { label: "Nodes with LLDP", value: hasData ? nodesWithLLDP : "—", color: "text-green-400" },
          { label: "Total Neighbors", value: hasData ? totalNeighbors : "—", color: "text-blue-400" },
          { label: "Unique Switches", value: hasData ? uniqueSwitches : "—", color: "text-indigo-400" },
        ].map((kpi) => (
          <div key={kpi.label} className="bg-slate-800 rounded-xl p-4 border border-slate-700">
            <p className="text-xs text-slate-400 uppercase tracking-wide">{kpi.label}</p>
            <p className={`text-3xl font-bold mt-1 ${kpi.color}`}>{kpi.value}</p>
          </div>
        ))}
      </div>

      {/* Loading / Error / Empty */}
      {loading && (
        <div className="flex items-center justify-center h-48 text-slate-400 text-sm animate-pulse">
          Collecting LLDP data from nodes…
        </div>
      )}

      {error && !loading && (
        <div className="rounded-lg bg-red-900/30 border border-red-700 p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {!loading && !error && !hasData && (
        <div className="rounded-xl border border-border bg-card p-8 text-center space-y-3">
          <svg className="mx-auto w-10 h-10 text-muted-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
            <circle cx={12} cy={12} r={9} />
            <path strokeLinecap="round" d="M12 8v4M12 16h.01" />
          </svg>
          <p className="text-foreground font-medium">No LLDP neighbors found</p>
          <p className="text-muted-foreground text-sm max-w-md mx-auto">
            LLDP requires <code className="bg-muted px-1 rounded">lldpd</code> to be installed on your nodes.
            Install it via the <span className="text-primary font-semibold">Configuration → Package Installs</span> page, or run:
          </p>
          <code className="inline-block bg-muted border border-border rounded px-3 py-1 text-sm text-foreground">
            sudo apt-get install -y lldpd &amp;&amp; sudo systemctl enable --now lldpd
          </code>
        </div>
      )}

      {/* Content */}
      {!loading && !error && hasData && (
        <>
          {view === "table" && (
            <DataTable
              data={rows}
              columns={columns}
              defaultPageSize={100}
              hideZeroCols={false}
            />
          )}
          {view === "graph" && <TopologyGraph snap={snap!} />}
        </>
      )}
    </div>
  );
}
