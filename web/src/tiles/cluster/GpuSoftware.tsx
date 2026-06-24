import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import { type ColumnDef } from "@tanstack/react-table";
import { getGpuSoftware, type GPUSoftwareSnapshotOrPending } from "@/shared/api";
import { DataTable } from "@/shared/DataTable";
import { fmt } from "./format";
import { useCluster } from "./ClusterContext";

// ── Row types ──────────────────────────────────────────────────────────────

interface VersionRow {
  node: string;
  rocm: string;
  amdgpu: string;
  amdsmi_tool: string;
  amdsmi_lib: string;
  hsmp: string;
  error?: string;
}

interface FirmwareRow {
  node: string;
  gpu: number;
  component: string;
  version: string;
}

// ── Column definitions ─────────────────────────────────────────────────────

const versionColumns: ColumnDef<VersionRow, unknown>[] = [
  { accessorKey: "node",       header: "Node" },
  { accessorKey: "rocm",       header: "ROCm" },
  { accessorKey: "amdgpu",     header: "amdgpu driver" },
  { accessorKey: "amdsmi_tool", header: "AMD SMI tool" },
  { accessorKey: "amdsmi_lib", header: "AMD SMI lib" },
  { accessorKey: "hsmp",       header: "HSMP" },
];

const firmwareColumns: ColumnDef<FirmwareRow, unknown>[] = [
  { accessorKey: "node",      header: "Node" },
  { accessorKey: "gpu",       header: "GPU" },
  { accessorKey: "component", header: "Component" },
  { accessorKey: "version",   header: "Version" },
];

function SectionTitle({ title }: { title: string }) {
  return <h3 className="mb-2 text-sm font-semibold text-foreground">{title}</h3>;
}

// ── Main component ─────────────────────────────────────────────────────────

export default function GpuSoftware() {
  const { live } = useCluster();
  const [snap, setSnap] = useState<GPUSoftwareSnapshotOrPending | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      setSnap(await getGpuSoftware());
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!live) return;
    void load();
  }, [live, load]);

  useEffect(() => {
    if (!live || !snap?.collecting) return;
    const id = setInterval(() => void load(true), 10_000);
    return () => clearInterval(id);
  }, [live, snap?.collecting, load]);

  // Flatten nodes → version rows
  const versionRows = useMemo<VersionRow[]>(() =>
    (snap?.nodes ?? []).map((n) => ({
      node: n.host,
      rocm:       n.version?.rocm_version    ?? (n.error ? "error" : "—"),
      amdgpu:     n.version?.amdgpu_version  ?? "—",
      amdsmi_tool: n.version?.amdsmi_tool    ?? "—",
      amdsmi_lib: n.version?.amdsmi_library  ?? "—",
      hsmp:       n.version?.amd_hsmp_version ?? "—",
      error:      n.error,
    })), [snap]);

  // Flatten nodes → per-GPU firmware rows
  const firmwareRows = useMemo<FirmwareRow[]>(() =>
    (snap?.nodes ?? []).flatMap((n) =>
      (n.firmware ?? []).flatMap((g) =>
        (g.fw_list ?? []).map((fw) => ({
          node:      n.host,
          gpu:       g.gpu,
          component: fw.fw_id,
          version:   fw.fw_version,
        })),
      ),
    ), [snap]);

  // Summary aggregates
  const totalNodes = versionRows.length;
  const errNodes   = versionRows.filter((r) => r.error).length;
  const { rocmMode, rocmVersionCount } = useMemo(() => {
    const counts: Record<string, number> = {};
    versionRows.forEach((r) => { if (r.rocm !== "—" && r.rocm !== "error") counts[r.rocm] = (counts[r.rocm] ?? 0) + 1; });
    const entries = Object.entries(counts);
    if (entries.length === 0) return { rocmMode: "—", rocmVersionCount: 0 };
    return {
      rocmMode: entries.sort((a, b) => b[1] - a[1])[0][0],
      rocmVersionCount: entries.length,
    };
  }, [versionRows]);

  if (!live && !snap) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center space-y-2">
        <p className="text-foreground font-medium">No GPU software data</p>
        <p className="text-foreground text-sm">Click <span className="text-primary font-semibold">Start</span> to begin monitoring.</p>
      </div>
    );
  }

  if (live && !snap && !error) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center space-y-2">
        <p className="text-foreground font-medium flex items-center justify-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          Collecting GPU software from nodes… this may take up to 30 s.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header bar */}
      <div className="flex items-center justify-between">
        <div className="space-y-0.5">
          {snap?.collected_at && (
            <p className="text-xs text-muted-foreground">
              Collected {fmt(snap.collected_at)}.
              {snap.collecting && (
                <span className="ml-2 inline-flex items-center gap-1 text-amber-500">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  refreshing…
                </span>
              )}
            </p>
          )}
          {snap && (
            <p className="text-xs text-muted-foreground">
              {totalNodes} nodes
              {" · "}ROCm {rocmMode}
              {rocmVersionCount > 1 && (
                <span className="text-amber-500"> ({rocmVersionCount} versions)</span>
              )}
              {errNodes > 0 && (
                <span className="text-destructive"> · {errNodes} {errNodes === 1 ? "error" : "errors"}</span>
              )}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={() => void load()}
          disabled={loading}
          className="inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm hover:border-primary hover:text-primary disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Software Versions table */}
      {versionRows.length > 0 && (
        <div>
          <SectionTitle title="Software Versions" />
          <DataTable
            data={versionRows}
            columns={versionColumns}
            searchPlaceholder="Filter by node or version…"
          />
        </div>
      )}

      {/* Firmware table */}
      {firmwareRows.length > 0 && (
        <div>
          <SectionTitle title="GPU Firmware" />
          <DataTable
            data={firmwareRows}
            columns={firmwareColumns}
            searchPlaceholder="Filter by node, GPU, or component…"
          />
        </div>
      )}
    </div>
  );
}
