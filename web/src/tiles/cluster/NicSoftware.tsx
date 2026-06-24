import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import { type ColumnDef } from "@tanstack/react-table";
import { getNicDevlink, type DevlinkSnapshot } from "@/shared/api";
import { DataTable } from "@/shared/DataTable";
import { fmt } from "./format";
import { useCluster } from "./ClusterContext";

// ── Row types ──────────────────────────────────────────────────────────────

interface DeviceRow {
  node:        string;
  pci:         string;
  vendor:      string;
  driver:      string;
  fw_version:  string;
  serial:      string;
  board_id:    string;
  asic_id:     string;
  asic_rev:    string;
  fw_psid:     string;
  fw_mgmt:     string;
  fw_mgmt_api: string;
  fw_cpld:     string;
}

// ── Column definitions ─────────────────────────────────────────────────────

const deviceColumns: ColumnDef<DeviceRow, unknown>[] = [
  { accessorKey: "node",        header: "Node" },
  { accessorKey: "pci",         header: "PCI" },
  { accessorKey: "vendor",      header: "Vendor" },
  { accessorKey: "driver",      header: "Driver" },
  { accessorKey: "fw_version",  header: "FW" },
  { accessorKey: "serial",      header: "Serial" },
  { accessorKey: "board_id",    header: "Board ID" },
  { accessorKey: "asic_id",     header: "ASIC ID" },
  { accessorKey: "asic_rev",    header: "ASIC Rev" },
  { accessorKey: "fw_psid",     header: "FW PSID" },
  { accessorKey: "fw_mgmt",     header: "FW Mgmt" },
  { accessorKey: "fw_mgmt_api", header: "FW Mgmt API" },
  { accessorKey: "fw_cpld",     header: "FW CPLD" },
];

function SectionTitle({ title }: { title: string }) {
  return <h3 className="mb-2 text-sm font-semibold text-foreground">{title}</h3>;
}

// ── Main component ─────────────────────────────────────────────────────────

export default function NicSoftware() {
  const { live } = useCluster();
  const [snap, setSnap] = useState<DevlinkSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      setSnap(await getNicDevlink());
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

  // Flatten nodes × devices → flat rows
  const deviceRows = useMemo<DeviceRow[]>(() =>
    (snap?.nodes ?? []).flatMap((n) =>
      (n.devices ?? []).map((d) => ({
        node:        n.host,
        pci:         d.pci_address,
        vendor:      d.vendor,
        driver:      d.driver,
        fw_version:  d.fw_version,
        serial:      d.serial_number,
        board_id:    d.board_id,
        asic_id:     d.asic_id,
        asic_rev:    d.asic_rev,
        fw_psid:     d.fw_psid,
        fw_mgmt:     d.fw_mgmt,
        fw_mgmt_api: d.fw_mgmt_api,
        fw_cpld:     d.fw_cpld,
      })),
    ), [snap]);

  // Summary aggregates
  const totalNodes   = snap?.nodes.length ?? 0;
  const totalDevices = deviceRows.length;
  const errNodes     = (snap?.nodes ?? []).filter((n) => n.error).length;
  const { fwMode, fwVersionCount } = useMemo(() => {
    const counts: Record<string, number> = {};
    deviceRows.forEach((r) => { if (r.fw_version) counts[r.fw_version] = (counts[r.fw_version] ?? 0) + 1; });
    const entries = Object.entries(counts);
    if (entries.length === 0) return { fwMode: "—", fwVersionCount: 0 };
    return {
      fwMode: entries.sort((a, b) => b[1] - a[1])[0][0],
      fwVersionCount: entries.length,
    };
  }, [deviceRows]);

  if (!live && !snap) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center space-y-2">
        <p className="text-foreground font-medium">No NIC software data</p>
        <p className="text-foreground text-sm">Click <span className="text-primary font-semibold">Start</span> to begin monitoring.</p>
      </div>
    );
  }

  if (live && !snap && !error) {
    return (
      <div className="rounded-xl border border-border bg-card p-8 text-center space-y-2">
        <p className="text-foreground font-medium flex items-center justify-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          Collecting NIC devlink info from nodes… this may take up to 30 s.
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
              {totalNodes} nodes · {totalDevices} devices
              {" · "}FW {fwMode}
              {fwVersionCount > 1 && (
                <span className="text-amber-500"> ({fwVersionCount} versions)</span>
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

      {/* Devices table */}
      {deviceRows.length > 0 && (
        <div>
          <SectionTitle title="NIC Devices" />
          <DataTable
            data={deviceRows}
            columns={deviceColumns}
            searchPlaceholder="Filter by node, PCI, vendor, FW…"
          />
        </div>
      )}

      {deviceRows.length === 0 && snap && !error && (
        <p className="text-sm text-muted-foreground">No devlink device data returned.</p>
      )}
    </div>
  );
}
