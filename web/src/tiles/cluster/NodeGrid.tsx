import { Cpu, Loader2, Wifi, WifiOff } from "lucide-react";
import type { ClusterNode } from "@/shared/api";
import { useCluster } from "./ClusterContext";
import { fmt } from "./format";

function StatusBadge({ node }: { node: ClusterNode }) {
  if (!node.probed) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
        Not probed
      </span>
    );
  }
  if (node.reachable) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-green-500/10 px-2 py-0.5 text-xs font-medium text-green-600">
        <Wifi className="h-3 w-3" /> Reachable
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-destructive/10 px-2 py-0.5 text-xs font-medium text-destructive">
      <WifiOff className="h-3 w-3" /> Unreachable
    </span>
  );
}

function NodeCard({ node }: { node: ClusterNode }) {
  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <span className="font-mono text-sm font-medium">{node.host}</span>
        <StatusBadge node={node} />
      </div>

      {node.probed && node.reachable ? (
        <dl className="space-y-1.5 text-sm">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Cpu className="h-4 w-4" />
            <span className="text-foreground">
              {node.gpu_type || "Unknown GPU"}
              {node.gpu_count ? ` ×${node.gpu_count}` : ""}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <dt className="text-muted-foreground">ROCm</dt>
            <dd className="font-mono text-xs">{node.rocm_version || "—"}</dd>
          </div>
          <div className="flex items-center justify-between">
            <dt className="text-muted-foreground">Checked</dt>
            <dd className="text-xs text-muted-foreground">{fmt(node.checked_at)}</dd>
          </div>
        </dl>
      ) : node.probed ? (
        <p className="text-sm text-destructive">{node.error || "Probe failed"}</p>
      ) : (
        <p className="text-sm text-muted-foreground">
          Run a connectivity probe from the Inventory page to populate GPU/ROCm details.
        </p>
      )}
    </div>
  );
}

export default function NodeGrid() {
  const { nodes, nodesError } = useCluster();

  return (
    <div>
      <p className="mb-4 text-sm text-muted-foreground">
        Fleet nodes from the shared inventory, with their latest connectivity and basic-info probe
        results.
      </p>

      {nodesError && (
        <div className="mb-4 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          {nodesError}
        </div>
      )}

      {nodes === null && !nodesError && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading…
        </div>
      )}

      {nodes && nodes.nodes.length === 0 && (
        <p className="text-muted-foreground">No nodes in the inventory.</p>
      )}

      {nodes && nodes.nodes.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {nodes.nodes.map((n) => (
            <NodeCard key={n.host} node={n} />
          ))}
        </div>
      )}
    </div>
  );
}
