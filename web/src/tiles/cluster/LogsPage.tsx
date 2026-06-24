import { useCallback, useEffect, useState } from "react";
import { Loader2, RefreshCw, Search } from "lucide-react";
import {
  getDmesgLogs,
  searchLogs,
  type LogsSnapshot,
  type NodeLogs,
  type SearchResponse,
} from "@/shared/api";
import { fmt } from "./format";
import { useCluster } from "./ClusterContext";

function LogBucket({ title, body }: { title: string; body: string }) {
  const clean = body.trim() === "";
  return (
    <details className="mt-2 rounded-lg border border-border/60">
      <summary className="flex cursor-pointer items-center justify-between px-3 py-1.5 text-xs font-medium">
        <span>{title}</span>
        <span className={clean ? "text-green-600" : "text-amber-600"}>
          {clean ? "clean" : `${body.split("\n").length} lines`}
        </span>
      </summary>
      {!clean && (
        <pre className="max-h-64 overflow-auto border-t border-border/60 bg-muted/40 px-3 py-2 text-[11px] leading-relaxed">
          {body}
        </pre>
      )}
    </details>
  );
}

function NodeLogsCard({ node }: { node: NodeLogs }) {
  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <span className="font-mono text-sm font-medium">{node.host}</span>
      {node.error ? (
        <p className="mt-2 text-xs text-destructive">{node.error}</p>
      ) : (
        <>
          <LogBucket title="AMD hardware / driver" body={node.amd_logs} />
          <LogBucket title="System errors (dmesg)" body={node.dmesg_errors} />
          <LogBucket title="Userspace / ML errors" body={node.userspace_errors} />
        </>
      )}
    </div>
  );
}

function SearchPanel() {
  const [grep, setGrep] = useState("grep -i error");
  const [res, setRes] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    try {
      setRes(await searchLogs(grep));
      setError(null);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
      setRes(null);
    } finally {
      setLoading(false);
    }
  }, [grep]);

  return (
    <div className="mb-6 rounded-xl border border-border bg-card p-4">
      <h3 className="mb-1 text-sm font-semibold">Search dmesg</h3>
      <p className="mb-3 text-xs text-muted-foreground">
        Pipe of <code className="font-mono">grep</code>/<code className="font-mono">egrep</code>{" "}
        segments only (validated server-side); first 5 matches per node.
      </p>
      <div className="flex gap-2">
        <input
          value={grep}
          onChange={(e) => setGrep(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void run();
          }}
          placeholder="grep -i 'xgmi' | grep -v warn"
          className="flex-1 rounded-lg border border-border bg-background px-3 py-1.5 font-mono text-sm focus:border-primary focus:outline-none"
        />
        <button
          type="button"
          onClick={() => void run()}
          disabled={loading}
          className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
          Search
        </button>
      </div>

      {error && <p className="mt-3 text-xs text-destructive">{error}</p>}

      {res && (
        <div className="mt-3">
          <p className="mb-2 text-xs text-muted-foreground">
            {res.nodes_with_results}/{res.total_nodes_searched} nodes matched
          </p>
          <div className="space-y-2">
            {res.results
              .filter((r) => r.output.trim() !== "")
              .map((r) => (
                <div key={r.host} className="rounded-lg border border-border/60">
                  <div className="px-3 py-1.5 font-mono text-xs font-medium">{r.host}</div>
                  <pre className="max-h-48 overflow-auto border-t border-border/60 bg-muted/40 px-3 py-2 text-[11px]">
                    {r.output}
                  </pre>
                </div>
              ))}
            {res.nodes_with_results === 0 && (
              <p className="text-xs text-muted-foreground">No matches on any node.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default function LogsPage() {
  const { live } = useCluster();
  const [snap, setSnap] = useState<LogsSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const data = await getDmesgLogs();
      if (data !== null) setSnap(data);
      setError(null);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  // Fetch on mount only when monitoring is active; re-fetch when it starts.
  useEffect(() => {
    if (!live) return;
    void load();
  }, [live, load]);

  // Poll every 10s while waiting: snap is null (server returned 204, first
  // sweep still running) or snap.collecting (background refresh). Only while live.
  useEffect(() => {
    if (!live) return;
    if (snap !== null && !snap.collecting) return;
    const id = setInterval(() => void load(true), 10_000);
    return () => clearInterval(id);
  }, [live, snap, load]);

  return (
    <div>
      <SearchPanel />

      <div className="mb-4 flex items-center gap-3">
        <div>
          <p className="text-sm text-muted-foreground">
            AMD hardware, system, and userspace/ML error logs from{" "}
            <code className="font-mono text-xs">dmesg</code> across the fleet.
          </p>
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
        </div>
        <button
          type="button"
          onClick={() => void load()}
          disabled={loading}
          className="ml-auto inline-flex items-center gap-1.5 rounded-lg border border-border px-3 py-1.5 text-sm hover:border-primary hover:text-primary disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="mb-4 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {!live && !snap && (
        <div className="rounded-xl border border-border bg-card p-8 text-center space-y-2">
          <p className="text-foreground font-medium">No log data</p>
          <p className="text-foreground text-sm">
            Click <span className="text-primary font-semibold">Start</span> to begin monitoring and collect fleet logs.
          </p>
        </div>
      )}

      {live && snap === null && !error && (
        <div className="rounded-xl border border-border bg-card p-8 text-center space-y-2">
          <p className="text-foreground font-medium flex items-center justify-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            Collecting logs across the fleet… this may take up to 90 s.
          </p>
        </div>
      )}

      {snap && snap.nodes.length > 0 && (() => {
        const withIssues = snap.nodes.filter(
          (n) => n.error || n.amd_logs.trim() || n.dmesg_errors.trim() || n.userspace_errors.trim(),
        );
        const cleanCount = snap.nodes.length - withIssues.length;
        return (
          <>
            {cleanCount > 0 && (
              <p className="mb-3 text-sm text-green-600">
                {cleanCount} of {snap.nodes.length} nodes are clean (no matching log entries).
              </p>
            )}
            {withIssues.length === 0 ? (
              <p className="text-sm text-green-600">All {snap.nodes.length} nodes are clean.</p>
            ) : (
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                {withIssues.map((n) => (
                  <NodeLogsCard key={n.host} node={n} />
                ))}
              </div>
            )}
          </>
        );
      })()}
    </div>
  );
}
