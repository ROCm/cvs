import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ExternalLink, FileText, Loader2, RadioTower, RotateCw } from "lucide-react";
import {
  executionReportUrl,
  isTerminal,
  listExecutions,
  rerunExecution,
  type Execution,
} from "@/shared/api";

const STATUS_STYLES: Record<string, string> = {
  queued: "bg-muted text-muted-foreground",
  running: "bg-primary/10 text-primary",
  passed: "bg-green-500/10 text-green-600",
  failed: "bg-destructive/10 text-destructive",
  error: "bg-destructive/10 text-destructive",
  interrupted: "bg-amber-500/10 text-amber-600",
};

function fmt(ts?: string): string {
  if (!ts) return "—";
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString();
}

export default function ExecutionHistory() {
  const navigate = useNavigate();
  const [executions, setExecutions] = useState<Execution[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rerunning, setRerunning] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const es = await listExecutions();
        if (!cancelled) setExecutions(es);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    };
    load();
    // Refresh so running rows advance to terminal + reports appear.
    const t = window.setInterval(load, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, []);

  const handleRerun = async (id: string) => {
    setRerunning(id);
    setError(null);
    try {
      const ex = await rerunExecution(id);
      navigate(`/cvs/executions/${encodeURIComponent(ex.id)}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setRerunning(null);
    }
  };

  return (
    <div className="mx-auto max-w-5xl">
      <div className="mb-1 flex items-center gap-2">
        <h1 className="text-2xl font-semibold">Execution History</h1>
        <Link to="/cvs" className="ml-auto text-sm text-primary hover:underline">
          ← Back to suites
        </Link>
      </div>
      <p className="mb-6 text-muted-foreground">Past and in-progress test runs.</p>

      {error && (
        <div className="mb-4 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {executions === null && !error && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading…
        </div>
      )}

      {executions && executions.length === 0 && (
        <p className="text-muted-foreground">No executions yet.</p>
      )}

      {executions && executions.length > 0 && (
        <div className="overflow-hidden rounded-xl border border-border">
          <table className="w-full text-sm">
            <thead className="bg-muted/50 text-left text-xs uppercase text-muted-foreground">
              <tr>
                <th className="px-4 py-2 font-medium">Execution</th>
                <th className="px-4 py-2 font-medium">Suite</th>
                <th className="px-4 py-2 font-medium">Status</th>
                <th className="px-4 py-2 font-medium">Created</th>
                <th className="px-4 py-2 font-medium">Report / Logs</th>
                <th className="px-4 py-2 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {executions.map((e) => (
                <tr key={e.id} className="hover:bg-muted/30">
                  <td className="px-4 py-2">
                    <Link
                      to={`/cvs/executions/${encodeURIComponent(e.id)}`}
                      className="font-mono text-xs text-primary hover:underline"
                    >
                      {e.id}
                    </Link>
                  </td>
                  <td className="px-4 py-2 font-mono text-xs">{e.suite}</td>
                  <td className="px-4 py-2">
                    <span
                      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${
                        STATUS_STYLES[e.status] ?? "bg-muted"
                      }`}
                    >
                      {e.status}
                      {!isTerminal(e.status) && (
                        <Loader2 className="ml-1 h-3 w-3 animate-spin" />
                      )}
                    </span>
                  </td>
                  <td className="px-4 py-2 text-xs text-muted-foreground">{fmt(e.created_at)}</td>
                  <td className="px-4 py-2">
                    <div className="flex items-center gap-3">
                      {e.has_report ? (
                        <a
                          href={executionReportUrl(e.id)}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 text-primary hover:underline"
                        >
                          <FileText className="h-3.5 w-3.5" /> Report
                          <ExternalLink className="h-3 w-3" />
                        </a>
                      ) : (
                        !isTerminal(e.status) && (
                          <Link
                            to={`/cvs/executions/${encodeURIComponent(e.id)}`}
                            className="inline-flex items-center gap-1 text-primary hover:underline"
                          >
                            <RadioTower className="h-3.5 w-3.5" /> Live logs
                          </Link>
                        )
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-2">
                    <button
                      type="button"
                      onClick={() => handleRerun(e.id)}
                      disabled={rerunning !== null}
                      title="Run again with the same suite, cluster, and config"
                      className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1 text-xs font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {rerunning === e.id ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <RotateCw className="h-3.5 w-3.5" />
                      )}
                      Rerun
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
