import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronRight, ExternalLink, FileText, Loader2, ScrollText } from "lucide-react";
import {
  executionLogUrl,
  executionReportUrl,
  getExecution,
  getExecutionClusterFile,
  getExecutionConfig,
  isTerminal,
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

// ExpandableFile lazily fetches a text file the first time it is opened and
// renders it in a scrollable <pre>.
function ExpandableFile({
  label,
  load,
}: {
  label: string;
  load: () => Promise<string | null>;
}) {
  const [open, setOpen] = useState(false);
  const [content, setContent] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggle = async () => {
    const next = !open;
    setOpen(next);
    if (next && !loaded) {
      try {
        setContent(await load());
      } catch (e) {
        setError(String(e));
      } finally {
        setLoaded(true);
      }
    }
  };

  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <button
        type="button"
        onClick={toggle}
        className="flex w-full items-center gap-2 bg-muted/40 px-3 py-2 text-left text-sm font-medium hover:bg-muted"
      >
        <ChevronRight
          className={`h-4 w-4 transition-transform ${open ? "rotate-90" : ""}`}
        />
        {label}
      </button>
      {open && (
        <div className="border-t border-border">
          {!loaded ? (
            <div className="flex items-center gap-2 p-3 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading…
            </div>
          ) : error ? (
            <p className="p-3 text-sm text-destructive">{error}</p>
          ) : content == null ? (
            <p className="p-3 text-sm text-muted-foreground">(none)</p>
          ) : (
            <pre className="max-h-[50vh] overflow-auto bg-muted p-3 font-mono text-xs">
              {content}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export default function ExecutionDetail() {
  const { id = "" } = useParams();
  const [execution, setExecution] = useState<Execution | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    const load = async () => {
      try {
        const ex = await getExecution(id);
        if (cancelled) return;
        setExecution(ex);
        // Stop polling once terminal (status/has_report are final).
        if (isTerminal(ex.status)) window.clearInterval(t);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    };
    const t = window.setInterval(load, 3000);
    load();
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [id]);

  return (
    <div className="mx-auto max-w-5xl">
      <div className="mb-1 flex items-center gap-2">
        <h1 className="text-2xl font-semibold">Execution</h1>
        <span className="font-mono text-sm text-muted-foreground">{id}</span>
        <Link to="/cvs/history" className="ml-auto text-sm text-primary hover:underline">
          ← Back to history
        </Link>
      </div>

      {error && (
        <div className="mb-4 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {!execution && !error && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading…
        </div>
      )}

      {execution && (
        <>
          <div className="mb-4 flex flex-wrap items-center gap-3">
            <span
              className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${
                STATUS_STYLES[execution.status] ?? "bg-muted"
              }`}
            >
              {execution.status}
              {!isTerminal(execution.status) && (
                <Loader2 className="ml-1 h-3 w-3 animate-spin" />
              )}
            </span>
            {execution.suite && (
              <span className="font-mono text-sm text-muted-foreground">{execution.suite}</span>
            )}
            {execution.exit_code != null && (
              <span className="text-xs text-muted-foreground">exit {execution.exit_code}</span>
            )}
          </div>

          {execution.error && (
            <p className="mb-4 text-sm text-destructive">{execution.error}</p>
          )}

          <dl className="mb-6 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
            <div>
              <dt className="text-xs uppercase text-muted-foreground">Cluster</dt>
              <dd className="font-mono text-xs">{execution.cluster_id || "—"}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase text-muted-foreground">Created</dt>
              <dd className="text-xs">{fmt(execution.created_at)}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase text-muted-foreground">Started</dt>
              <dd className="text-xs">{fmt(execution.started_at)}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase text-muted-foreground">Finished</dt>
              <dd className="text-xs">{fmt(execution.finished_at)}</dd>
            </div>
          </dl>

          <div className="mb-6 space-y-2">
            <ExpandableFile
              label="Cluster file"
              load={() => getExecutionClusterFile(execution.id)}
            />
            <ExpandableFile
              label="Config file"
              load={() => getExecutionConfig(execution.id)}
            />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {execution.has_report ? (
              <a
                href={executionReportUrl(execution.id)}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2 rounded-lg border border-border px-3 py-1.5 text-sm hover:border-primary hover:text-primary"
              >
                <FileText className="h-4 w-4" /> HTML report
                <ExternalLink className="h-3 w-3" />
              </a>
            ) : (
              <span className="text-sm text-muted-foreground">
                {isTerminal(execution.status) ? "No report" : "Report pending…"}
              </span>
            )}
            <a
              href={executionLogUrl(execution.id)}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-lg border border-border px-3 py-1.5 text-sm hover:border-primary hover:text-primary"
            >
              <ScrollText className="h-4 w-4" /> Run log
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </>
      )}
    </div>
  );
}
