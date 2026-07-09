import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { FlaskConical, Loader2, ChevronRight, History } from "lucide-react";
import { listSuites, type Suite } from "@/shared/api";

interface CategoryGroup {
  category: string;
  module: string;
  suites: Suite[];
}

function groupByCategory(suites: Suite[]): CategoryGroup[] {
  const byModule = new Map<string, CategoryGroup>();
  for (const s of suites) {
    const key = s.module || s.category;
    let g = byModule.get(key);
    if (!g) {
      g = { category: s.category, module: s.module, suites: [] };
      byModule.set(key, g);
    }
    g.suites.push(s);
  }
  return [...byModule.values()].sort((a, b) => a.module.localeCompare(b.module));
}

export default function SuiteList() {
  const [suites, setSuites] = useState<Suite[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listSuites()
      .then(setSuites)
      .catch((e) => setError(String(e)));
  }, []);

  const groups = useMemo(() => (suites ? groupByCategory(suites) : []), [suites]);

  return (
    <div className="mx-auto max-w-4xl">
      <div className="mb-4 flex items-start justify-between">
        <div className="inline-flex rounded-lg bg-primary/10 p-3 text-primary">
          <FlaskConical className="h-6 w-6" />
        </div>
        <Link
          to="/cvs/history"
          className="inline-flex items-center gap-1 rounded-lg border border-border px-3 py-1.5 text-sm text-foreground hover:border-primary hover:text-primary"
        >
          <History className="h-4 w-4" /> Execution history
        </Link>
      </div>
      <h1 className="mb-1 text-2xl font-semibold">Test Execution</h1>
      <p className="mb-6 text-muted-foreground">
        Test suites discovered from the server via{" "}
        <code className="rounded bg-muted px-1 py-0.5 text-sm">cvs list</code>.
        {suites && <span className="ml-1 text-muted-foreground">({suites.length} suites)</span>}
      </p>

      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
          Failed to load suites: {error}
        </div>
      )}

      {!error && suites === null && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading suites...
        </div>
      )}

      {groups.length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {groups.map((g) => (
            <div key={g.module} className="rounded-xl border border-border bg-card p-5 shadow-sm">
              <div className="mb-2 flex items-center justify-between">
                <h2 className="text-lg font-semibold capitalize">{g.category}</h2>
                <span className="rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
                  {g.suites.length} suite{g.suites.length === 1 ? "" : "s"}
                </span>
              </div>
              <p className="mb-3 font-mono text-xs text-muted-foreground">{g.module}</p>
              <ul className="space-y-1">
                {g.suites.map((s) => (
                  <li key={s.module_path}>
                    <Link
                      to={`/cvs/${encodeURIComponent(s.name)}`}
                      className="group flex items-center justify-between rounded px-2 py-1 text-sm text-foreground hover:bg-primary/10 hover:text-primary"
                    >
                      <span className="font-mono">{s.name}</span>
                      <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-primary" />
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
