import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Loader2 } from "lucide-react";
import { getSuiteSchema, type SuiteSchema } from "@/shared/api";
import DynamicForm from "./DynamicForm";

export default function SuiteDetail() {
  const { suite = "" } = useParams();
  const [data, setData] = useState<SuiteSchema | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    setError(null);
    getSuiteSchema(suite)
      .then(setData)
      .catch((e) => setError(String(e)));
  }, [suite]);

  return (
    <div className="mx-auto max-w-5xl">
      <Link to="/cvs" className="mb-4 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-primary">
        <ArrowLeft className="h-4 w-4" /> All suites
      </Link>

      <h1 className="mb-1 font-mono text-2xl font-semibold">{suite}</h1>

      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
          Failed to load schema: {error}
        </div>
      )}

      {!error && data === null && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading config schema...
        </div>
      )}

      {data && !data.has_config && (
        <div className="rounded-lg border border-dashed border-border bg-card p-8 text-center text-muted-foreground">
          This suite has no config file — it runs without a <code>--config_file</code>.
        </div>
      )}

      {data && data.has_config && (
        <>
          <p className="mb-6 text-sm text-muted-foreground">
            Form generated from{" "}
            <code className="rounded bg-muted px-1 py-0.5">{data.config_path}</code>{" "}
            <span className="uppercase text-muted-foreground">({data.format})</span>
          </p>
          <DynamicForm schema={data.schema} example={data.example as never} />
          <div className="mt-6 border-t border-border pt-4">
            <button
              disabled
              title="Execution lands in S3"
              className="cursor-not-allowed rounded-lg bg-muted px-4 py-2 text-sm font-medium text-muted-foreground"
            >
              Run suite (coming in S3)
            </button>
          </div>
        </>
      )}
    </div>
  );
}
