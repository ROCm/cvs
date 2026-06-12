import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Loader2 } from "lucide-react";
import {
  getSuiteExamples,
  getSuiteSchema,
  type ConfigRef,
  type SuiteSchema,
} from "@/shared/api";
import { useInventory } from "@/shared/inventory";
import DynamicForm from "./DynamicForm";
import ConfigPicker from "./ConfigPicker";
import RunPanel from "./RunPanel";

// detectedGpu picks the first GPU type reported by the inventory probe, used to
// pre-select the platform facet in the config picker.
function useDetectedGpu(): string | undefined {
  const { status } = useInventory();
  return useMemo(() => {
    const statuses = status?.inventory?.statuses ?? [];
    return statuses.find((s) => s.gpu_type)?.gpu_type;
  }, [status]);
}

export default function SuiteDetail() {
  const { suite = "" } = useParams();
  const detectedGpu = useDetectedGpu();

  const [examples, setExamples] = useState<ConfigRef[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [data, setData] = useState<SuiteSchema | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<unknown>(null);
  const [remaining, setRemaining] = useState(0);

  // Load the suite's example configs (with facets) once per suite.
  useEffect(() => {
    setExamples(null);
    setSelected(null);
    setData(null);
    setError(null);
    getSuiteExamples(suite)
      .then(setExamples)
      .catch((e) => setError(String(e)));
  }, [suite]);

  // Load the schema for the selected config. When a suite has no facets/no
  // example, `selected` stays null and we load the default schema directly.
  useEffect(() => {
    if (examples === null) return;
    setData(null);
    setError(null);
    const noPick = examples.length === 0;
    if (!noPick && selected === null) return; // wait for the picker to resolve
    getSuiteSchema(suite, selected ?? undefined)
      .then(setData)
      .catch((e) => setError(String(e)));
  }, [suite, examples, selected]);

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

      {examples && examples.length > 0 && (
        <ConfigPicker
          examples={examples}
          selected={selected}
          onSelect={setSelected}
          detectedGpu={detectedGpu}
        />
      )}

      {!error && data === null && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading config schema...
        </div>
      )}

      {data && !data.has_config && (
        <>
          <div className="rounded-lg border border-dashed border-border bg-card p-8 text-center text-muted-foreground">
            This suite has no config file — it runs without a <code>--config_file</code>.
          </div>
          <RunPanel suite={suite} config={null} remaining={0} />
        </>
      )}

      {data && data.has_config && (
        <>
          <p className="mb-6 text-sm text-muted-foreground">
            Form generated from{" "}
            <code className="rounded bg-muted px-1 py-0.5">{data.config_path}</code>{" "}
            <span className="uppercase text-muted-foreground">({data.format})</span>
          </p>
          <DynamicForm
            schema={data.schema}
            example={data.example as never}
            onChange={(c, r) => {
              setConfig(c);
              setRemaining(r);
            }}
          />
          <RunPanel suite={suite} config={config} remaining={remaining} />
        </>
      )}
    </div>
  );
}
