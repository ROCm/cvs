import { useState } from "react";
import type { Field } from "@/shared/api";

// JSON-ish value type for config trees.
type Json = string | number | boolean | null | Json[] | { [k: string]: Json };

function getAt(root: Json, path: (string | number)[]): Json {
  let cur: Json = root;
  for (const key of path) {
    if (cur == null || typeof cur !== "object") return null;
    cur = (cur as Record<string | number, Json>)[key as never] as Json;
  }
  return cur;
}

function setAt(root: Json, path: (string | number)[], value: Json): Json {
  if (path.length === 0) return value;
  const [head, ...rest] = path;
  const clone: Json = Array.isArray(root)
    ? [...(root as Json[])]
    : { ...(root as Record<string, Json>) };
  (clone as Record<string | number, Json>)[head as never] = setAt(
    getAt(root, [head]),
    rest,
    value,
  );
  return clone;
}

interface FieldProps {
  field: Field;
  path: (string | number)[];
  value: Json;
  onChange: (path: (string | number)[], value: Json) => void;
}

function FieldRow({ field, path, value, onChange }: FieldProps) {
  if (field.type === "object") {
    return (
      <fieldset className="rounded-lg border border-border p-3">
        <legend className="px-1 font-mono text-sm font-medium text-foreground">{field.name}</legend>
        {field.description && <p className="mb-2 text-xs text-muted-foreground">{field.description}</p>}
        <div className="space-y-3">
          {(field.fields ?? []).map((child) => (
            <FieldRow
              key={child.name}
              field={child}
              path={[...path, child.name]}
              value={value}
              onChange={onChange}
            />
          ))}
        </div>
      </fieldset>
    );
  }

  const current = getAt(value, path);

  if (field.type === "array") {
    return (
      <div>
        <label className="block font-mono text-sm text-foreground">{field.name}</label>
        {field.description && <p className="text-xs text-muted-foreground">{field.description}</p>}
        <textarea
          className="mt-1 w-full rounded border border-input bg-card px-2 py-1 font-mono text-xs"
          rows={Math.min(8, Array.isArray(current) ? current.length + 1 : 2)}
          value={JSON.stringify(current ?? [], null, 2)}
          onChange={(e) => {
            try {
              onChange(path, JSON.parse(e.target.value));
            } catch {
              /* ignore invalid JSON until valid */
            }
          }}
        />
      </div>
    );
  }

  if (field.type === "boolean") {
    return (
      <label className="flex items-center gap-2 text-sm text-foreground">
        <input
          type="checkbox"
          checked={Boolean(current)}
          onChange={(e) => onChange(path, e.target.checked)}
        />
        <span className="font-mono">{field.name}</span>
        {field.description && <span className="text-xs text-muted-foreground">— {field.description}</span>}
      </label>
    );
  }

  // string | number | null -> text input
  return (
    <div>
      <label className="block font-mono text-sm text-foreground">{field.name}</label>
      {field.description && <p className="text-xs text-muted-foreground">{field.description}</p>}
      <input
        type={field.type === "number" ? "number" : "text"}
        className="mt-1 w-full rounded border border-input bg-card px-2 py-1 text-sm"
        value={current == null ? "" : String(current)}
        onChange={(e) =>
          onChange(path, field.type === "number" ? Number(e.target.value) : e.target.value)
        }
      />
    </div>
  );
}

interface DynamicFormProps {
  schema: Field[];
  example: Json;
}

export default function DynamicForm({ schema, example }: DynamicFormProps) {
  const [config, setConfig] = useState<Json>(example ?? {});

  const onChange = (path: (string | number)[], value: Json) =>
    setConfig((prev) => setAt(prev, path, value));

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <div className="space-y-3">
        {schema.map((f) => (
          <FieldRow key={f.name} field={f} path={[f.name]} value={config} onChange={onChange} />
        ))}
      </div>
      <div>
        <p className="mb-1 text-xs font-medium uppercase text-muted-foreground">Config preview</p>
        <pre className="max-h-[28rem] overflow-auto rounded-lg border border-border bg-muted p-3 text-xs">
          {JSON.stringify(config, null, 2)}
        </pre>
      </div>
    </div>
  );
}
