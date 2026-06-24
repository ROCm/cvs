import { useEffect, useState } from "react";
import { ChevronDown, ChevronRight, Plus, RotateCcw, Trash2 } from "lucide-react";
import type { Field } from "@/shared/api";

// JSON-ish value type for config trees.
type Json = string | number | boolean | null | Json[] | { [k: string]: Json };

// Placeholder tokens that the user is required to replace before running.
// Matched case-insensitively as a substring so embedded forms like
// "<changeme>-no of nodes" or "[<changeme>]" are also flagged.
// Add more tokens here later (e.g. "<change_me>", "TODO") as needed.
const PLACEHOLDER_TOKENS = ["<changeme>"];

function containsPlaceholder(value: Json): boolean {
  if (typeof value !== "string") return false;
  const lower = value.toLowerCase();
  return PLACEHOLDER_TOKENS.some((t) => lower.includes(t));
}

// Counts unresolved placeholders anywhere under a value (skips `_comment_*`
// helper keys, which are surfaced as descriptions rather than edited).
function countPlaceholders(value: Json): number {
  if (value == null) return 0;
  if (typeof value === "string") return containsPlaceholder(value) ? 1 : 0;
  if (Array.isArray(value)) return value.reduce<number>((n, v) => n + countPlaceholders(v), 0);
  if (typeof value === "object") {
    return Object.entries(value).reduce<number>(
      (n, [k, v]) => (k.startsWith("_comment") ? n : n + countPlaceholders(v)),
      0,
    );
  }
  return 0;
}

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

// Client-side shape inference for a single value. Array elements aren't fully
// described by the server schema's single `item` descriptor (arrays can be
// heterogeneous, e.g. `tests` entries with differing keys), so we infer each
// element's fields from its actual value. Mirrors the server's InferSchema:
// `_comment_*` keys are dropped and fields are sorted by name.
function inferField(name: string, value: Json): Field {
  if (Array.isArray(value)) {
    const f: Field = { name, type: "array" };
    if (value.length > 0) f.item = inferField("", value[0]);
    return f;
  }
  if (value !== null && typeof value === "object") {
    const obj = value as Record<string, Json>;
    const meta = (prefix: string, field: string): string | undefined => {
      const v = obj[`${prefix}${field}`];
      if (v === undefined) return undefined;
      return typeof v === "string" ? v : JSON.stringify(v);
    };
    const fields: Field[] = Object.keys(obj)
      .filter((k) => !k.startsWith("_comment") && !k.startsWith("_example"))
      .map((k) => {
        const f = inferField(k, obj[k]);
        const desc = meta("_comment_", k);
        const ex = meta("_example_", k);
        if (desc) f.description = desc;
        if (ex) f.example = ex;
        return f;
      })
      .sort((a, b) => a.name.localeCompare(b.name));
    return { name, type: "object", fields };
  }
  switch (typeof value) {
    case "boolean":
      return { name, type: "boolean" };
    case "number":
      return { name, type: "number" };
    default:
      return { name, type: value === null ? "null" : "string" };
  }
}

// A blank value matching a field descriptor, used when adding the first item to
// an empty array (where there's no existing element to clone).
function emptyFromField(f?: Field): Json {
  if (!f) return "";
  switch (f.type) {
    case "object": {
      const o: Record<string, Json> = {};
      for (const c of f.fields ?? []) o[c.name] = emptyFromField(c);
      return o;
    }
    case "array":
      return [];
    case "boolean":
      return false;
    case "number":
      return 0;
    default:
      return "";
  }
}

// Label for an array element card: prefer its `name` field, else a 1-based index.
function elementTitle(el: Json, i: number): string {
  if (el !== null && typeof el === "object" && !Array.isArray(el)) {
    const name = (el as Record<string, Json>).name;
    if (typeof name === "string" && name.trim()) return name;
  }
  return `Item ${i + 1}`;
}

const PLACEHOLDER_BADGE =
  "ml-1 rounded-full border border-destructive/40 bg-destructive/10 px-1.5 py-0.5 text-[10px] font-semibold text-destructive";

interface FieldProps {
  field: Field;
  path: (string | number)[];
  depth: number;
  value: Json;
  onChange: (path: (string | number)[], value: Json) => void;
  isOpen: (key: string, depth: number) => boolean;
  toggle: (key: string, depth: number) => void;
}

function FieldRow({ field, path, depth, value, onChange, isOpen, toggle }: FieldProps) {
  const key = path.join("\u0000");

  if (field.type === "object") {
    const open = isOpen(key, depth);
    const count = countPlaceholders(getAt(value, path));
    return (
      <fieldset className="rounded-lg border border-border p-3">
        <legend className="px-1">
          <button
            type="button"
            onClick={() => toggle(key, depth)}
            className="flex items-center gap-1 text-left hover:text-primary"
          >
            {open ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}
            <span className="font-mono text-sm font-medium text-foreground">{field.name}</span>
            {count > 0 && <span className={PLACEHOLDER_BADGE}>{count} to set</span>}
          </button>
        </legend>
        {open && (
          <>
            {field.description && <p className="mb-2 text-xs text-muted-foreground">{field.description}</p>}
            {field.example && (
              <p className="mb-2 text-xs text-muted-foreground">
                Example: <code className="rounded bg-muted px-1 py-0.5">{field.example}</code>
              </p>
            )}
            <div className="space-y-3">
              {(field.fields ?? []).map((child) => (
                <FieldRow
                  key={child.name}
                  field={child}
                  path={[...path, child.name]}
                  depth={depth + 1}
                  value={value}
                  onChange={onChange}
                  isOpen={isOpen}
                  toggle={toggle}
                />
              ))}
            </div>
          </>
        )}
      </fieldset>
    );
  }

  const current = getAt(value, path);

  if (field.type === "array") {
    const open = isOpen(key, depth);
    const arr = Array.isArray(current) ? (current as Json[]) : [];
    const count = countPlaceholders(current);

    const removeAt = (i: number) =>
      onChange(path, arr.filter((_, j) => j !== i));
    const addItem = () => {
      const next =
        arr.length > 0
          ? (JSON.parse(JSON.stringify(arr[arr.length - 1])) as Json)
          : emptyFromField(field.item);
      onChange(path, [...arr, next]);
    };

    return (
      <fieldset className="rounded-lg border border-border p-3">
        <legend className="px-1">
          <button
            type="button"
            onClick={() => toggle(key, depth)}
            className="flex items-center gap-1 text-left hover:text-primary"
          >
            {open ? (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            )}
            <span className="font-mono text-sm font-medium text-foreground">{field.name}</span>
            <span className="text-xs text-muted-foreground">[{arr.length}]</span>
            {count > 0 && <span className={PLACEHOLDER_BADGE}>{count} to set</span>}
          </button>
        </legend>
        {open && (
          <>
            {field.description && <p className="mb-2 text-xs text-muted-foreground">{field.description}</p>}
            {field.example && (
              <p className="mb-2 text-xs text-muted-foreground">
                Example: <code className="rounded bg-muted px-1 py-0.5">{field.example}</code>
              </p>
            )}
            <div className="space-y-2">
              {arr.map((el, i) => {
                const elField: Field = { ...inferField("", el), name: elementTitle(el, i) };
                return (
                  <div key={i} className="flex items-start gap-2">
                    <div className="min-w-0 flex-1">
                      <FieldRow
                        field={elField}
                        path={[...path, i]}
                        depth={depth + 1}
                        value={value}
                        onChange={onChange}
                        isOpen={isOpen}
                        toggle={toggle}
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => removeAt(i)}
                      title="Remove item"
                      className="mt-0.5 shrink-0 rounded border border-border p-1.5 text-muted-foreground hover:border-destructive hover:text-destructive"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                );
              })}
              <button
                type="button"
                onClick={addItem}
                className="inline-flex items-center gap-1 rounded border border-dashed border-border px-2 py-1 text-xs text-muted-foreground hover:border-primary hover:text-primary"
              >
                <Plus className="h-3 w-3" /> Add item
              </button>
            </div>
          </>
        )}
      </fieldset>
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
  const flagged = containsPlaceholder(current);
  return (
    <div>
      <label className={`block font-mono text-sm ${flagged ? "text-destructive" : "text-foreground"}`}>
        {field.name}
        {flagged && <span className={PLACEHOLDER_BADGE}>required</span>}
      </label>
      {field.description && <p className="text-xs text-muted-foreground">{field.description}</p>}
      {field.example && (
        <p className="text-xs text-muted-foreground">
          Example: <code className="rounded bg-muted px-1 py-0.5">{field.example}</code>
        </p>
      )}
      <input
        type={field.type === "number" ? "number" : "text"}
        className={`mt-1 w-full rounded border bg-card px-2 py-1 text-sm ${
          flagged ? "border-destructive focus:ring-destructive" : "border-input"
        }`}
        value={current == null ? "" : String(current)}
        placeholder={field.example}
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
  // Notifies the parent of the current edited config and the number of
  // unresolved `<changeme>` placeholders (so a Run flow can submit / gate).
  onChange?: (config: Json, remaining: number) => void;
}

export default function DynamicForm({ schema, example, onChange: onConfigChange }: DynamicFormProps) {
  const [config, setConfig] = useState<Json>(example ?? {});

  // Open-state model: explicit per-section overrides in `openMap`, an optional
  // global override from Expand/Collapse all, and a depth-based default
  // (top-level open, nested collapsed). This drives both object and array
  // sections, including dynamically-added array items.
  const [openMap, setOpenMap] = useState<Record<string, boolean>>({});
  const [override, setOverride] = useState<"expand" | "collapse" | null>(null);

  // The server-provided example is the baseline ("defaults") the form seeds
  // from. `useState` only reads it on first mount, so resync whenever it
  // changes — otherwise navigating to a different suite would keep the previous
  // suite's edits. This also gives Reset a stable source of truth.
  useEffect(() => {
    setConfig(example ?? {});
    setOpenMap({});
    setOverride(null);
  }, [example]);

  const isOpen = (key: string, depth: number) => {
    if (key in openMap) return openMap[key];
    if (override === "expand") return true;
    if (override === "collapse") return false;
    return depth < 1;
  };
  const toggle = (key: string, depth: number) =>
    setOpenMap((p) => ({ ...p, [key]: !isOpen(key, depth) }));
  const setAll = (expand: boolean) => {
    setOverride(expand ? "expand" : "collapse");
    setOpenMap({});
  };

  const onChange = (path: (string | number)[], value: Json) =>
    setConfig((prev) => setAt(prev, path, value));

  // Surface the current config + remaining placeholder count to the parent.
  useEffect(() => {
    onConfigChange?.(config, countPlaceholders(config));
    // onConfigChange is treated as stable; depending on it would loop on inline fns.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config]);

  const baseline = example ?? {};
  const dirty = JSON.stringify(config) !== JSON.stringify(baseline);
  const handleReset = () => {
    if (!dirty) return;
    if (!window.confirm("Reset the form to its default values? Your changes will be lost.")) return;
    setConfig(baseline);
    setOpenMap({});
    setOverride(null);
  };

  const remaining = countPlaceholders(config);

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="text-xs">
            {remaining > 0 ? (
              <span className="font-medium text-destructive">
                {remaining} value{remaining === 1 ? "" : "s"} still need attention
              </span>
            ) : (
              <span className="text-muted-foreground">All placeholders resolved</span>
            )}
          </div>
          <div className="flex gap-2 text-xs">
            <button
              type="button"
              onClick={handleReset}
              disabled={!dirty}
              title={dirty ? "Reset the form to its default values" : "No changes to reset"}
              className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-muted-foreground hover:text-primary disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:text-muted-foreground"
            >
              <RotateCcw className="h-3 w-3" /> Reset
            </button>
            <button
              type="button"
              onClick={() => setAll(true)}
              className="rounded border border-border px-2 py-1 text-muted-foreground hover:text-primary"
            >
              Expand all
            </button>
            <button
              type="button"
              onClick={() => setAll(false)}
              className="rounded border border-border px-2 py-1 text-muted-foreground hover:text-primary"
            >
              Collapse all
            </button>
          </div>
        </div>
        {schema.map((f) => (
          <FieldRow
            key={f.name}
            field={f}
            path={[f.name]}
            depth={0}
            value={config}
            onChange={onChange}
            isOpen={isOpen}
            toggle={toggle}
          />
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
