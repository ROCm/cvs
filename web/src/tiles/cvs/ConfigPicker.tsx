import { useEffect, useMemo } from "react";
import type { ConfigRef } from "@/shared/api";

interface ConfigPickerProps {
  examples: ConfigRef[];
  // Resolved config name currently selected (file name without extension).
  selected: string | null;
  onSelect: (name: string) => void;
  // GPU type detected by the inventory probe (e.g. "MI300X"); used to
  // pre-select the platform facet.
  detectedGpu?: string;
}

// distinct returns the unique, sorted, non-empty values produced by `pick`.
function distinct(rows: ConfigRef[], pick: (c: ConfigRef) => string | undefined): string[] {
  const set = new Set<string>();
  for (const r of rows) {
    const v = pick(r);
    if (v) set.add(v);
  }
  return [...set].sort();
}

// matchPlatform finds the facet platform best matching the detected GPU type,
// e.g. detected "AMD Instinct MI300X" -> "mi300x".
function matchPlatform(platforms: string[], gpu?: string): string | undefined {
  if (!gpu) return undefined;
  const g = gpu.toLowerCase();
  return platforms.find((p) => g.includes(p));
}

// hasFacets reports whether these examples carry the platform/model/topology
// convention (multi-config dirs). Flat single-config suites do not.
function hasFacets(examples: ConfigRef[]): boolean {
  return examples.some((e) => e.platform || e.model || e.topology);
}

export default function ConfigPicker({ examples, selected, onSelect, detectedGpu }: ConfigPickerProps) {
  const faceted = hasFacets(examples);
  const current = useMemo(
    () => examples.find((e) => e.name === selected),
    [examples, selected],
  );

  const platform = current?.platform ?? "";
  const model = current?.model ?? "";
  const topology = current?.topology ?? "";

  const platforms = useMemo(() => distinct(examples, (c) => c.platform), [examples]);
  const models = useMemo(
    () => distinct(examples.filter((c) => c.platform === platform), (c) => c.model),
    [examples, platform],
  );
  const topologies = useMemo(
    () =>
      distinct(
        examples.filter((c) => c.platform === platform && c.model === model),
        (c) => c.topology,
      ),
    [examples, platform, model],
  );

  // Initialize / re-initialize selection when the example set changes: prefer
  // the config whose platform matches the detected GPU, else the first.
  useEffect(() => {
    if (examples.length === 0) return;
    if (current) return; // already valid
    if (faceted) {
      const wantPlat = matchPlatform(distinct(examples, (c) => c.platform), detectedGpu);
      const match = wantPlat
        ? examples.find((c) => c.platform === wantPlat)
        : undefined;
      onSelect((match ?? examples[0]).name);
    } else {
      onSelect(examples[0].name);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [examples, detectedGpu]);

  if (examples.length <= 1 || !faceted) {
    // Nothing to pick (single/flat config); the parent loads it directly.
    return null;
  }

  // Re-resolve a concrete config when a facet changes, keeping the other facets
  // where possible and falling back to the first valid combination.
  function selectFacet(next: Partial<Pick<ConfigRef, "platform" | "model" | "topology">>) {
    const p = next.platform ?? platform;
    const m = next.platform ? undefined : next.model ?? model;
    const t = next.platform || next.model ? undefined : next.topology ?? topology;

    let pool = examples.filter((c) => c.platform === p);
    if (m !== undefined) pool = pool.filter((c) => c.model === m);
    if (t !== undefined) pool = pool.filter((c) => c.topology === t);
    if (pool.length > 0) onSelect(pool[0].name);
  }

  const category = current?.category;
  const framework = current?.framework;

  return (
    <div className="mb-6 rounded-xl border border-border bg-card p-4">
      <div className="mb-3 text-sm font-medium text-foreground">Configuration</div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {(category || framework) && (
          <div className="col-span-2 sm:col-span-3 text-xs text-muted-foreground">
            {category && <span className="mr-3">Category: <span className="font-mono text-foreground">{category}</span></span>}
            {framework && <span>Framework: <span className="font-mono text-foreground">{framework}</span></span>}
          </div>
        )}

        <Facet
          label="Platform"
          value={platform}
          options={platforms}
          onChange={(v) => selectFacet({ platform: v })}
          hint={matchPlatform(platforms, detectedGpu) === platform ? "from detected GPU" : undefined}
        />
        <Facet
          label="Model"
          value={model}
          options={models}
          onChange={(v) => selectFacet({ model: v })}
        />
        <Facet
          label="Topology"
          value={topology}
          options={topologies}
          onChange={(v) => selectFacet({ topology: v })}
        />
      </div>
    </div>
  );
}

function Facet({
  label,
  value,
  options,
  onChange,
  hint,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
  hint?: string;
}) {
  if (options.length === 0) return null;
  return (
    <label className="flex flex-col gap-1 text-xs">
      <span className="font-medium text-muted-foreground">
        {label}
        {hint && <span className="ml-1 font-normal text-primary">({hint})</span>}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-border bg-background px-2 py-1.5 text-sm text-foreground focus:border-primary focus:outline-none"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}
