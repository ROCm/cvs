export function fmt(ts?: string | null): string {
  if (!ts) return "—";
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? "—" : d.toLocaleString();
}

// humanBytes renders a byte counter compactly (e.g. 1.5 GB).
export function humanBytes(n: number): string {
  if (!Number.isFinite(n)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

// humanNum adds thousands separators to a counter.
export function humanNum(n: number): string {
  return Number.isFinite(n) ? n.toLocaleString() : "—";
}
