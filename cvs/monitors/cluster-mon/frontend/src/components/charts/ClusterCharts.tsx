/**
 * Reusable chart components for cluster monitoring visualizations.
 * Uses Recharts 2.x — already a project dependency.
 */

import { useId } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell,
} from 'recharts'

export const COLORS = {
  blue:   '#3b82f6',
  green:  '#22c55e',
  red:    '#ef4444',
  amber:  '#f59e0b',
  purple: '#a855f7',
  gray:   '#9ca3af',
  teal:   '#14b8a6',
  orange: '#f97316',
  indigo: '#6366f1',
}

// Lighter "top" shade for each colour — used in gradient fills
const GRAD_TOP: Record<string, string> = {
  '#3b82f6': '#93c5fd',
  '#22c55e': '#86efac',
  '#ef4444': '#fca5a5',
  '#f59e0b': '#fcd34d',
  '#a855f7': '#d8b4fe',
  '#9ca3af': '#e5e7eb',
  '#14b8a6': '#99f6e4',
  '#f97316': '#fdba74',
  '#6366f1': '#a5b4fc',
}

// Safe gradient ID from hex colour
const gid = (c: string) => `cg${c.replace('#', '')}`

// SVG <defs> that define gradient fills for bars.
// uid is a per-chart-instance prefix so IDs never clash across charts on the same page.
function GradDefs({ colors, uid, vertical = true }: { colors: string[]; uid: string; vertical?: boolean }) {
  const unique = [...new Set(colors)]
  return (
    <defs>
      {unique.map(c => {
        const top = GRAD_TOP[c] ?? c
        return (
          <linearGradient key={c} id={`${uid}${gid(c)}`}
            x1="0" y1={vertical ? '0' : '0'}
            x2={vertical ? '0' : '1'} y2={vertical ? '1' : '0'}>
            <stop offset="0%"   stopColor={top} stopOpacity={0.95} />
            <stop offset="100%" stopColor={c}   stopOpacity={1} />
          </linearGradient>
        )
      })}
    </defs>
  )
}

// Shorten long hostnames for chart labels
export function shortName(host: string, maxLen = 16): string {
  // Take the first component before the first dot if it fits
  const first = host.split('.')[0]
  return first.length <= maxLen ? first : first.slice(0, maxLen - 1) + '…'
}

// Format large numbers compactly
export function compactNum(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}G`
  if (n >= 1_000_000)     return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000)         return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

// ── Donut / Pie ──────────────────────────────────────────────────────────────

interface DonutSlice { name: string; value: number; color: string }

export function DonutChart({
  data, title, centerLabel, centerSub, size = 180,
}: {
  data: DonutSlice[]
  title: string
  centerLabel: string
  centerSub?: string
  size?: number
}) {
  const uid = useId().replace(/:/g, 'u')
  return (
    <div className="flex flex-col items-center">
      <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wide">{title}</p>
      <PieChart width={size} height={size}>
        <defs>
          {data.map((entry, i) => {
            const top = GRAD_TOP[entry.color] ?? entry.color
            return (
              <radialGradient key={i} id={`${uid}pg${i}`} cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor={top} stopOpacity={0.9} />
                <stop offset="100%" stopColor={entry.color} stopOpacity={1} />
              </radialGradient>
            )
          })}
        </defs>
        <Pie
          data={data}
          cx={size / 2 - 1}
          cy={size / 2 - 1}
          innerRadius={size * 0.3}
          outerRadius={size * 0.44}
          paddingAngle={2}
          dataKey="value"
        >
          {data.map((entry, i) => (
            <Cell key={i} fill={`url(#${uid}pg${i})`} stroke={entry.color} strokeWidth={0.5} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{ background:'rgba(17,24,39,0.92)', border:'none', borderRadius:8, color:'#f9fafb', fontSize:11 }}
          formatter={(v: number, name: string) => [v, name]}
        />
      </PieChart>
      {/* Center label overlaid via negative margin trick */}
      <div className="relative" style={{ marginTop: -(size * 0.62) }}>
        <div className="flex flex-col items-center justify-center"
          style={{ height: size * 0.62 }}>
          <span className="text-xl font-bold text-gray-900 dark:text-gray-100">{centerLabel}</span>
          {centerSub && <span className="text-xs text-gray-500 dark:text-gray-400">{centerSub}</span>}
        </div>
      </div>
      {/* Legend */}
      <div className="flex gap-3 mt-1 flex-wrap justify-center" style={{ marginTop: size * 0.38 }}>
        {data.map((d, i) => (
          <div key={i} className="flex items-center gap-1 text-xs text-gray-600 dark:text-gray-400">
            <div className="w-2.5 h-2.5 rounded-full" style={{ background: d.color }} />
            {d.name}: <span className="font-semibold">{d.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Horizontal stacked bar chart ─────────────────────────────────────────────

interface StackedBarData { name: string; [key: string]: number | string }

/** Compute a Y-axis width that fits the longest label without truncation. */
function yAxisWidth(names: string[], charPx = 7, min = 120, max = 320): number {
  const longest = names.reduce((m, n) => Math.max(m, n.length), 0)
  return Math.min(max, Math.max(min, longest * charPx))
}

export function HorizontalStackedBar({
  data, keys, colors, title, unit = '%', maxValue = 100, height,
}: {
  data: StackedBarData[]
  keys: string[]
  colors: string[]
  title: string
  unit?: string
  maxValue?: number
  height?: number
}) {
  const h = height ?? Math.max(120, data.length * 32 + 60)
  const yWidth = yAxisWidth(data.map(d => String(d.name)))
  return (
    <div>
      {title && <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wide">{title}</p>}
      <div style={{ overflowX: 'auto' }}>
      <ResponsiveContainer width="100%" height={h}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e5e7eb" strokeOpacity={0.5} />
          <XAxis type="number" domain={[0, maxValue]} tickFormatter={v => `${v}${unit}`}
            tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="name" width={yWidth}
            tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background:'rgba(17,24,39,0.92)', border:'none', borderRadius:8, color:'#f9fafb', fontSize:11 }}
            formatter={(v: number, k: string) => [`${v.toFixed(1)}${unit}`, k]}
          />
          {keys.map((k, i) => (
            <Bar key={k} dataKey={k} stackId="a"
              fill={colors[i]}
              maxBarSize={18} radius={i === keys.length - 1 ? [0, 3, 3, 0] : undefined} />
          ))}
        </BarChart>
      </ResponsiveContainer>
      </div>
      {/* Legend rendered as plain HTML below the chart — no SVG overlap */}
      <div className="flex justify-center gap-4 mt-2 flex-wrap">
        {keys.map((k, i) => (
          <div key={k} className="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-400">
            <div className="w-3 h-2.5 rounded-sm" style={{ background: colors[i] }} />
            {k}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Vertical bar chart ───────────────────────────────────────────────────────

export function VerticalBarChart({
  data, dataKey, color, title, unit = '', height = 200, xKey = 'name',
}: {
  data: any[]
  dataKey: string
  color: string
  title?: string
  unit?: string
  height?: number
  xKey?: string
}) {
  return (
    <div>
      {title && <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wide">{title}</p>}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ left: 8, right: 8, top: 4, bottom: 48 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" strokeOpacity={0.5} />
          <XAxis dataKey={xKey} tick={{ fontSize: 9 }} angle={-35} textAnchor="end"
            tickFormatter={v => shortName(String(v), 12)} interval={0} />
          <YAxis tick={{ fontSize: 10 }} tickFormatter={v => compactNum(Number(v))} />
          <Tooltip
            contentStyle={{ background:'rgba(17,24,39,0.92)', border:'none', borderRadius:8, color:'#f9fafb', fontSize:11 }}
            formatter={(v: number) => [`${compactNum(v)}${unit}`, dataKey]}
          />
          <Bar dataKey={dataKey} radius={[4, 4, 0, 0]} maxBarSize={30}>
            {data.map((d: any, i: number) => (
              <Cell key={i} fill={d._color ?? color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Grouped bar chart ────────────────────────────────────────────────────────

export function GroupedBarChart({
  data, keys, colors, title, unit = '', height = 220, xKey = 'name',
}: {
  data: any[]
  keys: string[]
  colors: string[]
  title?: string
  unit?: string
  height?: number
  xKey?: string
}) {
  return (
    <div>
      {title && <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wide">{title}</p>}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ left: 8, right: 8, top: 4, bottom: 48 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" strokeOpacity={0.5} />
          <XAxis dataKey={xKey} tick={{ fontSize: 9 }} angle={-35} textAnchor="end"
            tickFormatter={v => shortName(String(v), 12)} interval={0} />
          <YAxis tick={{ fontSize: 10 }} tickFormatter={v => `${v}${unit}`} />
          <Tooltip
            contentStyle={{ background:'rgba(17,24,39,0.92)', border:'none', borderRadius:8, color:'#f9fafb', fontSize:11 }}
            formatter={(v: number, k: string) => [`${v}${unit}`, k]}
          />
          {keys.map((k, i) => (
            <Bar key={k} dataKey={k}
              fill={colors[i]}
              radius={[4, 4, 0, 0]} maxBarSize={20} />
          ))}
        </BarChart>
      </ResponsiveContainer>
      {/* Legend rendered as plain HTML below the chart — no SVG overlap */}
      <div className="flex justify-center gap-4 mt-1 flex-wrap">
        {keys.map((k, i) => (
          <div key={k} className="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-400">
            <div className="w-3 h-2.5 rounded-sm" style={{ background: colors[i] }} />
            {k}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Stat card ────────────────────────────────────────────────────────────────

export function StatCard({
  label, value, sub, color = 'text-gray-900 dark:text-gray-100', bg = 'bg-gray-50 dark:bg-gray-800',
}: {
  label: string; value: string | number; sub?: string; color?: string; bg?: string
}) {
  return (
    <div className={`${bg} rounded-xl p-4 text-center`}>
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
      <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mt-0.5">{label}</div>
      {sub && <div className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">{sub}</div>}
    </div>
  )
}
