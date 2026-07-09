/**
 * TopNodesChart — reusable horizontal bar chart showing the top N nodes
 * sorted by a given metric value. Used across Compute, Network and Storage.
 */

import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts'
import { ExpandableChartCard } from './ChartModal'

interface NodeValue {
  node: string
  value: number
  label?: string   // extra context (e.g. "96%" or "4.2 GB/s")
}

interface TopNodesChartProps {
  data: NodeValue[]
  title: string
  unit?: string
  color?: string
  /** How many top nodes to show (default 10) */
  topN?: number
  /** If true, bars that exceed warnThreshold are shown in amber */
  warnThreshold?: number
  /** If true, bars that exceed critThreshold are shown in red */
  critThreshold?: number
}

function buildRows(data: NodeValue[], topN: number, unit: string, warnThreshold?: number, critThreshold?: number) {
  return [...data]
    .sort((a, b) => b.value - a.value)
    .slice(0, topN)
    .map(d => ({
      name:   d.node,          // full hostname — no truncation
      value:  d.value,
      _label: d.label ?? `${d.value}${unit}`,
      _color: critThreshold && d.value >= critThreshold ? '#ef4444'
            : warnThreshold && d.value >= warnThreshold ? '#f59e0b'
            : '#3b82f6',
    }))
}

function yAxisWidth(rows: { name: string }[], charPx = 7, min = 120, max = 380): number {
  const longest = rows.reduce((m, r) => Math.max(m, r.name.length), 0)
  return Math.min(max, Math.max(min, longest * charPx))
}

function Chart({ rows, unit, height }: { rows: ReturnType<typeof buildRows>; unit: string; height: number }) {
  const yWidth = yAxisWidth(rows)
  return (
    <div style={{ overflowX: 'auto' }}>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 48, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e5e7eb" strokeOpacity={0.5} />
          <XAxis type="number" tick={{ fontSize: 10 }} tickFormatter={v => `${v}${unit}`} />
          <YAxis type="category" dataKey="name" width={yWidth} tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: 'rgba(17,24,39,0.92)', border: 'none', borderRadius: 8, color: '#f9fafb', fontSize: 11 }}
            formatter={(v: number, _k: string, entry: any) => [entry?.payload?._label ?? `${v}${unit}`, 'Value']}
          />
          <Bar dataKey="value" maxBarSize={18} radius={[0, 3, 3, 0]}>
            {rows.map((r, i) => <Cell key={i} fill={r._color} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function TopNodesChart({
  data,
  title,
  unit = '',
  color = '#3b82f6',
  topN = 10,
  warnThreshold,
  critThreshold,
}: TopNodesChartProps) {
  if (!data || data.length === 0) return null

  const rows      = buildRows(data, topN, unit, warnThreshold, critThreshold)
  const rowsLarge = buildRows(data, Math.min(data.length, 20), unit, warnThreshold, critThreshold)
  const smallH    = Math.max(120, rows.length * 28 + 40)
  const largeH    = Math.max(250, rowsLarge.length * 36 + 60)

  return (
    <ExpandableChartCard
      title={`${title} — Top ${rows.length} Nodes`}
      small={<Chart rows={rows}      unit={unit} height={smallH} />}
      large={<Chart rows={rowsLarge} unit={unit} height={largeH} />}
    />
  )
}
