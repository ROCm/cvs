/**
 * CPUMetricsPage — Compute → CPUs → Metrics
 *
 * Live CPU and memory performance metrics from all cluster nodes:
 *   - Per-node CPU utilization (user/system/idle/iowait %)
 *   - Load averages (1m / 5m / 15m)
 *   - Memory utilization
 */

import { useState, useEffect } from 'react'
import { RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { api } from '@/services/api'
import {
  HorizontalStackedBar, DonutChart, COLORS,
} from '@/components/charts/ClusterCharts'
import { ExpandableChartCard } from '@/components/charts/ChartModal'
import { TopNodesChart } from '@/components/charts/TopNodesChart'

interface CpuStat {
  user_pct: number; system_pct: number; idle_pct: number
  iowait_pct: number; steal_pct: number; used_pct: number
}

interface HostMetrics {
  cpu_stat: Record<string, CpuStat>
  loadavg: { load1: number; load5: number; load15: number }
  mem_summary: {
    total: string; used: string; available: string; used_pct: number
    swap_total: string; swap_used: string
  }
}

interface CPUMetricsData {
  metrics: Record<string, HostMetrics>
  errors: Record<string, string>
  last_updated: string | null
}

function pctBar(pct: number, color: string) {
  return `<div style="display:flex;align-items:center;gap:6px">
    <div style="flex:1;background:#e5e7eb;border-radius:4px;height:8px">
      <div style="width:${Math.min(pct, 100)}%;background:${color};height:8px;border-radius:4px"></div>
    </div>
    <span style="font-size:11px;min-width:38px;text-align:right">${pct}%</span>
  </div>`
}

function colorForPct(pct: number) {
  if (pct >= 90) return '#ef4444'
  if (pct >= 70) return '#f59e0b'
  return '#3b82f6'
}

export function CPUMetricsPage() {
  const [data, setData]       = useState<CPUMetricsData | null>(null)
  const [loading, setLoading] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    try {
      const resp = await api.getCPUData() as any
      setData(resp)
    } catch (e) {
      console.error('Failed to fetch CPU metrics:', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    const id = setInterval(fetchData, 300_000)
    return () => clearInterval(id)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const hosts = Object.keys(data?.metrics ?? {})

  // Build aggregate CPU table (one row per node — aggregate "cpu" entry)
  const cpuRows = hosts.map(host => {
    const m = data!.metrics[host]
    const agg = m?.cpu_stat?.cpu
    return {
      node:       host,
      used_pct:   agg?.used_pct   ?? 0,
      user_pct:   agg?.user_pct   ?? 0,
      system_pct: agg?.system_pct ?? 0,
      iowait_pct: agg?.iowait_pct ?? 0,
      idle_pct:   agg?.idle_pct   ?? 0,
      load1:      m?.loadavg?.load1  ?? 0,
      load5:      m?.loadavg?.load5  ?? 0,
      load15:     m?.loadavg?.load15 ?? 0,
    }
  })

  // Build memory table
  const memRows = hosts.map(host => {
    const ms = data?.metrics?.[host]?.mem_summary
    return {
      node:       host,
      total:      ms?.total      ?? '—',
      used:       ms?.used       ?? '—',
      available:  ms?.available  ?? '—',
      used_pct:   ms?.used_pct   ?? 0,
      swap_total: ms?.swap_total ?? '—',
      swap_used:  ms?.swap_used  ?? '—',
    }
  })

  const cpuCols = [
    { title: 'Node',       data: 'node',       className: 'dt-left font-medium text-xs' },
    {
      title: 'CPU Used %',
      data:  'used_pct',
      className: 'dt-left',
      render: (v: number) => pctBar(v, colorForPct(v)),
    },
    { title: 'User %',     data: 'user_pct',   className: 'dt-right text-xs',
      render: (v: number) => `${v}%` },
    { title: 'System %',   data: 'system_pct', className: 'dt-right text-xs',
      render: (v: number) => `${v}%` },
    { title: 'I/O Wait %', data: 'iowait_pct', className: 'dt-right text-xs',
      render: (v: number, _t: string) =>
        v > 5 ? `<span class="text-orange-600 font-semibold">${v}%</span>` : `${v}%` },
    { title: 'Idle %',     data: 'idle_pct',   className: 'dt-right text-xs',
      render: (v: number) => `${v}%` },
    { title: 'Load 1m',    data: 'load1',      className: 'dt-right text-xs' },
    { title: 'Load 5m',    data: 'load5',      className: 'dt-right text-xs' },
    { title: 'Load 15m',   data: 'load15',     className: 'dt-right text-xs' },
  ]

  const memCols = [
    { title: 'Node',        data: 'node',       className: 'dt-left font-medium text-xs' },
    {
      title: 'Mem Used %',
      data:  'used_pct',
      className: 'dt-left',
      render: (v: number) => pctBar(v, colorForPct(v)),
    },
    { title: 'Total',       data: 'total',      className: 'dt-right text-xs' },
    { title: 'Used',        data: 'used',       className: 'dt-right text-xs' },
    { title: 'Available',   data: 'available',  className: 'dt-right text-xs' },
    { title: 'Swap Total',  data: 'swap_total', className: 'dt-right text-xs' },
    { title: 'Swap Used',   data: 'swap_used',  className: 'dt-right text-xs' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">CPU & Memory Metrics</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Live utilization from /proc/stat and /proc/meminfo · refreshes every 60s
            {data?.last_updated && ` · ${new Date(data.last_updated).toLocaleString()}`}
          </p>
        </div>
        <button onClick={fetchData} disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium">
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Errors */}
      {data && Object.keys(data.errors ?? {}).length > 0 && (
        <div className="p-3 bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-700 rounded-lg">
          <p className="text-xs font-semibold text-yellow-800 dark:text-yellow-300 mb-1">Collection errors:</p>
          <ul className="text-xs text-yellow-700 dark:text-yellow-400 space-y-0.5">
            {Object.entries(data.errors).map(([h, e]) => (
              <li key={h}><span className="font-mono">{h}</span>: {e}</li>
            ))}
          </ul>
        </div>
      )}

      {!data && (
        <Card>
          <CardContent className="py-12 text-center text-gray-400">
            <p>No CPU metrics yet. Collector runs every 60 seconds.</p>
          </CardContent>
        </Card>
      )}

      {data && hosts.length > 0 && (
        <>
          {/* ── Top 10 Nodes by CPU Utilization ── */}
          <TopNodesChart
            data={cpuRows.map(r => ({ node: r.node, value: r.used_pct, label: `${r.used_pct}%` }))}
            title="CPU Utilization"
            unit="%"
            warnThreshold={70}
            critThreshold={90}
          />

          {/* ── Visual Overview ── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ExpandableChartCard title="Avg CPU Utilization"
              small={<div className="flex justify-center"><DonutChart title="" data={[{name:'Used',value:Math.round(cpuRows.reduce((s,r)=>s+r.used_pct,0)/(cpuRows.length||1)),color:COLORS.blue},{name:'System',value:Math.round(cpuRows.reduce((s,r)=>s+r.system_pct,0)/(cpuRows.length||1)),color:COLORS.orange},{name:'Idle',value:Math.round(cpuRows.reduce((s,r)=>s+r.idle_pct,0)/(cpuRows.length||1)),color:COLORS.gray}]} centerLabel={`${(cpuRows.reduce((s,r)=>s+r.used_pct,0)/(cpuRows.length||1)).toFixed(0)}%`} centerSub="avg used" size={170}/></div>}
              large={<div className="flex justify-center pt-8"><DonutChart title="" data={[{name:'Used',value:Math.round(cpuRows.reduce((s,r)=>s+r.used_pct,0)/(cpuRows.length||1)),color:COLORS.blue},{name:'System',value:Math.round(cpuRows.reduce((s,r)=>s+r.system_pct,0)/(cpuRows.length||1)),color:COLORS.orange},{name:'Idle',value:Math.round(cpuRows.reduce((s,r)=>s+r.idle_pct,0)/(cpuRows.length||1)),color:COLORS.gray}]} centerLabel={`${(cpuRows.reduce((s,r)=>s+r.used_pct,0)/(cpuRows.length||1)).toFixed(0)}%`} centerSub="avg used" size={340}/></div>}
            />
            <ExpandableChartCard title="Avg Memory Utilization"
              small={<div className="flex justify-center"><DonutChart title="" data={[{name:'Used',value:Math.round(memRows.reduce((s,r)=>s+r.used_pct,0)/(memRows.length||1)),color:COLORS.purple},{name:'Available',value:Math.round(100-memRows.reduce((s,r)=>s+r.used_pct,0)/(memRows.length||1)),color:COLORS.gray}]} centerLabel={`${(memRows.reduce((s,r)=>s+r.used_pct,0)/(memRows.length||1)).toFixed(0)}%`} centerSub="avg used" size={170}/></div>}
              large={<div className="flex justify-center pt-8"><DonutChart title="" data={[{name:'Used',value:Math.round(memRows.reduce((s,r)=>s+r.used_pct,0)/(memRows.length||1)),color:COLORS.purple},{name:'Available',value:Math.round(100-memRows.reduce((s,r)=>s+r.used_pct,0)/(memRows.length||1)),color:COLORS.gray}]} centerLabel={`${(memRows.reduce((s,r)=>s+r.used_pct,0)/(memRows.length||1)).toFixed(0)}%`} centerSub="avg used" size={340}/></div>}
            />
          </div>

          {/* Load average — full-width row, horizontal bars so node names fit */}
          <ExpandableChartCard title="Load Average by Node"
            small={<HorizontalStackedBar data={cpuRows.map(r=>({name:r.node,'1m':r.load1,'5m':r.load5,'15m':r.load15}))} keys={['1m','5m','15m']} colors={[COLORS.blue,COLORS.teal,COLORS.indigo]} unit="" maxValue={Math.max(1,...cpuRows.map(r=>Math.max(r.load1,r.load5,r.load15)))*1.1} height={Math.max(160,cpuRows.length*32+60)}/>}
            large={<HorizontalStackedBar data={cpuRows.map(r=>({name:r.node,'1m':r.load1,'5m':r.load5,'15m':r.load15}))} keys={['1m','5m','15m']} colors={[COLORS.blue,COLORS.teal,COLORS.indigo]} unit="" maxValue={Math.max(1,...cpuRows.map(r=>Math.max(r.load1,r.load5,r.load15)))*1.1} height={Math.max(250,cpuRows.length*42+80)}/>}
          />

          <ExpandableChartCard title="CPU Breakdown per Node"
            small={<HorizontalStackedBar data={cpuRows.map(r=>({name:r.node,User:r.user_pct,System:r.system_pct,IOWait:r.iowait_pct,Idle:r.idle_pct}))} keys={['User','System','IOWait','Idle']} colors={[COLORS.blue,COLORS.orange,COLORS.amber,COLORS.gray]} unit="%" maxValue={100}/>}
            large={<HorizontalStackedBar data={cpuRows.map(r=>({name:r.node,User:r.user_pct,System:r.system_pct,IOWait:r.iowait_pct,Idle:r.idle_pct}))} keys={['User','System','IOWait','Idle']} colors={[COLORS.blue,COLORS.orange,COLORS.amber,COLORS.gray]} unit="%" maxValue={100} height={Math.max(300, cpuRows.length*45+80)}/>}
          />

          <ExpandableChartCard title="Memory Usage per Node"
            small={<HorizontalStackedBar data={memRows.map(r=>({name:r.node,Used:r.used_pct,Available:100-r.used_pct}))} keys={['Used','Available']} colors={[COLORS.purple,COLORS.gray]} unit="%" maxValue={100}/>}
            large={<HorizontalStackedBar data={memRows.map(r=>({name:r.node,Used:r.used_pct,Available:100-r.used_pct}))} keys={['Used','Available']} colors={[COLORS.purple,COLORS.gray]} unit="%" maxValue={100} height={Math.max(300, memRows.length*45+80)}/>}
          />

          {/* Summary stat cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              {
                label: 'Avg CPU Used',
                value: `${(cpuRows.reduce((s, r) => s + r.used_pct, 0) / (cpuRows.length || 1)).toFixed(1)}%`,
                color: 'text-blue-600',
              },
              {
                label: 'Avg Load (1m)',
                value: (cpuRows.reduce((s, r) => s + r.load1, 0) / (cpuRows.length || 1)).toFixed(2),
                color: 'text-purple-600',
              },
              {
                label: 'Avg Mem Used',
                value: `${(memRows.reduce((s, r) => s + r.used_pct, 0) / (memRows.length || 1)).toFixed(1)}%`,
                color: 'text-green-600',
              },
              { label: 'Nodes', value: hosts.length, color: 'text-gray-700 dark:text-gray-300' },
            ].map(({ label, value, color }) => (
              <Card key={label}>
                <CardContent className="pt-4 pb-4 text-center">
                  <div className={`text-2xl font-bold ${color}`}>{value}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{label}</div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* CPU utilization table */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">CPU Utilization — All Nodes</CardTitle>
            </CardHeader>
            <CardContent>
              <CustomDataTable
                columns={cpuCols}
                data={cpuRows}
                defaultPageLength={25}
                pageLengthOptions={[25, 50, 100]}
                scrollX={true}
              />
            </CardContent>
          </Card>

          {/* Memory table */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Memory Utilization — All Nodes</CardTitle>
            </CardHeader>
            <CardContent>
              <CustomDataTable
                columns={memCols}
                data={memRows}
                defaultPageLength={25}
                pageLengthOptions={[25, 50, 100]}
                scrollX={true}
              />
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
