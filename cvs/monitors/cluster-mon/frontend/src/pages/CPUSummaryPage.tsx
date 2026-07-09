/**
 * CPUSummaryPage — Compute → CPUs → Summary
 *
 * Cluster-level view: one row per node, all nodes in one table.
 *
 * Table 1 — CPU Hardware (lscpu pivot)
 *   Columns: Node | Architecture | CPUs | Cores/Socket | Sockets | Threads/Core |
 *            Model Name | Max MHz | L2 Cache | L3 Cache | NUMA Nodes | Virtualization
 *
 * Table 2 — Memory Hardware
 *   Columns: Node | Total Memory | Available | Used % | Cached | Swap Total | Swap Used
 */

import { useState, useEffect } from 'react'
import { RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { api } from '@/services/api'

interface CPUData {
  summary: Record<string, {
    lscpu: Record<string, string>
    lsmem: { summary: Record<string, string>; ranges: Record<string, string>[] }
    mem_summary: {
      total: string; used: string; available: string; cached: string
      swap_total: string; swap_used: string; used_pct: number
    }
    logged_in:  Record<string, string>[]
    user_procs: Record<string, string>[]
    kfd_procs:  Record<string, string>[]
  }>
  errors: Record<string, string>
  last_updated: string | null
}

// lscpu fields to show as columns (subset — most useful for comparison)
const LSCPU_COLS = [
  { key: 'Architecture',       title: 'Architecture' },
  { key: 'CPU(s)',             title: 'Logical CPUs' },
  { key: 'Core(s) per socket', title: 'Cores/Socket' },
  { key: 'Socket(s)',          title: 'Sockets' },
  { key: 'Thread(s) per core', title: 'Threads/Core' },
  { key: 'Model name',         title: 'Model Name' },
  { key: 'CPU max MHz',        title: 'Max MHz' },
  { key: 'CPU min MHz',        title: 'Min MHz' },
  { key: 'L2 cache',           title: 'L2 Cache' },
  { key: 'L3 cache',           title: 'L3 Cache' },
  { key: 'NUMA node(s)',       title: 'NUMA Nodes' },
  { key: 'Virtualization',     title: 'Virtualization' },
]

function usedPctRender(pct: number): string {
  const color = pct >= 80 ? '#ef4444' : pct >= 60 ? '#f59e0b' : '#22c55e'
  return `<div style="display:flex;align-items:center;gap:6px">
    <div style="flex:1;background:#e5e7eb;border-radius:4px;height:6px;min-width:60px">
      <div style="width:${Math.min(pct,100)}%;background:${color};height:6px;border-radius:4px"></div>
    </div>
    <span style="font-size:11px;min-width:36px;text-align:right">${pct}%</span>
  </div>`
}

export function CPUSummaryPage() {
  const [data, setData]       = useState<CPUData | null>(null)
  const [loading, setLoading] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    try {
      setData(await api.getCPUData() as CPUData)
    } catch (e) {
      console.error('CPU summary fetch failed:', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    const id = setInterval(fetchData, 300_000)
    return () => clearInterval(id)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const hosts = Object.keys(data?.summary ?? {})

  // --- Logged-in users: flatten all nodes ---
  const loggedInRows = hosts.flatMap(host =>
    (data!.summary[host]?.logged_in ?? []).map(r => ({ node: host, ...r }))
  )

  // --- User processes: flatten all nodes ---
  const userProcRows = hosts.flatMap(host =>
    (data!.summary[host]?.user_procs ?? []).map(r => ({ node: host, ...r }))
  )

  // --- KFD processes: flatten all nodes ---
  const kfdProcRows = hosts.flatMap(host =>
    (data!.summary[host]?.kfd_procs ?? []).map(r => ({ node: host, ...r }))
  )

  const procCols = [
    { title: 'Node',  data: 'node',  className: 'dt-left font-medium text-xs' },
    { title: 'PID',   data: 'pid',   className: 'dt-right font-mono text-xs' },
    { title: 'User',  data: 'user',  className: 'dt-left text-xs' },
    { title: '%CPU',  data: 'cpu',   className: 'dt-right text-xs' },
    { title: '%MEM',  data: 'mem',   className: 'dt-right text-xs' },
    { title: 'VSZ',   data: 'vsz',   className: 'dt-right text-xs' },
    { title: 'RSS',   data: 'rss',   className: 'dt-right text-xs' },
    { title: 'STAT',  data: 'stat',  className: 'dt-center text-xs' },
    { title: 'Start', data: 'start', className: 'dt-center text-xs' },
    { title: 'Time',  data: 'time',  className: 'dt-right text-xs' },
    { title: 'Command / Args', data: 'args', className: 'dt-left font-mono text-xs' },
  ]

  const loginCols = [
    { title: 'Node', data: 'node', className: 'dt-left font-medium text-xs' },
    { title: 'User', data: 'user', className: 'dt-left text-xs' },
    { title: 'TTY',  data: 'tty',  className: 'dt-left font-mono text-xs' },
    { title: 'Date', data: 'date', className: 'dt-center text-xs' },
    { title: 'Time', data: 'time', className: 'dt-center text-xs' },
    { title: 'From', data: 'from', className: 'dt-left font-mono text-xs' },
  ]

  // --- CPU table: one row per node ---
  const cpuRows = hosts.map(host => {
    const lscpu = data!.summary[host]?.lscpu ?? {}
    const row: Record<string, string> = { node: host }
    LSCPU_COLS.forEach(({ key, title }) => {
      row[title] = lscpu[key] ?? '—'
    })
    return row
  })

  const cpuCols = [
    { title: 'Node', data: 'node', className: 'dt-left font-medium text-xs' },
    ...LSCPU_COLS.map(({ title }) => ({
      title,
      data: title,
      className: title === 'Model Name' ? 'dt-left text-xs' : 'dt-center text-xs',
    })),
  ]

  // --- Memory table: one row per node ---
  const memRows = hosts.map(host => {
    const ms = data!.summary[host]?.mem_summary
    const ls = data!.summary[host]?.lsmem?.summary ?? {}
    return {
      node:         host,
      total:        ms?.total          ?? '—',
      used:         ms?.used           ?? '—',
      used_pct:     ms?.used_pct       ?? 0,
      available:    ms?.available      ?? '—',
      cached:       ms?.cached         ?? '—',
      swap_total:   ms?.swap_total     ?? '—',
      swap_used:    ms?.swap_used      ?? '—',
      block_size:   ls['Memory block size'] ?? '—',
      online_mem:   ls['Total online memory']  ?? '—',
      offline_mem:  ls['Total offline memory'] ?? '—',
    }
  })

  const memCols = [
    { title: 'Node',          data: 'node',        className: 'dt-left font-medium text-xs' },
    { title: 'Total',         data: 'total',        className: 'dt-right text-xs' },
    {
      title: 'Used',
      data:  'used_pct',
      className: 'dt-left',
      render: (v: number, _t: string) => usedPctRender(v),
    },
    { title: 'Available',     data: 'available',    className: 'dt-right text-xs' },
    { title: 'Cached+Buf',    data: 'cached',       className: 'dt-right text-xs' },
    { title: 'Swap Total',    data: 'swap_total',   className: 'dt-right text-xs' },
    { title: 'Swap Used',     data: 'swap_used',    className: 'dt-right text-xs' },
    { title: 'Block Size',    data: 'block_size',   className: 'dt-center text-xs' },
    { title: 'Online Mem',    data: 'online_mem',   className: 'dt-right text-xs' },
    { title: 'Offline Mem',   data: 'offline_mem',  className: 'dt-right text-xs' },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">CPU Summary</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Hardware details for all {hosts.length} node(s) — lscpu &amp; lsmem
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
      {data && Object.keys(data.errors).length > 0 && (
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
            <p>No CPU data yet — collector runs every 60 seconds after the daemon connects.</p>
          </CardContent>
        </Card>
      )}

      {data && hosts.length > 0 && (
        <>
          {/* CPU Hardware */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                CPU Hardware — All Nodes
                <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                  {hosts.length} node(s) · lscpu
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CustomDataTable
                columns={cpuCols}
                data={cpuRows}
                defaultPageLength={50}
                pageLengthOptions={[50, 100, 500]}
                scrollX={true}
              />
            </CardContent>
          </Card>

          {/* Memory Hardware */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Memory Hardware — All Nodes
                <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                  {hosts.length} node(s) · /proc/meminfo + lsmem
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CustomDataTable
                columns={memCols}
                data={memRows}
                defaultPageLength={50}
                pageLengthOptions={[50, 100, 500]}
                scrollX={true}
              />
            </CardContent>
          </Card>

          {/* Logged-in Users */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Logged-in Users — All Nodes
                <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                  {loggedInRows.length} session(s) · who
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {loggedInRows.length === 0
                ? <p className="text-sm text-gray-400 italic py-4">No active login sessions.</p>
                : <CustomDataTable columns={loginCols} data={loggedInRows}
                    defaultPageLength={50} pageLengthOptions={[50, 100]} scrollX={true} />
              }
            </CardContent>
          </Card>

          {/* User Processes */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                User Processes — All Nodes
                <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                  {userProcRows.length} process(es) · ps -eo pid,user,pcpu,pmem,args
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {userProcRows.length === 0
                ? <p className="text-sm text-gray-400 italic py-4">No processes found.</p>
                : <CustomDataTable columns={procCols} data={userProcRows}
                    defaultPageLength={25} pageLengthOptions={[25, 50, 100]} scrollX={true} />
              }
            </CardContent>
          </Card>

          {/* KFD Processes */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                KFD Processes — All Nodes
                <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                  {kfdProcRows.length} process(es) · processes with /dev/kfd open (AMD GPU compute)
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {kfdProcRows.length === 0
                ? <p className="text-sm text-gray-400 italic py-4">
                    No KFD processes running (no GPU compute jobs active or /dev/kfd not present).
                  </p>
                : <CustomDataTable columns={procCols} data={kfdProcRows}
                    defaultPageLength={25} pageLengthOptions={[25, 50, 100]} scrollX={true} />
              }
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
