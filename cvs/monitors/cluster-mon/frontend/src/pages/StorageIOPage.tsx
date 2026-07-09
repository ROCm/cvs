/**
 * StorageIOPage — Storage → IO Metrics
 * iostat per-disk + top IO processes + IO chart.
 */
import { useState, useEffect } from 'react'
import { RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { ExpandableChartCard } from '@/components/charts/ChartModal'
import { GroupedBarChart, VerticalBarChart, COLORS, shortName, compactNum } from '@/components/charts/ClusterCharts'
import { TopNodesChart } from '@/components/charts/TopNodesChart'
import { api } from '@/services/api'

interface StorageData {
  io_stats:      Record<string, any[]>
  disk_stats:    Record<string, any[]>
  top_io_procs:  Record<string, any[]>
  errors:        Record<string, string>
  last_updated:  string | null
}

export function StorageIOPage() {
  const [data, setData]       = useState<StorageData | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedHost, setSelectedHost] = useState('')

  const fetch = async () => {
    setLoading(true)
    try {
      const resp = await api.getStorageData() as StorageData
      setData(resp)
      const hs = Object.keys(resp.io_stats ?? {})
      if (hs.length && !selectedHost) setSelectedHost(hs[0])
    } catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  useEffect(() => { fetch(); const id = setInterval(fetch, 300_000); return () => clearInterval(id) }, [])

  const hosts = Object.keys(data?.io_stats ?? {})

  // IO stats for selected host
  const ioRows = (data?.io_stats?.[selectedHost] ?? []).filter(r => r.device && r.device !== 'device')
  const ioCols = ioRows.length === 0 ? [] : [
    { title: 'Device',     data: 'device',  className: 'dt-left font-mono text-xs' },
    { title: 'r/s',        data: 'r_s',     className: 'dt-right text-xs' },
    { title: 'w/s',        data: 'w_s',     className: 'dt-right text-xs' },
    { title: 'rkB/s',      data: 'rk_bs',   className: 'dt-right text-xs' },
    { title: 'wkB/s',      data: 'wk_bs',   className: 'dt-right text-xs' },
    { title: 'r_await ms', data: 'r_await', className: 'dt-right text-xs' },
    { title: 'w_await ms', data: 'w_await', className: 'dt-right text-xs' },
    { title: 'aqu-sz',     data: 'aqu_sz',  className: 'dt-right text-xs' },
    {
      title: '%util',
      data: 'pct_util',
      className: 'dt-right text-xs',
      render: (v: number) => v > 80 ? `<span class="text-red-600 font-bold">${v}%</span>` :
                             v > 50 ? `<span class="text-amber-600">${v}%</span>` : `${v}%`,
    },
  ]

  // IO chart: read KB/s vs write KB/s
  const ioChartData = ioRows.map(r => ({
    name:  r.device,
    'Read KB/s':  Number(r.rk_bs) || 0,
    'Write KB/s': Number(r.wk_bs) || 0,
  })).filter(r => r['Read KB/s'] + r['Write KB/s'] > 0)

  // Top IO processes — all nodes combined
  const procRows = hosts.flatMap(h =>
    (data!.top_io_procs?.[h] ?? []).map(r => ({ node: h, ...r }))
  ).sort((a, b) => b.total_bytes - a.total_bytes).slice(0, 30)

  const procCols = procRows.length === 0 ? [] : [
    { title: 'Node',      data: 'node',     className: 'dt-left font-medium text-xs' },
    { title: 'PID',       data: 'pid',      className: 'dt-right font-mono text-xs' },
    { title: 'Command',   data: 'command',  className: 'dt-left font-mono text-xs' },
    { title: 'Read MB',   data: 'read_mb',  className: 'dt-right text-xs' },
    { title: 'Write MB',  data: 'write_mb', className: 'dt-right text-xs' },
    { title: 'Total MB',  data: 'total_mb', className: 'dt-right font-semibold text-xs' },
  ]

  const procChartData = procRows.slice(0, 15).map(r => ({
    name: `${r.command}(${r.pid})`,
    'Total MB': r.total_mb,
    _color: r.total_mb > 1000 ? COLORS.red : r.total_mb > 100 ? COLORS.amber : COLORS.blue,
  }))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Storage IO Metrics</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Disk IO throughput and top IO processes
            {data?.last_updated && ` · ${new Date(data.last_updated).toLocaleString()}`}
          </p>
        </div>
        <button onClick={fetch} disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium">
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </div>

      {!data && <Card><CardContent className="py-12 text-center text-gray-400">Collecting IO data... (runs every 5 min)</CardContent></Card>}

      {data && (
        <>
          {/* ── Top 10 nodes by total disk KB written (lifetime) ── */}
          {(() => {
            const nodeIO = hosts.map(h => {
              const disks = data!.disk_stats?.[h] ?? []
              const totalKB = disks.reduce((s: number, d: any) => s + (d.total_kb ?? 0), 0)
              return { node: h, value: Math.round(totalKB / 1024), label: `${Math.round(totalKB/1024)} MB` }
            }).filter(r => r.value > 0)
            return <TopNodesChart data={nodeIO} title="Total Block Layer IO (cumulative)" unit=" MB" />
          })()}

          {/* Host selector + IO chart */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="flex items-center gap-3 lg:col-span-1">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Node:</label>
              <select value={selectedHost} onChange={e => setSelectedHost(e.target.value)}
                className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded text-sm bg-white dark:bg-gray-800 dark:text-gray-200">
                {hosts.map(h => <option key={h} value={h}>{h}</option>)}
              </select>
            </div>
            {ioChartData.length > 0 && (
              <div className="lg:col-span-3">
                <ExpandableChartCard title={`Disk IO Throughput — ${selectedHost}`}
                  small={<GroupedBarChart data={ioChartData} keys={['Read KB/s','Write KB/s']} colors={[COLORS.blue,COLORS.orange]} height={200}/>}
                  large={<GroupedBarChart data={ioChartData} keys={['Read KB/s','Write KB/s']} colors={[COLORS.blue,COLORS.orange]} height={460}/>}
                />
              </div>
            )}
          </div>

          {/* iostat table */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Disk IO Statistics — {selectedHost}
                <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">iostat -x</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {ioRows.length === 0
                ? <p className="text-sm text-gray-400 italic py-4">No IO data for this node.</p>
                : <CustomDataTable columns={ioCols} data={ioRows} defaultPageLength={25} pageLengthOptions={[25,50]} scrollX={true}/>}
            </CardContent>
          </Card>

          {/* Block layer stats (/proc/diskstats) */}
          {(() => {
            const dsRows = (data!.disk_stats?.[selectedHost] ?? []).map((r: any) => ({ ...r }))
            if (!dsRows.length) return null
            const dsCols = [
              { title: 'Device',        data: 'device',        className: 'dt-left font-mono text-xs' },
              { title: 'Reads',         data: 'reads',         className: 'dt-right text-xs' },
              { title: 'Read KB',       data: 'read_kb',       className: 'dt-right text-xs' },
              { title: 'Avg Read ms',   data: 'read_avg_ms',   className: 'dt-right text-xs' },
              { title: 'Writes',        data: 'writes',        className: 'dt-right text-xs' },
              { title: 'Write KB',      data: 'write_kb',      className: 'dt-right text-xs' },
              { title: 'Avg Write ms',  data: 'write_avg_ms',  className: 'dt-right text-xs' },
              { title: 'IO In-Prog',    data: 'io_in_progress',className: 'dt-right text-xs' },
              { title: 'Total KB',      data: 'total_kb',      className: 'dt-right font-semibold text-xs' },
            ]
            return (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    Block Layer Statistics — {selectedHost}
                    <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">/proc/diskstats (cumulative since boot)</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <CustomDataTable columns={dsCols} data={dsRows} defaultPageLength={25} pageLengthOptions={[25,50]} scrollX={true}/>
                </CardContent>
              </Card>
            )
          })()}

          {/* Top IO processes chart + table */}
          {procRows.length > 0 && (
            <>
              <ExpandableChartCard title="Top IO Processes by Total Bytes — All Nodes"
                small={<VerticalBarChart data={procChartData} dataKey="Total MB" color={COLORS.blue} height={220} unit=" MB"/>}
                large={<VerticalBarChart data={procChartData} dataKey="Total MB" color={COLORS.blue} height={460} unit=" MB"/>}
              />
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    Top IO Processes — All Nodes
                    <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">cumulative IO since process start</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <CustomDataTable columns={procCols} data={procRows} defaultPageLength={25} pageLengthOptions={[25,50,100]} scrollX={true}/>
                </CardContent>
              </Card>
            </>
          )}
        </>
      )}
    </div>
  )
}
