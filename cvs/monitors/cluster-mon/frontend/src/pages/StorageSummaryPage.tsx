/**
 * StorageSummaryPage — Storage → Summary
 * Block devices (lsblk), NVMe drives, and filesystem usage with charts.
 */
import { useState, useEffect } from 'react'
import { RefreshCw, HardDrive, Database } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { ExpandableChartCard } from '@/components/charts/ChartModal'
import { HorizontalStackedBar, VerticalBarChart, COLORS, shortName } from '@/components/charts/ClusterCharts'
// shortName is used only for chart labels, not for table node columns
import { TopNodesChart } from '@/components/charts/TopNodesChart'
import { api } from '@/services/api'

interface StorageData {
  block_devices: Record<string, any[]>
  filesystems:   Record<string, any[]>
  nvme_devices:  Record<string, any[]>
  errors:        Record<string, string>
  last_updated:  string | null
  state:         string
}

export function StorageSummaryPage() {
  const [data, setData]       = useState<StorageData | null>(null)
  const [loading, setLoading] = useState(false)

  const fetch = async () => {
    setLoading(true)
    try { setData(await api.getStorageData() as StorageData) }
    catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  useEffect(() => { fetch(); const id = setInterval(fetch, 300_000); return () => clearInterval(id) }, [])

  const hosts = Object.keys(data?.filesystems ?? {})

  // ── Filesystem usage — all nodes flat ──────────────────────────────────
  const fsRows = hosts.flatMap(h =>
    (data!.filesystems[h] ?? []).map(r => ({ node: h, ...r }))
  )
  const fsCols = fsRows.length === 0 ? [] : [
    { title: 'Node',        data: 'node',       className: 'dt-left font-medium text-xs' },
    { title: 'Source',      data: 'source',     className: 'dt-left font-mono text-xs' },
    { title: 'Type',        data: 'fstype',     className: 'dt-left text-xs' },
    { title: 'Size',        data: 'size',        className: 'dt-right text-xs' },
    { title: 'Used',        data: 'used',        className: 'dt-right text-xs' },
    { title: 'Avail',       data: 'avail',       className: 'dt-right text-xs' },
    {
      title: 'Use %',
      data: 'use_pct',
      className: 'dt-right text-xs',
      render: (v: number) =>
        v >= 90 ? `<span class="text-red-600 font-bold">${v}%</span>` :
        v >= 75 ? `<span class="text-amber-600 font-semibold">${v}%</span>` :
        `${v}%`,
    },
    { title: 'Mount',       data: 'mountpoint', className: 'dt-left font-mono text-xs' },
  ]

  // Chart: filesystem usage bars (top 20 most used)
  const fsChartData = fsRows
    .filter(r => r.mountpoint && r.use_pct !== undefined)
    .sort((a, b) => b.use_pct - a.use_pct)
    .slice(0, 20)
    .map(r => ({
      name: `${r.node}:${r.mountpoint}`,
      Used: r.use_pct,
      Free: 100 - r.use_pct,
      _color: r.use_pct >= 90 ? COLORS.red : r.use_pct >= 75 ? COLORS.amber : COLORS.blue,
    }))

  // ── Block devices — all nodes flat ──────────────────────────────────────
  const bdRows = hosts.flatMap(h =>
    (data!.block_devices[h] ?? []).map(r => ({ node: h, ...r }))
  )
  const bdCols = bdRows.length === 0 ? [] : [
    { title: 'Node',       data: 'node',       className: 'dt-left font-medium text-xs' },
    { title: 'Name',       data: 'name',       className: 'dt-left font-mono text-xs' },
    { title: 'Size',       data: 'size',       className: 'dt-right text-xs' },
    { title: 'Type',       data: 'type',       className: 'dt-center text-xs' },
    { title: 'FS Type',    data: 'fstype',     className: 'dt-left text-xs' },
    { title: 'Mount',      data: 'mountpoint', className: 'dt-left font-mono text-xs' },
    { title: 'Model',      data: 'model',      className: 'dt-left text-xs' },
    { title: 'Transport',  data: 'tran',       className: 'dt-center text-xs' },
    { title: 'Read-only',  data: 'ro',         className: 'dt-center text-xs',
      render: (v: string) => v === '1' ? '<span class="text-amber-600">RO</span>' : '—' },
  ]

  // ── NVMe devices ────────────────────────────────────────────────────────
  const nvmeRows = hosts.flatMap(h =>
    (data!.nvme_devices[h] ?? []).map(r => ({ node: h, ...r }))
  )
  const nvmeCols = nvmeRows.length === 0 ? [] : [
    { title: 'Node',     data: 'node',     className: 'dt-left font-medium text-xs' },
    { title: 'Device',   data: 'device',   className: 'dt-left font-mono text-xs' },
    { title: 'Model',    data: 'model',    className: 'dt-left text-xs' },
    { title: 'Size',     data: 'size',     className: 'dt-right text-xs' },
    { title: 'Firmware', data: 'firmware', className: 'dt-center text-xs' },
    { title: 'Serial',   data: 'serial',   className: 'dt-left font-mono text-xs' },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Storage Summary</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Block devices, NVMe drives and filesystem usage across all nodes
            {data?.last_updated && ` · ${new Date(data.last_updated).toLocaleString()}`}
          </p>
        </div>
        <button onClick={fetch} disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium">
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </div>

      {data && Object.keys(data.errors).length > 0 && (
        <div className="p-3 bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-700 rounded-lg">
          <p className="text-xs font-semibold text-yellow-800 dark:text-yellow-300 mb-1">Errors:</p>
          <ul className="text-xs text-yellow-700 dark:text-yellow-400 space-y-0.5">
            {Object.entries(data.errors).map(([h, e]) => <li key={h}><span className="font-mono">{h}</span>: {e}</li>)}
          </ul>
        </div>
      )}

      {!data && <Card><CardContent className="py-12 text-center text-gray-400">Collecting storage data... (runs every 5 min)</CardContent></Card>}

      {data && (
        <>
          {/* ── Top 10 nodes by highest filesystem usage % ── */}
          {(() => {
            // Pick the most-used filesystem per node as that node's "storage pressure"
            const nodeMax = hosts.map(h => {
              const fss = data!.filesystems[h] ?? []
              const maxPct = fss.reduce((m, r) => Math.max(m, r.use_pct ?? 0), 0)
              const worst  = fss.find(r => (r.use_pct ?? 0) === maxPct)
              return {
                node:  h,
                value: maxPct,
                label: `${maxPct}% (${worst?.mountpoint ?? ''})`,
              }
            }).filter(r => r.value > 0)
            return (
              <TopNodesChart
                data={nodeMax}
                title="Peak Filesystem Usage"
                unit="%"
                warnThreshold={75}
                critThreshold={90}
              />
            )
          })()}

          {/* Filesystem usage chart */}
          {fsChartData.length > 0 && (
            <ExpandableChartCard title="Filesystem Usage % (top 20 mounts)"
              small={<HorizontalStackedBar data={fsChartData} keys={['Used','Free']} colors={[COLORS.blue, COLORS.gray]} unit="%" maxValue={100} height={Math.max(160, fsChartData.length * 24 + 60)}/>}
              large={<HorizontalStackedBar data={fsChartData} keys={['Used','Free']} colors={[COLORS.blue, COLORS.gray]} unit="%" maxValue={100} height={Math.max(300, fsChartData.length * 36 + 80)}/>}
            />
          )}

          {/* Filesystem table */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Database className="h-4 w-4 text-blue-600" />
                Filesystem Usage — All Nodes
                <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">{fsRows.length} mount(s)</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {fsRows.length === 0
                ? <p className="text-sm text-gray-400 italic">No filesystem data.</p>
                : <CustomDataTable columns={fsCols} data={fsRows} defaultPageLength={50} pageLengthOptions={[50,100,500]} scrollX={true}/>}
            </CardContent>
          </Card>

          {/* Block devices table */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <HardDrive className="h-4 w-4 text-purple-600" />
                Block Devices — All Nodes
                <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">{bdRows.length} device(s)</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {bdRows.length === 0
                ? <p className="text-sm text-gray-400 italic">No block device data.</p>
                : <CustomDataTable columns={bdCols} data={bdRows} defaultPageLength={50} pageLengthOptions={[50,100,500]} scrollX={true}/>}
            </CardContent>
          </Card>

          {/* NVMe table */}
          {nvmeRows.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  NVMe Devices — All Nodes
                  <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">{nvmeRows.length} drive(s)</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CustomDataTable columns={nvmeCols} data={nvmeRows} defaultPageLength={25} pageLengthOptions={[25,50]} scrollX={true}/>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
