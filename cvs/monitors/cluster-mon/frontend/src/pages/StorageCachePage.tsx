/**
 * StorageCachePage — Storage → Cache & Memory
 * Page cache, dirty pages, vmstat IO counters — all nodes.
 */
import { useState, useEffect } from 'react'
import { RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { ExpandableChartCard } from '@/components/charts/ChartModal'
import { HorizontalStackedBar, COLORS } from '@/components/charts/ClusterCharts'
import { api } from '@/services/api'

interface StorageData {
  mem_cache:   Record<string, any>
  vm_stats:    Record<string, any>
  errors:      Record<string, string>
  last_updated: string | null
}

export function StorageCachePage() {
  const [data, setData]       = useState<StorageData | null>(null)
  const [loading, setLoading] = useState(false)

  const fetch = async () => {
    setLoading(true)
    try { setData(await api.getStorageData() as StorageData) }
    catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  useEffect(() => { fetch(); const id = setInterval(fetch, 300_000); return () => clearInterval(id) }, [])

  const hosts = Object.keys(data?.mem_cache ?? {})

  // ── Cache table: one row per node ──────────────────────────────────────
  const cacheRows = hosts.map(h => {
    const mc = data!.mem_cache[h] ?? {}
    const vm = data!.vm_stats[h]  ?? {}
    return {
      node:         h,
      page_cache:   mc.page_cache_mib ?? 0,
      cached:       mc.cached_mib ?? 0,
      buffers:      mc.buffers_mib ?? 0,
      dirty:        mc.dirty_mib ?? 0,
      writeback:    mc.writeback_mib ?? 0,
      slab:         mc.slab_mib ?? 0,
      pgmajfault:   vm.pgmajfault ?? 0,
      nr_dirty:     vm.nr_dirty ?? 0,
      swap_in:      vm.pswpin ?? 0,
      swap_out:     vm.pswpout ?? 0,
    }
  })

  const cacheCols = [
    { title: 'Node',          data: 'node',       className: 'dt-left font-medium text-xs' },
    { title: 'Page Cache MiB',data: 'page_cache', className: 'dt-right text-xs' },
    { title: 'Cached MiB',    data: 'cached',     className: 'dt-right text-xs' },
    { title: 'Buffers MiB',   data: 'buffers',    className: 'dt-right text-xs' },
    {
      title: 'Dirty MiB',
      data:  'dirty',
      className: 'dt-right text-xs',
      render: (v: number) => v > 500 ? `<span class="text-amber-600 font-semibold">${v}</span>` : String(v),
    },
    { title: 'Writeback MiB', data: 'writeback',  className: 'dt-right text-xs' },
    { title: 'Slab MiB',      data: 'slab',       className: 'dt-right text-xs' },
    {
      title: 'Major Faults',
      data:  'pgmajfault',
      className: 'dt-right text-xs',
      render: (v: number) => v > 1000 ? `<span class="text-red-600 font-bold">${v.toLocaleString()}</span>` : v.toLocaleString(),
    },
    { title: 'Dirty Pages',   data: 'nr_dirty',  className: 'dt-right text-xs' },
    { title: 'Swap In',       data: 'swap_in',   className: 'dt-right text-xs' },
    { title: 'Swap Out',      data: 'swap_out',  className: 'dt-right text-xs' },
  ]

  // Chart data — full hostname on Y-axis (HorizontalStackedBar handles width automatically)
  const cacheChartData = cacheRows.map(r => ({
    name:      r.node,
    Cached:    r.cached,
    Buffers:   r.buffers,
    Slab:      r.slab,
  }))

  const dirtyChartData = cacheRows.map(r => ({
    name:      r.node,
    Dirty:     r.dirty,
    Writeback: r.writeback,
  }))

  // Avg dirty MiB across nodes
  const avgDirty = cacheRows.length
    ? Math.round(cacheRows.reduce((s, r) => s + r.dirty, 0) / cacheRows.length)
    : 0

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Cache & Memory</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Page cache, dirty pages and VM counters — all nodes
            {data?.last_updated && ` · ${new Date(data.last_updated).toLocaleString()}`}
          </p>
        </div>
        <button onClick={fetch} disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium">
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </div>

      {!data && <Card><CardContent className="py-12 text-center text-gray-400">Collecting cache data... (runs every 5 min)</CardContent></Card>}

      {data && hosts.length > 0 && (
        <>
          {/* Summary stat cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Avg Page Cache', value: `${Math.round(cacheRows.reduce((s,r)=>s+r.page_cache,0)/(cacheRows.length||1))} MiB`, color: 'text-blue-600' },
              { label: 'Avg Dirty Pages', value: `${avgDirty} MiB`, color: avgDirty > 500 ? 'text-amber-600' : 'text-green-600' },
              { label: 'Total Swap Outs', value: cacheRows.reduce((s,r)=>s+r.swap_out,0).toLocaleString(), color: 'text-purple-600' },
              { label: 'Total Maj Faults', value: cacheRows.reduce((s,r)=>s+r.pgmajfault,0).toLocaleString(), color: 'text-orange-600' },
            ].map(({ label, value, color }) => (
              <Card key={label}>
                <CardContent className="pt-4 pb-4 text-center">
                  <div className={`text-xl font-bold ${color}`}>{value}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{label}</div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Cache breakdown chart — horizontal so full node names fit on Y-axis */}
          <ExpandableChartCard title="Page Cache Breakdown by Node (MiB)"
            small={<HorizontalStackedBar data={cacheChartData} keys={['Cached','Buffers','Slab']} colors={[COLORS.blue,COLORS.teal,COLORS.purple]} unit=" MiB" maxValue={Math.max(1,...cacheRows.map(r=>r.cached+r.buffers+r.slab))} height={Math.max(160,cacheRows.length*32+60)}/>}
            large={<HorizontalStackedBar data={cacheChartData} keys={['Cached','Buffers','Slab']} colors={[COLORS.blue,COLORS.teal,COLORS.purple]} unit=" MiB" maxValue={Math.max(1,...cacheRows.map(r=>r.cached+r.buffers+r.slab))} height={Math.max(250,cacheRows.length*42+80)}/>}
          />

          {/* Dirty pages horizontal bar */}
          <ExpandableChartCard title="Dirty vs Writeback Pages by Node (MiB)"
            small={<HorizontalStackedBar data={dirtyChartData} keys={['Dirty','Writeback']} colors={[COLORS.amber,COLORS.orange]} unit=" MiB" maxValue={Math.max(1,...cacheRows.map(r=>r.dirty+r.writeback))} height={Math.max(160,cacheRows.length*32+60)}/>}
            large={<HorizontalStackedBar data={dirtyChartData} keys={['Dirty','Writeback']} colors={[COLORS.amber,COLORS.orange]} unit=" MiB" maxValue={Math.max(1,...cacheRows.map(r=>r.dirty+r.writeback))} height={Math.max(250,cacheRows.length*42+80)}/>}
          />

          {/* Full cache table */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Cache & VM Stats — All Nodes
                <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">/proc/meminfo + /proc/vmstat</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <CustomDataTable columns={cacheCols} data={cacheRows} defaultPageLength={25} pageLengthOptions={[25,50,100]} scrollX={true}/>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
