/**
 * ScaleUpLinksPage — Networks → Scale-up → Links
 *
 * IFoE port topology map: compute tray ↔ switch ASIC port mapping.
 */

import { RefreshCw, Download } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { useIFoEData, TopologyRow } from '@/hooks/useIFoEData'
import { DonutChart, HorizontalStackedBar, COLORS } from '@/components/charts/ClusterCharts'
import { ExpandableChartCard } from '@/components/charts/ChartModal'

function linkBadge(status: string): string {
  const up   = /link_up/i.test(status)
  const down = /no_phy|no_block|down/i.test(status)
  const cls  = up   ? 'text-green-600 font-semibold' :
               down ? 'text-red-600 font-semibold'   : 'text-gray-500'
  return `<span class="${cls}">${status || '—'}</span>`
}

function TopologySection({ rows }: { rows: TopologyRow[] }) {
  const exportCSV = () => {
    const cols: (keyof TopologyRow)[] = [
      'compute_tray','gpu_index','station_index','port_index',
      'ifoe_interface','compute_mac','link_status','speed',
      'switch_tray','asic','switch_port','mapped',
    ]
    const blob = new Blob(
      [cols.join(',') + '\n' + rows.map(r => cols.map(c => `"${r[c]}"`).join(',')).join('\n')],
      { type: 'text/csv' }
    )
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'ifoe_topology.csv'
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const mappedCount = rows.filter(r => r.mapped).length
  const upCount     = rows.filter(r => /link_up/i.test(r.link_status)).length

  const tableRows = rows.map(r => ({
    compute_tray:   r.compute_tray,
    gpu_index:      r.gpu_index,
    station_index:  r.station_index,
    port_index:     r.port_index,
    ifoe_interface: r.ifoe_interface,
    compute_mac:    r.compute_mac,
    link_status:    r.link_status,
    speed:          r.speed,
    switch_tray:    r.switch_tray || '—',
    asic:           r.asic || '—',
    switch_port:    r.switch_port || '—',
    mapped:         r.mapped ? '✓' : '✗',
  }))

  const columns = [
    { title: 'Compute Tray',  data: 'compute_tray',   className: 'dt-left font-medium text-xs' },
    { title: 'GPU',           data: 'gpu_index',       className: 'dt-center text-xs' },
    { title: 'Station',       data: 'station_index',   className: 'dt-center text-xs' },
    { title: 'Port',          data: 'port_index',      className: 'dt-center text-xs' },
    { title: 'Interface',     data: 'ifoe_interface',  className: 'dt-left font-mono text-xs' },
    { title: 'MAC',           data: 'compute_mac',     className: 'dt-left font-mono text-xs' },
    { title: 'Link Status',   data: 'link_status',     className: 'dt-left text-xs',
      render: (val: string) => linkBadge(val) },
    { title: 'Speed',         data: 'speed',           className: 'dt-left text-xs' },
    { title: 'Switch Tray',   data: 'switch_tray',     className: 'dt-left font-mono text-xs' },
    { title: 'ASIC',          data: 'asic',            className: 'dt-center text-xs' },
    { title: 'Switch Port',   data: 'switch_port',     className: 'dt-left font-mono text-xs' },
    { title: 'Mapped',        data: 'mapped',          className: 'dt-center text-sm',
      render: (val: string) =>
        val === '✓' ? '<span class="text-green-500 text-base">✓</span>'
                    : '<span class="text-red-400 text-base">✗</span>' },
  ]

  return (
    <div className="space-y-4">
      {/* Summary stats */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Total Ports',   value: rows.length },
          { label: 'Link UP',       value: upCount,               color: 'text-green-600' },
          { label: 'Link DOWN',     value: rows.length - upCount, color: rows.length - upCount > 0 ? 'text-red-600' : '' },
          { label: 'Switch Mapped', value: `${mappedCount}/${rows.length}` },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 text-center">
            <div className={`text-xl font-bold ${(color as string) || 'text-gray-900 dark:text-gray-100'}`}>{value}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">{label}</div>
          </div>
        ))}
      </div>

      <div className="flex justify-end">
        <button onClick={exportCSV}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-800 dark:text-gray-200">
          <Download className="h-4 w-4" /> Export CSV
        </button>
      </div>

      {rows.length === 0 ? (
        <p className="text-sm text-gray-400 italic py-4">
          No topology data. Ensure both compute trays and switch trays are configured and reachable.
        </p>
      ) : (
        <CustomDataTable columns={columns} data={tableRows}
          defaultPageLength={25} pageLengthOptions={[25, 50, 100]} scrollX={true} />
      )}
    </div>
  )
}

export function ScaleUpLinksPage() {
  const { data, refreshing, triggerRefresh } = useIFoEData()

  const topology = data?.topology ?? []

  // Filter errors to switch hosts only
  const switchHostSet = new Set(Object.keys(data?.switch_vlan ?? {}))
  const switchErrors  = Object.fromEntries(
    Object.entries(data?.errors ?? {}).filter(([h]) => switchHostSet.has(h))
  )

  // Per-switch mapped/unmapped counts for chart
  const swCounts = (() => {
    const c: Record<string, {mapped: number; unmapped: number}> = {}
    topology.forEach(r => {
      const sw = r.switch_tray || 'Unknown'
      if (!c[sw]) c[sw] = {mapped: 0, unmapped: 0}
      r.mapped ? c[sw].mapped++ : c[sw].unmapped++
    })
    return Object.entries(c).map(([name, v]) => ({name, Mapped: v.mapped, Unmapped: v.unmapped}))
  })()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Scale-up Fabric — Links</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            IFoE compute↔switch port topology map
            {data?.last_updated && ` · ${new Date(data.last_updated).toLocaleString()}`}
          </p>
        </div>
        <button onClick={triggerRefresh} disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium">
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh Now
        </button>
      </div>

      {data && Object.keys(switchErrors).length > 0 && Object.values(switchErrors).every(e => String(e).startsWith('ABORT')) && (
        <div className="p-4 bg-red-50 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded-lg">
          <p className="text-sm font-semibold text-red-800 dark:text-red-300 mb-1">Switch SSH credentials not available</p>
          <p className="text-xs text-red-700 dark:text-red-400">
            Go to <strong>Configuration → Scale-up Switches</strong>, enter credentials, and click <strong>Save & Apply</strong>.
          </p>
        </div>
      )}

      {!data && (
        <Card><CardContent className="py-12 text-center text-gray-400">
          No IFoE data. Configure switch trays in Configuration and click Save & Apply.
        </CardContent></Card>
      )}

      {data && (
        <>
          {/* Visual overview charts */}
          {topology.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ExpandableChartCard title="Port Mapping Status"
                small={<div className="flex justify-center"><DonutChart title="" data={[
                  {name:'Mapped',   value: topology.filter(r => r.mapped).length,  color: COLORS.green},
                  {name:'Unmapped', value: topology.filter(r => !r.mapped).length, color: COLORS.gray},
                ]} centerLabel={`${topology.filter(r=>r.mapped).length}/${topology.length}`}
                   centerSub="mapped" size={170}/></div>}
                large={<div className="flex justify-center pt-8"><DonutChart title="" data={[
                  {name:'Mapped',   value: topology.filter(r => r.mapped).length,  color: COLORS.green},
                  {name:'Unmapped', value: topology.filter(r => !r.mapped).length, color: COLORS.gray},
                ]} centerLabel={`${topology.filter(r=>r.mapped).length}/${topology.length}`}
                   centerSub="mapped" size={340}/></div>}
              />
              {swCounts.length > 0 && (
                <ExpandableChartCard title="Mapped Ports by Switch Tray"
                  small={<HorizontalStackedBar data={swCounts} keys={['Mapped','Unmapped']}
                    colors={[COLORS.green, COLORS.gray]} unit=" ports"
                    maxValue={Math.max(1,...swCounts.map(r=>r.Mapped+r.Unmapped))}
                    height={Math.max(120, swCounts.length*32+60)}/>}
                  large={<HorizontalStackedBar data={swCounts} keys={['Mapped','Unmapped']}
                    colors={[COLORS.green, COLORS.gray]} unit=" ports"
                    maxValue={Math.max(1,...swCounts.map(r=>r.Mapped+r.Unmapped))}
                    height={Math.max(200, swCounts.length*42+80)}/>}
                />
              )}
            </div>
          )}

          {/* Topology table */}
          <Card>
            <CardHeader>
              <CardTitle>
                Compute ↔ Switch Port Topology
                <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                  MAC-address correlation: IFoE endpoint → switch ASIC port
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <TopologySection rows={topology} />
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
