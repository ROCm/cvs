/**
 * ScaleUpNetworkPage
 *
 * IFoE fabric / switch tray data:
 *   - SONiC switch VLAN configuration
 *   - SONiC switch MAC tables (per ASIC)
 *   - Compute ↔ Switch topology map
 *
 * Networks → Backend → Scale-up → Overview.
 */

import { useState } from 'react'
import { RefreshCw, CheckCircle, XCircle, Download } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { useIFoEData, TopologyRow } from '@/hooks/useIFoEData'
import { DonutChart, GroupedBarChart, COLORS, shortName } from '@/components/charts/ClusterCharts'
import { ExpandableChartCard } from '@/components/charts/ChartModal'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function linkBadge(status: string): string {
  const up   = /link_up/i.test(status)
  const down = /no_phy|no_block|down/i.test(status)
  const cls  = up   ? 'text-green-600 font-semibold' :
               down ? 'text-red-600 font-semibold'   : 'text-gray-500'
  return `<span class="${cls}">${status || '—'}</span>`
}

function dynamicCols(rows: Record<string, string>[], monoKeys: string[] = []) {
  if (!rows.length) return []
  return Object.keys(rows[0]).map(k => ({
    title: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    data:  k,
    className: monoKeys.includes(k) ? 'dt-left font-mono text-xs' : 'dt-left text-xs',
  }))
}

// ---------------------------------------------------------------------------
// Topology section
// ---------------------------------------------------------------------------

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

  // Flatten topology rows for DataTable (mapped as 0/1 icon)
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
    _mapped_raw:    r.mapped,
  }))

  const columns = [
    { title: 'Compute Tray',   data: 'compute_tray',   className: 'dt-left font-medium text-xs' },
    { title: 'GPU',            data: 'gpu_index',       className: 'dt-center text-xs' },
    { title: 'Station',        data: 'station_index',   className: 'dt-center text-xs' },
    { title: 'Port',           data: 'port_index',      className: 'dt-center text-xs' },
    { title: 'Interface',      data: 'ifoe_interface',  className: 'dt-left font-mono text-xs' },
    { title: 'MAC',            data: 'compute_mac',     className: 'dt-left font-mono text-xs' },
    {
      title: 'Link Status',
      data:  'link_status',
      className: 'dt-left text-xs',
      render: (val: string) => linkBadge(val),
    },
    { title: 'Speed',          data: 'speed',           className: 'dt-left text-xs' },
    { title: 'Switch Tray',    data: 'switch_tray',     className: 'dt-left font-mono text-xs' },
    { title: 'ASIC',           data: 'asic',            className: 'dt-center text-xs' },
    { title: 'Switch Port',    data: 'switch_port',     className: 'dt-left font-mono text-xs' },
    {
      title: 'Mapped',
      data:  'mapped',
      className: 'dt-center text-sm',
      render: (val: string) =>
        val === '✓'
          ? '<span class="text-green-500 text-base">✓</span>'
          : '<span class="text-red-400 text-base">✗</span>',
    },
  ]

  return (
    <div className="space-y-3">
      {/* Summary stats */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Total Ports',   value: rows.length },
          { label: 'Link UP',       value: upCount,        color: 'text-green-600 dark:text-green-400' },
          { label: 'Link DOWN',     value: rows.length - upCount, color: rows.length - upCount > 0 ? 'text-red-600 dark:text-red-400' : '' },
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
        <CustomDataTable
          columns={columns}
          data={tableRows}
          defaultPageLength={25}
          pageLengthOptions={[25, 50, 100]}
        />
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

type Tab = 'topology' | 'switch'

export function ScaleUpNetworkPage() {
  const { data, refreshing, triggerRefresh } = useIFoEData()
  const [activeTab, setActiveTab]     = useState<Tab>('topology')
  const [selectedSwitchTray, setSwitchTray] = useState('')
  const [selectedAsic, setAsic]       = useState<'asic0' | 'asic1'>('asic0')

  const switchTrays     = Object.keys(data?.switch_vlan ?? {})
  const activeSwitchTray = selectedSwitchTray || switchTrays[0] || ''

  const vlanRows = (data?.switch_vlan?.[activeSwitchTray]?.[selectedAsic] ?? []) as Record<string, string>[]
  const macRows  = (data?.switch_mac?.[activeSwitchTray]?.[selectedAsic]  ?? []) as Record<string, string>[]

  const vlanCols = dynamicCols(vlanRows, ['ports'])
  const macCols  = dynamicCols(macRows, ['mac_address', 'port'])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Scale-up Fabric — Overview
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            IFoE switch tray configuration and compute↔switch port topology
            {data?.last_updated && ` · ${new Date(data.last_updated).toLocaleString()}`}
          </p>
        </div>
        <button onClick={triggerRefresh} disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium">
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh Now
        </button>
      </div>

      {/* Errors */}
      {data && Object.keys(data.errors).length > 0 && (
        <div className="p-4 bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-700 rounded-lg">
          <p className="text-sm font-semibold text-yellow-800 dark:text-yellow-300 mb-1">Collection errors:</p>
          <ul className="text-xs text-yellow-700 dark:text-yellow-400 space-y-0.5">
            {Object.entries(data.errors).map(([host, err]) => (
              <li key={host}><span className="font-mono">{host}</span>: {String(err)}</li>
            ))}
          </ul>
        </div>
      )}

      {!data && (
        <Card>
          <CardContent className="py-12 text-center text-gray-400">
            <p className="text-lg">No IFoE data available.</p>
            <p className="text-sm mt-1">Configure switch trays in Configuration and click Save & Apply.</p>
          </CardContent>
        </Card>
      )}

      {data && (
        <>
          {/* ── Visual Overview ── */}
          {data.topology.length > 0 && (() => {
            const swCounts = (() => {
              const c: Record<string, {mapped:number;unmapped:number}> = {}
              data.topology.forEach(r => {
                const sw = shortName(r.switch_tray || 'Unknown')
                if (!c[sw]) c[sw] = {mapped:0,unmapped:0}
                r.mapped ? c[sw].mapped++ : c[sw].unmapped++
              })
              return Object.entries(c).map(([name,v])=>({name,Mapped:v.mapped,Unmapped:v.unmapped}))
            })()
            return (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <ExpandableChartCard title="Port Mapping Status"
                  small={<div className="flex justify-center"><DonutChart title="" data={[{name:'Mapped',value:data.topology.filter(r=>r.mapped).length,color:COLORS.green},{name:'Unmapped',value:data.topology.filter(r=>!r.mapped).length,color:COLORS.gray}]} centerLabel={`${data.topology.filter(r=>r.mapped).length}/${data.topology.length}`} centerSub="mapped" size={170}/></div>}
                  large={<div className="flex justify-center pt-8"><DonutChart title="" data={[{name:'Mapped',value:data.topology.filter(r=>r.mapped).length,color:COLORS.green},{name:'Unmapped',value:data.topology.filter(r=>!r.mapped).length,color:COLORS.gray}]} centerLabel={`${data.topology.filter(r=>r.mapped).length}/${data.topology.length}`} centerSub="mapped" size={340}/></div>}
                />
                <ExpandableChartCard title="IFoE Link Status"
                  small={<div className="flex justify-center"><DonutChart title="" data={[{name:'Link UP',value:data.topology.filter(r=>/link_up/i.test(r.link_status)).length,color:COLORS.green},{name:'Link DOWN',value:data.topology.filter(r=>!/link_up/i.test(r.link_status)).length,color:COLORS.red}]} centerLabel={`${data.topology.filter(r=>/link_up/i.test(r.link_status)).length}`} centerSub="links UP" size={170}/></div>}
                  large={<div className="flex justify-center pt-8"><DonutChart title="" data={[{name:'Link UP',value:data.topology.filter(r=>/link_up/i.test(r.link_status)).length,color:COLORS.green},{name:'Link DOWN',value:data.topology.filter(r=>!/link_up/i.test(r.link_status)).length,color:COLORS.red}]} centerLabel={`${data.topology.filter(r=>/link_up/i.test(r.link_status)).length}`} centerSub="links UP" size={340}/></div>}
                />
                <ExpandableChartCard title="Mapped Ports by Switch Tray"
                  small={<GroupedBarChart data={swCounts} keys={['Mapped','Unmapped']} colors={[COLORS.green,COLORS.gray]} height={180}/>}
                  large={<GroupedBarChart data={swCounts} keys={['Mapped','Unmapped']} colors={[COLORS.green,COLORS.gray]} height={450}/>}
                />
              </div>
            )
          })()}

          {/* Tabs */}
          <div className="flex gap-1 border-b border-gray-200 dark:border-gray-700">
            {([
              { key: 'topology', label: 'Topology Map' },
              { key: 'switch',   label: 'Switch Trays (SONiC)' },
            ] as { key: Tab; label: string }[]).map(({ key, label }) => (
              <button key={key} onClick={() => setActiveTab(key)}
                className={`px-5 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                  activeTab === key
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                }`}>
                {label}
              </button>
            ))}
          </div>

          {/* Topology Map */}
          {activeTab === 'topology' && (
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
                <TopologySection rows={data.topology} />
              </CardContent>
            </Card>
          )}

          {/* Switch Trays */}
          {activeTab === 'switch' && (
            <div className="space-y-6">
              {/* Selectors */}
              <div className="flex items-center gap-4 flex-wrap">
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Switch Tray:</label>
                  <select value={activeSwitchTray} onChange={e => setSwitchTray(e.target.value)}
                    className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded text-sm bg-white dark:bg-gray-800 dark:text-gray-200">
                    {switchTrays.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-gray-700 dark:text-gray-300">ASIC:</label>
                  <select value={selectedAsic} onChange={e => setAsic(e.target.value as 'asic0' | 'asic1')}
                    className="px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded text-sm bg-white dark:bg-gray-800 dark:text-gray-200">
                    <option value="asic0">asic0</option>
                    <option value="asic1">asic1</option>
                  </select>
                </div>
              </div>

              {/* VLAN Brief */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    VLAN Configuration
                    <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                      show vlan brief · {activeSwitchTray} / {selectedAsic}
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {vlanRows.length === 0 ? (
                    <p className="text-sm text-gray-400 italic py-4">No VLAN data.</p>
                  ) : (
                    <CustomDataTable
                      columns={vlanCols}
                      data={vlanRows}
                      defaultPageLength={25}
                      pageLengthOptions={[25, 50, 100]}
                    />
                  )}
                </CardContent>
              </Card>

              {/* MAC Table */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    MAC Table
                    <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                      show mac · {activeSwitchTray} / {selectedAsic}
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {macRows.length === 0 ? (
                    <p className="text-sm text-gray-400 italic py-4">No MAC data.</p>
                  ) : (
                    <CustomDataTable
                      columns={macCols}
                      data={macRows}
                      defaultPageLength={25}
                      pageLengthOptions={[25, 50, 100]}
                    />
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  )
}
