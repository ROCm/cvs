import { useState, useEffect, useCallback } from 'react'
import { RefreshCw, CheckCircle, XCircle, Download } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { api } from '@/services/api'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface IFoEData {
  compute_devices: Record<string, Record<string, string>[]>
  compute_ports: Record<string, Record<string, string>[]>
  switch_vlan: Record<string, Record<string, Record<string, string>[]>>
  switch_mac: Record<string, Record<string, Record<string, string>[]>>
  topology: TopologyRow[]
  last_updated: string | null
  errors: Record<string, string>
  state: string
}

interface TopologyRow {
  compute_tray: string
  gpu_index: string
  station_index: string
  port_index: string
  ifoe_interface: string
  compute_mac: string
  link_status: string
  speed: string
  switch_tray: string
  asic: string
  switch_port: string
  mapped: boolean
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function linkStatusBadge(status: string) {
  const up = /up/i.test(status)
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${
        up ? 'bg-green-100 text-green-800' : status ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-500'
      }`}
    >
      {status || '—'}
    </span>
  )
}

function GenericTable({ rows }: { rows: Record<string, string>[] }) {
  if (!rows || rows.length === 0) {
    return <p className="text-sm text-gray-400 italic">No data</p>
  }
  const cols = Object.keys(rows[0])
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-xs border-collapse">
        <thead>
          <tr className="bg-gray-50">
            {cols.map((c) => (
              <th key={c} className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200 whitespace-nowrap">
                {c.replace(/_/g, ' ').toUpperCase()}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
              {cols.map((c) => (
                <td key={c} className="px-3 py-1.5 border-b border-gray-100 whitespace-nowrap">
                  {c.includes('status') || c.includes('state')
                    ? linkStatusBadge(row[c])
                    : row[c] || '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Topology tab
// ---------------------------------------------------------------------------

function TopologyTable({ rows }: { rows: TopologyRow[] }) {
  const [filterCompute, setFilterCompute] = useState('')
  const [filterSwitch, setFilterSwitch] = useState('')
  const [filterStatus, setFilterStatus] = useState('')
  const [filterMapped, setFilterMapped] = useState<'all' | 'mapped' | 'unmapped'>('all')

  const computeTrays = [...new Set(rows.map((r) => r.compute_tray))].filter(Boolean)
  const switchTrays = [...new Set(rows.map((r) => r.switch_tray))].filter(Boolean)

  const filtered = rows.filter((r) => {
    if (filterCompute && r.compute_tray !== filterCompute) return false
    if (filterSwitch && r.switch_tray !== filterSwitch) return false
    if (filterStatus && !r.link_status.toLowerCase().includes(filterStatus.toLowerCase())) return false
    if (filterMapped === 'mapped' && !r.mapped) return false
    if (filterMapped === 'unmapped' && r.mapped) return false
    return true
  })

  const exportCSV = () => {
    const cols: (keyof TopologyRow)[] = [
      'compute_tray', 'gpu_index', 'station_index', 'port_index',
      'ifoe_interface', 'compute_mac', 'link_status', 'speed',
      'switch_tray', 'asic', 'switch_port', 'mapped',
    ]
    const header = cols.join(',')
    const csvRows = filtered.map((r) => cols.map((c) => `"${r[c]}"`).join(','))
    const blob = new Blob([header + '\n' + csvRows.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'ifoe_topology.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <select
          className="px-3 py-1.5 border border-gray-300 rounded text-sm"
          value={filterCompute}
          onChange={(e) => setFilterCompute(e.target.value)}
        >
          <option value="">All Compute Trays</option>
          {computeTrays.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>

        <select
          className="px-3 py-1.5 border border-gray-300 rounded text-sm"
          value={filterSwitch}
          onChange={(e) => setFilterSwitch(e.target.value)}
        >
          <option value="">All Switch Trays</option>
          {switchTrays.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>

        <input
          type="text"
          placeholder="Filter by link status..."
          className="px-3 py-1.5 border border-gray-300 rounded text-sm"
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
        />

        <select
          className="px-3 py-1.5 border border-gray-300 rounded text-sm"
          value={filterMapped}
          onChange={(e) => setFilterMapped(e.target.value as any)}
        >
          <option value="all">All Ports</option>
          <option value="mapped">Mapped Only</option>
          <option value="unmapped">Unmapped Only</option>
        </select>

        <button
          onClick={exportCSV}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm border border-gray-300 rounded hover:bg-gray-50"
        >
          <Download className="h-4 w-4" />
          Export CSV
        </button>

        <span className="text-sm text-gray-500">{filtered.length} rows</span>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full text-xs border-collapse">
          <thead>
            <tr className="bg-gray-50">
              {['Compute Tray', 'GPU', 'Station', 'Port', 'Interface', 'MAC', 'Link Status', 'Speed',
                'Switch Tray', 'ASIC', 'Switch Port', 'Mapped?'].map((h) => (
                <th key={h} className="px-3 py-2 text-left font-semibold text-gray-600 border-b border-gray-200 whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={12} className="px-3 py-4 text-center text-gray-400 italic">
                  No rows match the current filters
                </td>
              </tr>
            ) : (
              filtered.map((row, i) => (
                <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-3 py-1.5 border-b border-gray-100 font-mono text-xs">{row.compute_tray}</td>
                  <td className="px-3 py-1.5 border-b border-gray-100">{row.gpu_index}</td>
                  <td className="px-3 py-1.5 border-b border-gray-100">{row.station_index}</td>
                  <td className="px-3 py-1.5 border-b border-gray-100">{row.port_index}</td>
                  <td className="px-3 py-1.5 border-b border-gray-100 font-mono text-xs">{row.ifoe_interface}</td>
                  <td className="px-3 py-1.5 border-b border-gray-100 font-mono text-xs">{row.compute_mac}</td>
                  <td className="px-3 py-1.5 border-b border-gray-100">{linkStatusBadge(row.link_status)}</td>
                  <td className="px-3 py-1.5 border-b border-gray-100">{row.speed || '—'}</td>
                  <td className="px-3 py-1.5 border-b border-gray-100 font-mono text-xs">{row.switch_tray || '—'}</td>
                  <td className="px-3 py-1.5 border-b border-gray-100">{row.asic || '—'}</td>
                  <td className="px-3 py-1.5 border-b border-gray-100 font-mono text-xs">{row.switch_port || '—'}</td>
                  <td className="px-3 py-1.5 border-b border-gray-100 text-center">
                    {row.mapped ? (
                      <CheckCircle className="h-4 w-4 text-green-500 inline" />
                    ) : (
                      <XCircle className="h-4 w-4 text-red-400 inline" />
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

type TabKey = 'compute' | 'switch' | 'topology'

export function IFoEDetailsPage() {
  const [data, setData] = useState<IFoEData | null>(null)
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [activeTab, setActiveTab] = useState<TabKey>('compute')

  // Compute sub-selectors
  const [selectedComputeTray, setSelectedComputeTray] = useState<string>('')
  // Switch sub-selectors
  const [selectedSwitchTray, setSelectedSwitchTray] = useState<string>('')
  const [selectedAsic, setSelectedAsic] = useState<'asic0' | 'asic1'>('asic0')
  // MAC table search
  const [macSearch, setMacSearch] = useState('')

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await api.getIFoEData() as IFoEData
      setData(resp)
      // Auto-select first tray if not set
      const computeKeys = Object.keys(resp.compute_devices || {})
      if (computeKeys.length && !selectedComputeTray) setSelectedComputeTray(computeKeys[0])
      const switchKeys = Object.keys(resp.switch_vlan || {})
      if (switchKeys.length && !selectedSwitchTray) setSelectedSwitchTray(switchKeys[0])
    } catch {
      // silently fail — page will show "no data"
    } finally {
      setLoading(false)
    }
  }, [selectedComputeTray, selectedSwitchTray])

  useEffect(() => {
    fetchData()
    const id = setInterval(fetchData, 300_000)
    return () => clearInterval(id)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      await api.refreshIFoEData()
      // Wait a moment then re-fetch
      setTimeout(fetchData, 3000)
    } finally {
      setRefreshing(false)
    }
  }

  const computeTrays = Object.keys(data?.compute_devices ?? {})
  const switchTrays = Object.keys(data?.switch_vlan ?? {})

  const deviceRows = data?.compute_devices?.[selectedComputeTray] ?? []
  const portRows = data?.compute_ports?.[selectedComputeTray] ?? []
  const vlanRows = data?.switch_vlan?.[selectedSwitchTray]?.[selectedAsic] ?? []
  const macRows = (data?.switch_mac?.[selectedSwitchTray]?.[selectedAsic] ?? []).filter((r) => {
    if (!macSearch) return true
    return Object.values(r).some((v) => String(v).toLowerCase().includes(macSearch.toLowerCase()))
  })

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">IFoE Details</h1>
          <p className="text-gray-500 mt-1">
            {data?.last_updated
              ? `Last updated: ${new Date(data.last_updated).toLocaleString()}`
              : 'No data collected yet'}
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium text-sm"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh Now
        </button>
      </div>

      {/* Errors banner */}
      {data && Object.keys(data.errors).length > 0 && (
        <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <p className="text-sm font-semibold text-yellow-800 mb-1">Collection errors:</p>
          <ul className="text-xs text-yellow-700 space-y-0.5">
            {Object.entries(data.errors).map(([host, err]) => (
              <li key={host}><span className="font-mono">{host}</span>: {err}</li>
            ))}
          </ul>
        </div>
      )}

      {/* No data state */}
      {!loading && !data && (
        <Card>
          <CardContent className="py-12 text-center text-gray-400">
            <p className="text-lg">No IFoE data available.</p>
            <p className="text-sm mt-1">Configure compute and switch trays in Rack Config, then click Refresh Now.</p>
          </CardContent>
        </Card>
      )}

      {data && (
        <>
          {/* Tab selector */}
          <div className="flex gap-1 border-b border-gray-200">
            {([
              { key: 'compute', label: 'Compute Trays' },
              { key: 'switch', label: 'Switch Trays' },
              { key: 'topology', label: 'Topology Map' },
            ] as { key: TabKey; label: string }[]).map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`px-5 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                  activeTab === key
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Compute Trays tab */}
          {activeTab === 'compute' && (
            <div className="space-y-6">
              {/* Tray selector */}
              <div className="flex items-center gap-3">
                <label className="text-sm font-medium text-gray-700">Compute Tray:</label>
                <select
                  className="px-3 py-1.5 border border-gray-300 rounded text-sm"
                  value={selectedComputeTray}
                  onChange={(e) => setSelectedComputeTray(e.target.value)}
                >
                  {computeTrays.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>

              {/* Table A — GPU Devices */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">GPU Devices (afmctl show device)</CardTitle>
                </CardHeader>
                <CardContent>
                  <GenericTable rows={deviceRows} />
                </CardContent>
              </Card>

              {/* Table B — IFoE Ports */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">IFoE Ports (afmctl show port)</CardTitle>
                </CardHeader>
                <CardContent>
                  <GenericTable rows={portRows} />
                </CardContent>
              </Card>
            </div>
          )}

          {/* Switch Trays tab */}
          {activeTab === 'switch' && (
            <div className="space-y-6">
              {/* Tray + ASIC selector */}
              <div className="flex items-center gap-4 flex-wrap">
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-gray-700">Switch Tray:</label>
                  <select
                    className="px-3 py-1.5 border border-gray-300 rounded text-sm"
                    value={selectedSwitchTray}
                    onChange={(e) => setSelectedSwitchTray(e.target.value)}
                  >
                    {switchTrays.map((t) => <option key={t} value={t}>{t}</option>)}
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-gray-700">ASIC:</label>
                  <select
                    className="px-3 py-1.5 border border-gray-300 rounded text-sm"
                    value={selectedAsic}
                    onChange={(e) => setSelectedAsic(e.target.value as 'asic0' | 'asic1')}
                  >
                    <option value="asic0">asic0</option>
                    <option value="asic1">asic1</option>
                  </select>
                </div>
              </div>

              {/* Table C — VLAN Brief */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">VLAN Brief (show vlan brief)</CardTitle>
                </CardHeader>
                <CardContent>
                  <GenericTable rows={vlanRows} />
                </CardContent>
              </Card>

              {/* Table D — MAC Table */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">MAC Table (show mac)</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <input
                    type="text"
                    placeholder="Search MAC, port, VLAN..."
                    className="w-72 px-3 py-1.5 border border-gray-300 rounded text-sm focus:ring-1 focus:ring-blue-500"
                    value={macSearch}
                    onChange={(e) => setMacSearch(e.target.value)}
                  />
                  <GenericTable rows={macRows} />
                </CardContent>
              </Card>
            </div>
          )}

          {/* Topology Map tab */}
          {activeTab === 'topology' && (
            <Card>
              <CardHeader>
                <CardTitle>Port-to-Switch Topology Map</CardTitle>
              </CardHeader>
              <CardContent>
                <TopologyTable rows={data.topology} />
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
