/**
 * GPUSummaryPage — Compute → GPUs → Summary
 *
 * Section 1: Compute Tray GPU Devices (rack IFoE deployments)
 *   - All trays combined into one searchable/sortable DataTable
 *   - Each row has a "Compute Tray" column prepended
 *
 * Section 2: GPU Software & Drivers (all deployment types)
 *   - ROCm version, amdgpu driver, firmware per node/compute-tray
 *   - Embedded inline (no duplicate page header)
 */

import { useState, useEffect } from 'react'
import { RefreshCw, Cpu, Package } from 'lucide-react'
import { DonutChart, HorizontalStackedBar, COLORS } from '@/components/charts/ClusterCharts'
import { ExpandableChartCard } from '@/components/charts/ChartModal'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { useIFoEData } from '@/hooks/useIFoEData'
import { useClusterStore } from '@/stores/clusterStore'

// ---------------------------------------------------------------------------
// Section 1 — Compute Tray GPU Devices (afmctl show device, all trays)
// ---------------------------------------------------------------------------

function GPUDevicesSection() {
  const { data, refreshing, triggerRefresh } = useIFoEData()
  // Authoritative node list from /api/nodes — used for correct total count in donut
  const allNodes = useClusterStore((state) => state.nodes)

  // Compute tray rows (have afmctl data)
  const computeTrayRows = Object.entries(data?.compute_trays ?? data?.compute_devices ?? {}).flatMap(
    ([host, rows]: [string, any[]]) => (rows || []).map((r: any) => ({ host, ...r }))
  )

  // Regular GPU node rows — flatten amd-smi list output (one row per GPU per host)
  const regularNodeGpus: Record<string, any[]> = (data as any)?.regular_node_gpus ?? {}
  const regularNodeRows = ((data as any)?.regular_nodes ?? []).flatMap((h: string) => {
    const gpus: any[] = regularNodeGpus[h] ?? []
    if (gpus.length === 0) return [{ host: h }]
    return gpus.map((g: any) => ({ host: h, ...g }))
  })

  const buildCols = (rows: any[], firstCol: string) => {
    if (rows.length === 0) return [{ title: firstCol === 'host' ? 'Host' : firstCol, data: firstCol, className: 'dt-left font-medium' }]
    const keys = Object.keys(rows[0])
    return keys.map(k => ({
      title: k === 'host' ? 'Host' : k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      data: k,
      className: k === 'host' ? 'dt-left font-medium' : 'dt-left font-mono text-xs',
    }))
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">GPU Node Devices</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            GPU nodes configured under Node Groups. Compute trays (IFoE-capable) are auto-detected via afmctl.
            {data?.last_updated && ` · ${new Date(data.last_updated).toLocaleString()}`}
          </p>
        </div>
        <button onClick={triggerRefresh} disabled={refreshing}
          className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm">
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* ── Visual Overview ── */}
      {data && (computeTrayRows.length > 0 || regularNodeRows.length > 0 || allNodes.length > 0) && (() => {
        const numComputeTrays = Object.keys(data?.compute_trays ?? data?.compute_devices ?? {}).length
        const numRegularNodes = ((data as any)?.regular_nodes ?? []).length
        // Use authoritative node count from /api/nodes; fall back to rack data sum
        const totalNodes = allNodes.length || (numComputeTrays + numRegularNodes)
        const unknownNodes = Math.max(0, totalNodes - numComputeTrays - numRegularNodes)

        const trayBarData = Object.entries(data?.compute_trays ?? data?.compute_devices ?? {}).map(
          ([host, rows]: [string, any[]]) => ({ name: host, GPUs: (rows || []).length })
        )
        const trayBarH = Math.max(120, trayBarData.length * 32 + 60)

        return (
          <div className="space-y-4">
            {/* Donut: node type breakdown */}
            <div className="flex justify-center">
              <DonutChart
                title="GPU Node Types"
                data={[
                  { name: 'Compute Trays', value: numComputeTrays, color: COLORS.purple },
                  { name: 'Regular GPU Nodes', value: numRegularNodes, color: COLORS.blue },
                  ...(unknownNodes > 0 ? [{ name: 'Other / Detecting', value: unknownNodes, color: COLORS.gray }] : []),
                ]}
                centerLabel={String(totalNodes)}
                centerSub="GPU nodes"
                size={180}
              />
            </div>

            {/* GPU devices per compute tray — horizontal so full hostnames fit */}
            {trayBarData.length > 0 && (
              <ExpandableChartCard title="GPU Devices per Compute Tray"
                small={<HorizontalStackedBar data={trayBarData} keys={['GPUs']} colors={[COLORS.purple]} unit=" GPUs" maxValue={Math.max(1,...trayBarData.map(r=>r.GPUs))} height={trayBarH}/>}
                large={<HorizontalStackedBar data={trayBarData} keys={['GPUs']} colors={[COLORS.purple]} unit=" GPUs" maxValue={Math.max(1,...trayBarData.map(r=>r.GPUs))} height={Math.max(200,trayBarData.length*42+80)}/>}
              />
            )}
          </div>
        )
      })()}

      {data && Object.keys(data.errors).length > 0 && (
        <div className="p-3 bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-700 rounded-lg">
          <p className="text-xs font-semibold text-yellow-800 dark:text-yellow-300 mb-1">Collection errors:</p>
          <ul className="text-xs text-yellow-700 dark:text-yellow-400 space-y-0.5">
            {Object.entries(data.errors).map(([host, err]) => (
              <li key={host}><span className="font-mono">{host}</span>: {String(err)}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Compute Trays */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Cpu className="h-4 w-4 text-purple-600" />
            Compute Trays
            <span className="text-sm font-normal text-gray-500 dark:text-gray-400">
              IFoE-capable GPU nodes (afmctl show device)
            </span>
            {computeTrayRows.length > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-purple-100 dark:bg-purple-900 text-purple-700 dark:text-purple-300 rounded text-xs font-semibold">
                {Object.keys(data?.compute_trays ?? data?.compute_devices ?? {}).length} trays
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {computeTrayRows.length === 0 ? (
            <p className="text-sm text-gray-400 dark:text-gray-500 italic py-4">
              {data ? 'No compute trays detected. GPU nodes that have afmctl installed will appear here.' : 'Loading...'}
            </p>
          ) : (
            <CustomDataTable columns={buildCols(computeTrayRows, 'host')} data={computeTrayRows}
              defaultPageLength={25} pageLengthOptions={[25, 50, 100]} />
          )}
        </CardContent>
      </Card>

      {/* Regular GPU Nodes */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Cpu className="h-4 w-4 text-blue-600" />
            Regular GPU Nodes
            <span className="text-sm font-normal text-gray-500 dark:text-gray-400">
              GPU nodes without IFoE connectivity — GPU list from amd-smi
            </span>
            {regularNodeRows.length > 0 && (
              <span className="ml-2 px-2 py-0.5 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded text-xs font-semibold">
                {regularNodeRows.length} nodes
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {regularNodeRows.length === 0 ? (
            <p className="text-sm text-gray-400 dark:text-gray-500 italic py-4">
              {data ? 'No regular GPU nodes detected, or all GPU nodes are compute trays.' : 'Loading...'}
            </p>
          ) : (
            <CustomDataTable
              columns={buildCols(regularNodeRows, 'host')}
              data={regularNodeRows}
              defaultPageLength={25}
              pageLengthOptions={[25, 50, 100]}
            />
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Section 2 — GPU Software & Drivers (cluster nodes + compute trays)
// ---------------------------------------------------------------------------

function GPUSoftwareSection() {
  const cachedSoftwareData = useClusterStore((state) => state.gpuSoftwareData)
  const setGPUSoftwareData = useClusterStore((state) => state.setGPUSoftwareData)
  const [loading, setLoading] = useState(false)
  const [lastUpdate, setLastUpdate] = useState('')

  const fetchData = async () => {
    setLoading(true)
    try {
      const resp = await fetch('/api/software/gpu')
      const data = await resp.json()
      setGPUSoftwareData(data)
      setLastUpdate(new Date().toLocaleString())
    } catch (e) {
      console.error('Failed to fetch GPU software info:', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!cachedSoftwareData) {
      fetchData()
    } else if (cachedSoftwareData.timestamp) {
      setLastUpdate(new Date(cachedSoftwareData.timestamp).toLocaleString())
    }
    const id = setInterval(fetchData, 300_000)
    return () => clearInterval(id)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const sw = cachedSoftwareData

  // Driver / ROCm versions table
  const driverRows = (() => {
    if (!sw?.rocm_version) return []
    return Object.entries(sw.rocm_version).map(([node, v]: [string, any]) => ({
      node,
      amdsmi_tool:      v.amdsmi_tool      || 'N/A',
      amdsmi_library:   v.amdsmi_library   || 'N/A',
      rocm_version:     v.rocm_version     || 'N/A',
      amdgpu_version:   v.amdgpu_version   || 'N/A',
      amd_hsmp_version: v.amd_hsmp_version || 'N/A',
    }))
  })()

  // Firmware table — one row per node, firmware components as columns
  const firmwareTable = (() => {
    if (!sw?.gpu_firmware) return { rows: [], columns: [] as any[] }

    const fwComponents = new Set<string>()
    const gpuIds = new Set<string>()

    Object.values(sw.gpu_firmware).forEach((data: any) => {
      if (!Array.isArray(data)) return
      data.forEach((gpu: any) => {
        gpuIds.add(`card${gpu.gpu ?? 0}`)
        ;(gpu.fw_list || []).forEach((fw: any) => fwComponents.add(fw.fw_id))
      })
    })

    const rows: any[] = []
    const sortedGpus = [...gpuIds].sort((a, b) =>
      parseInt(a.replace('card', '')) - parseInt(b.replace('card', ''))
    )

    Object.entries(sw.gpu_firmware).forEach(([node, data]: [string, any]) => {
      if (!Array.isArray(data)) return
      const row: any = { node }
      fwComponents.forEach((fw) => {
        sortedGpus.forEach((gpuId) => {
          const gpuNum = parseInt(gpuId.replace('card', ''))
          const gpuData = data.find((g: any) => g.gpu === gpuNum)
          const entry = gpuData?.fw_list?.find((f: any) => f.fw_id === fw)
          row[`${fw}_${gpuId}`] = entry?.fw_version || '-'
        })
      })
      rows.push(row)
    })

    const ref = rows[0] || {}
    const columns: any[] = [{ title: 'Node', data: 'node', className: 'dt-left font-medium' }]
    ;[...fwComponents].sort().forEach((fw) => {
      sortedGpus.forEach((gpuId) => {
        const key = `${fw}_${gpuId}`
        columns.push({
          title: `${fw}<br/><span class="text-xs text-gray-500">${gpuId}</span>`,
          data: key,
          className: 'dt-center font-mono text-xs',
          render: (val: string) =>
            val !== ref[key] && val !== '-' && ref[key] !== '-'
              ? `<span class="text-red-600 font-medium">${val}</span>`
              : val,
        })
      })
    })

    return { rows, columns }
  })()

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            GPU Software &amp; Drivers
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            ROCm version, amdgpu driver and firmware — cluster nodes &amp; compute trays
            {lastUpdate && ` · ${lastUpdate}`}
          </p>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* ROCm & Driver Versions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Package className="h-4 w-4" />
            ROCm &amp; Driver Versions
          </CardTitle>
        </CardHeader>
        <CardContent>
          {driverRows.length === 0 ? (
            <p className="text-center py-6 text-gray-400 dark:text-gray-500 text-sm">
              {loading
                ? 'Loading...'
                : 'No driver data. Ensure amd-smi is installed on nodes/compute trays.'}
            </p>
          ) : (
            <CustomDataTable
              columns={[
                { title: 'Node', data: 'node', className: 'dt-left font-medium' },
                { title: 'AMDSMI Tool', data: 'amdsmi_tool', className: 'dt-left font-mono text-xs' },
                { title: 'AMDSMI Library', data: 'amdsmi_library', className: 'dt-left font-mono text-xs' },
                { title: 'ROCm Version', data: 'rocm_version', className: 'dt-left font-mono' },
                { title: 'AMDGPU Driver', data: 'amdgpu_version', className: 'dt-left font-mono' },
                { title: 'AMD HSMP Driver', data: 'amd_hsmp_version', className: 'dt-left font-mono text-xs' },
              ]}
              data={driverRows}
              defaultPageLength={25}
              pageLengthOptions={[25, 50, 100]}
            />
          )}
        </CardContent>
      </Card>

      {/* GPU Firmware Versions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Cpu className="h-4 w-4" />
            GPU Firmware Versions
          </CardTitle>
        </CardHeader>
        <CardContent>
          {firmwareTable.rows.length === 0 ? (
            <p className="text-center py-6 text-gray-400 dark:text-gray-500 text-sm">
              {loading ? 'Loading...' : 'No firmware data available.'}
            </p>
          ) : (
            <>
              <CustomDataTable
                columns={firmwareTable.columns}
                data={firmwareTable.rows}
                defaultPageLength={25}
                pageLengthOptions={[25, 50, 100]}
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                Format: Component / GPU ID. Red = version mismatch vs first node.
              </p>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Combined page
// ---------------------------------------------------------------------------

export function GPUSummaryPage() {
  return (
    <div className="space-y-8">
      <GPUDevicesSection />
      <div className="border-t border-gray-200 dark:border-gray-700" />
      <GPUSoftwareSection />
    </div>
  )
}
