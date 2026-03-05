import { useEffect, useState } from 'react'
import { RefreshCw, Cpu, Package } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { useClusterStore } from '@/stores/clusterStore'

export function GPUSoftwarePage() {
  // Use cached data from store - shows immediately when tab is clicked
  const cachedSoftwareData = useClusterStore((state) => state.gpuSoftwareData)
  const setGPUSoftwareData = useClusterStore((state) => state.setGPUSoftwareData)

  const [loading, setLoading] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<string>('')
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchSoftwareInfo = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/software/gpu')
      const data = await response.json()
      setGPUSoftwareData(data)  // Store in global cache
      setLastUpdate(new Date().toLocaleString())
    } catch (error) {
      console.error('Failed to fetch GPU software info:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    // Only fetch if no cached data exists
    if (!cachedSoftwareData) {
      fetchSoftwareInfo()
    } else {
      // Set last update from cached data
      if (cachedSoftwareData.timestamp) {
        setLastUpdate(new Date(cachedSoftwareData.timestamp).toLocaleString())
      }
    }

    if (autoRefresh) {
      // Auto-refresh every 180 seconds (3 minutes) since software rarely changes
      const interval = setInterval(fetchSoftwareInfo, 180000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  // Use cached data for display
  const softwareData = cachedSoftwareData

  // Table 1: ROCM & Driver Versions
  const driverData = (() => {
    if (!softwareData) return []

    const rows: any[] = []
    const nodes = new Set([
      ...Object.keys(softwareData.rocm_version || {}),
    ])

    nodes.forEach((node) => {
      const versionData = softwareData.rocm_version?.[node] || {}

      rows.push({
        node,
        amdsmi_tool: versionData.amdsmi_tool || 'N/A',
        amdsmi_library: versionData.amdsmi_library || 'N/A',
        rocm_version: versionData.rocm_version || 'N/A',
        amdgpu_version: versionData.amdgpu_version || 'N/A',
        amd_hsmp_version: versionData.amd_hsmp_version || 'N/A',
      })
    })

    return rows
  })()

  // Table 2: GPU Firmware Versions (Pivoted - Firmware Components as columns, GPUs as sub-columns)
  const firmwareData = (() => {
    if (!softwareData?.gpu_firmware) return { rows: [], columns: [] }

    const rows: any[] = []
    const firmwareComponents = new Set<string>()
    const gpuIds = new Set<string>()

    // First pass: collect all unique firmware components and GPU IDs
    Object.entries(softwareData.gpu_firmware).forEach(([node, data]: [string, any]) => {
      if (data.error || !Array.isArray(data)) return

      data.forEach((gpu: any) => {
        gpuIds.add(`card${gpu.gpu || 0}`)
        if (gpu.fw_list && Array.isArray(gpu.fw_list)) {
          gpu.fw_list.forEach((fw: any) => {
            firmwareComponents.add(fw.fw_id)
          })
        }
      })
    })

    // Second pass: create pivoted rows (one row per node)
    Object.entries(softwareData.gpu_firmware).forEach(([node, data]: [string, any]) => {
      if (data.error || !Array.isArray(data)) return

      const row: any = { node }

      // For each firmware component and GPU combination, add the version
      firmwareComponents.forEach(fwComp => {
        gpuIds.forEach(gpuId => {
          const gpuNum = parseInt(gpuId.replace('card', ''))
          const gpuData = data.find((g: any) => g.gpu === gpuNum)

          if (gpuData && gpuData.fw_list) {
            const fwEntry = gpuData.fw_list.find((fw: any) => fw.fw_id === fwComp)
            row[`${fwComp}_${gpuId}`] = fwEntry?.fw_version || '-'
          } else {
            row[`${fwComp}_${gpuId}`] = '-'
          }
        })
      })

      rows.push(row)
    })

    // Get reference values from first row for comparison
    const referenceRow = rows[0] || {}

    // Create dynamic columns with red highlighting for mismatches
    const columns = [
      { title: 'Node', data: 'node', className: 'dt-left font-medium' }
    ]

    // Add columns for each firmware component + GPU combination
    Array.from(firmwareComponents).sort().forEach(fwComp => {
      // Sort GPU IDs numerically (card0, card1, card2, ... not card1, card10, card2)
      Array.from(gpuIds).sort((a, b) => {
        const numA = parseInt(a.replace('card', ''))
        const numB = parseInt(b.replace('card', ''))
        return numA - numB
      }).forEach(gpuId => {
        const colKey = `${fwComp}_${gpuId}`
        const referenceValue = referenceRow[colKey]

        columns.push({
          title: `${fwComp}<br/><span class="text-xs text-gray-500">${gpuId}</span>`,
          data: colKey,
          className: 'dt-center font-mono text-xs',
          render: (data: string, type: string, row: any) => {
            // Highlight in red if different from reference (first row)
            if (data !== referenceValue && data !== '-' && referenceValue !== '-') {
              return `<span class="text-red-600 font-medium">${data}</span>`
            }
            return data
          },
        })
      })
    })

    return { rows, columns }
  })()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">GPU Software Information</h1>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-4 h-4 text-blue-600 rounded"
            />
            Auto-refresh (180s)
          </label>
          <button
            onClick={fetchSoftwareInfo}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <div className="text-sm text-gray-600">
            Last update: {lastUpdate || 'Never'}
          </div>
        </div>
      </div>

      {/* ROCM & Driver Versions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Package className="h-5 w-5" />
            ROCM & Driver Versions
          </CardTitle>
        </CardHeader>
        <CardContent>
          {driverData.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No driver information available
            </div>
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
              data={driverData}
              defaultPageLength={50}
              pageLengthOptions={[50, 100, 1000]}
            />
          )}
        </CardContent>
      </Card>

      {/* GPU Firmware Versions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="h-5 w-5" />
            GPU Firmware Versions (by Component and GPU)
          </CardTitle>
        </CardHeader>
        <CardContent>
          {firmwareData.rows.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              {loading ? 'Loading firmware information...' : 'No firmware information available'}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <CustomDataTable
                columns={firmwareData.columns}
                data={firmwareData.rows}
                defaultPageLength={50}
                pageLengthOptions={[50, 100, 1000]}
              />
              <p className="text-xs text-gray-500 mt-2">
                Each firmware component shows versions for all GPUs. Columns format: Component_Name / GPU_ID
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ROCM Libraries */}
      {(() => {
        if (!softwareData?.rocm_libraries) return null

        const libraryData: any[] = []
        const allLibraries = new Set<string>()

        // First pass: collect all unique library names (filter out error messages)
        Object.entries(softwareData.rocm_libraries).forEach(([node, data]: [string, any]) => {
          if (data.libraries && Array.isArray(data.libraries)) {
            data.libraries.forEach((lib: string) => {
              // Skip error messages (ABORT, SessionError, ERROR, etc.)
              if (lib.includes('ABORT') || lib.includes('Error') || lib.includes('ERROR') ||
                  lib.includes('Session') || lib.trim().startsWith('ERROR')) {
                return
              }

              // Parse "package_name version" format
              const parts = lib.trim().split(/\s+/)
              if (parts.length >= 1) {
                allLibraries.add(parts[0])
              }
            })
          }
        })

        // Second pass: create rows (one per node) with library versions
        Object.entries(softwareData.rocm_libraries).forEach(([node, data]: [string, any]) => {
          const row: any = { node }

          if (data.libraries && Array.isArray(data.libraries)) {
            data.libraries.forEach((lib: string) => {
              // Skip error messages
              if (lib.includes('ABORT') || lib.includes('Error') || lib.includes('ERROR') ||
                  lib.includes('Session') || lib.trim().startsWith('ERROR')) {
                return
              }

              const parts = lib.trim().split(/\s+/)
              if (parts.length >= 2) {
                const libName = parts[0]
                const version = parts[1]
                row[libName] = version
              } else if (parts.length === 1) {
                row[parts[0]] = 'installed'
              }
            })
          }

          // Fill missing libraries with '-'
          allLibraries.forEach(libName => {
            if (!row[libName]) {
              row[libName] = '-'
            }
          })

          libraryData.push(row)
        })

        // Create columns
        const libraryColumns = [
          { title: 'Node', data: 'node', className: 'dt-left font-medium' }
        ]

        Array.from(allLibraries).sort().forEach(libName => {
          libraryColumns.push({
            title: libName,
            data: libName,
            className: 'dt-left font-mono text-xs',
          })
        })

        return (
          <Card>
            <CardHeader>
              <CardTitle>Installed ROCM Libraries</CardTitle>
            </CardHeader>
            <CardContent>
              <CustomDataTable
                columns={libraryColumns}
                data={libraryData}
                defaultPageLength={50}
                pageLengthOptions={[50, 100, 1000]}
              />
            </CardContent>
          </Card>
        )
      })()}
    </div>
  )
}
