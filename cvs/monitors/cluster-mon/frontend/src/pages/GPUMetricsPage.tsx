import { useEffect, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { useClusterStore } from '@/stores/clusterStore'
import { formatPercentage, formatTemperature, formatPower, formatBytes } from '@/utils/format'

export function GPUMetricsPage() {
  const latestMetrics = useClusterStore((state) => state.latestMetrics)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<string>('')

  useEffect(() => {
    if (latestMetrics?.timestamp) {
      setLastUpdate(new Date(latestMetrics.timestamp).toLocaleString())
    }

    console.log('GPU Metrics Page - latestMetrics updated:', {
      hasMetrics: !!latestMetrics,
      hasGpu: !!latestMetrics?.gpu,
      gpuKeys: latestMetrics?.gpu ? Object.keys(latestMetrics.gpu) : []
    })
  }, [latestMetrics])

  const getGPUTableData = () => {
    const data: any[] = []

    if (!latestMetrics?.gpu) return data

    const utilData = latestMetrics.gpu.utilization || {}
    const memData = latestMetrics.gpu.memory || {}
    const tempData = latestMetrics.gpu.temperature || {}
    const powerData = latestMetrics.gpu.power || {}

    Object.entries(utilData).forEach(([node, gpuData]: [string, any]) => {
      if (typeof gpuData === 'object' && !gpuData.error) {
        Object.entries(gpuData).forEach(([gpuId, metrics]: [string, any]) => {
          if (typeof metrics === 'object') {
            const memMetrics = memData[node]?.[gpuId] || {}
            const tempMetrics = tempData[node]?.[gpuId] || {}

            const memUsed = parseInt(memMetrics['VRAM Total Used Memory (B)']) || 0
            const memTotal = parseInt(memMetrics['VRAM Total Memory (B)']) || 0
            const memPercent = memTotal > 0 ? (memUsed / memTotal) * 100 : 0
            const util = parseFloat(metrics['GPU use (%)']) || 0
            const temp = parseFloat(tempMetrics['Temperature (Sensor junction) (C)']) ||
                        parseFloat(tempMetrics['Temperature (Sensor edge) (C)']) || 0

            data.push({
              node,
              gpu: gpuId,
              utilization: util,
              memUsed,
              memTotal,
              memPercent,
              temperature: temp,
              power: 0,
            })
          }
        })
      }
    })

    return data
  }

  const getPCIeTableData = () => {
    const data: any[] = []
    if (!latestMetrics?.gpu) {
      console.log('PCIe: No latestMetrics.gpu')
      return data
    }

    const pcieMetricData = latestMetrics.gpu.pcie || {}
    const pcieLinkData = latestMetrics.gpu.pcie_link_status || {}
    const infoData = latestMetrics.gpu.info || {}

    const samplePcieNode = Object.keys(pcieLinkData)[0]
    const samplePcieData = pcieLinkData[samplePcieNode]
    const samplePcieStr = samplePcieData
      ? JSON.stringify(samplePcieData, null, 2).split('\n').slice(0, 30).join('\n')
      : 'No data'

    console.log('PCIe Debug:', {
      pcieMetricDataNodes: Object.keys(pcieMetricData).length,
      pcieLinkDataNodes: Object.keys(pcieLinkData).length,
      sampleNode: samplePcieNode || 'None',
      sampleDataPreview: samplePcieStr
    })

    // Use pcie_link_status for actual link info, pcie for error counters
    Object.entries(pcieLinkData).forEach(([node, gpuData]: [string, any]) => {
      if (typeof gpuData === 'object') {
        Object.entries(gpuData).forEach(([gpuId, linkInfo]: [string, any]) => {
          // Get error metrics from pcie data
          const pcieErrors = pcieMetricData[node]?.gpu_data?.find((g: any) =>
            `card${g.gpu}` === gpuId
          )?.pcie || {}

          // Use pcie data directly (has all fields)
          const pcieData = pcieMetricData[node]?.[gpuId] || linkInfo

          data.push({
            node,
            gpu: gpuId,
            width: pcieData.width || '-',
            speed: pcieData.speed || '-',
            bandwidth: pcieData.bandwidth || '-',
            replay_count: pcieData.replay_count || 0,
            l0_to_recovery_count: pcieData.l0_to_recovery_count || 0,
            nak_sent: pcieData.nak_sent_count || 0,
            nak_received: pcieData.nak_received_count || 0,
          })
        })
      }
    })

    return data
  }

  const getXGMITableData = () => {
    const data: any[] = []
    if (!latestMetrics?.gpu?.xgmi) {
      console.log('XGMI: No latestMetrics.gpu.xgmi')
      return data
    }

    const sampleNode = Object.keys(latestMetrics.gpu.xgmi)[0]
    const sampleData = latestMetrics.gpu.xgmi[sampleNode]
    const sampleDataStr = sampleData
      ? JSON.stringify(sampleData, null, 2).split('\n').slice(0, 30).join('\n')
      : 'No data'

    console.log('XGMI Debug:', {
      xgmiNodes: Object.keys(latestMetrics.gpu.xgmi).length,
      sampleNode: sampleNode || 'None',
      sampleDataPreview: sampleDataStr
    })

    Object.entries(latestMetrics.gpu.xgmi).forEach(([node, nodeData]: [string, any]) => {
      // Handle both formats: direct array or wrapped in gpu_data
      const gpuArray = Array.isArray(nodeData) ? nodeData : (nodeData.gpu_data || [])

      if (Array.isArray(gpuArray)) {
        gpuArray.forEach((gpu: any) => {
          const xgmi = gpu.xgmi_err || gpu.xgmi
          if (xgmi && xgmi !== 'N/A' && typeof xgmi === 'object') {
            data.push({
              node,
              gpu: `card${gpu.gpu || 0}`,
              error_count: xgmi.error_count || 0,
              status: xgmi.status || 'N/A',
            })
          } else {
            data.push({
              node,
              gpu: `card${gpu.gpu || 0}`,
              error_count: 0,
              status: xgmi === 'N/A' ? 'N/A' : 'No errors',
            })
          }
        })
      }
    })

    return data
  }

  const getECCTableData = () => {
    const data: any[] = []
    if (!latestMetrics?.gpu?.ras_errors) return data

    Object.entries(latestMetrics.gpu.ras_errors).forEach(([node, nodeData]: [string, any]) => {
      // Handle both formats: direct array or wrapped in gpu_data
      const gpuArray = Array.isArray(nodeData) ? nodeData : (nodeData.gpu_data || [])

      if (Array.isArray(gpuArray)) {
        gpuArray.forEach((gpu: any) => {
          const ecc = gpu.ecc || {}
          const ecc_blocks = gpu.ecc_blocks || {}

          data.push({
            node,
            gpu: `card${gpu.gpu || 0}`,
            total_correctable: ecc.total_correctable_count || 0,
            total_uncorrectable: ecc.total_uncorrectable_count || 0,
            cache_correctable: ecc.cache_correctable_count || 0,
            cache_uncorrectable: ecc.cache_uncorrectable_count || 0,
            umc_correctable: ecc_blocks.UMC?.correctable_count || 0,
            umc_uncorrectable: ecc_blocks.UMC?.uncorrectable_count || 0,
            gfx_correctable: ecc_blocks.GFX?.correctable_count || 0,
            gfx_uncorrectable: ecc_blocks.GFX?.uncorrectable_count || 0,
          })
        })
      }
    })

    return data
  }

  const gpuData = getGPUTableData()
  const pcieData = getPCIeTableData()
  const xgmiData = getXGMITableData()
  const eccData = getECCTableData()

  // GPU Table columns
  const gpuColumns = [
    { title: 'Node', data: 'node', className: 'dt-left font-medium' },
    { title: 'GPU ID', data: 'gpu', className: 'dt-left' },
    {
      title: 'Utilization',
      data: 'utilization',
      className: 'dt-right',
      render: (data: number) => {
        const color = data > 90 ? 'text-red-600' : data > 70 ? 'text-yellow-600' : 'text-green-600'
        return `<span class="${color} font-medium">${formatPercentage(data)}</span>`
      },
    },
    {
      title: 'Memory Used',
      data: 'memUsed',
      className: 'dt-right',
      render: (data: number) => `<span class="font-mono text-xs">${formatBytes(data)}</span>`,
    },
    {
      title: 'Memory Total',
      data: 'memTotal',
      className: 'dt-right',
      render: (data: number) => `<span class="font-mono text-xs">${formatBytes(data)}</span>`,
    },
    {
      title: 'Memory %',
      data: 'memPercent',
      className: 'dt-right',
      render: (data: number) => formatPercentage(data),
    },
    {
      title: 'Temperature',
      data: 'temperature',
      className: 'dt-right',
      render: (data: number) => {
        const color = data > 85 ? 'text-red-600 font-medium' : data > 70 ? 'text-yellow-600' : 'text-gray-900'
        return `<span class="${color}">${formatTemperature(data)}</span>`
      },
    },
    {
      title: 'Power',
      data: 'power',
      className: 'dt-right',
      render: (data: number) => data > 0 ? formatPower(data) : '-',
    },
  ]

  // PCIe Table columns
  const pcieColumns = [
    { title: 'Node', data: 'node', className: 'dt-left font-medium' },
    { title: 'GPU ID', data: 'gpu', className: 'dt-left' },
    { title: 'Link Width', data: 'width', className: 'dt-center font-medium' },
    { title: 'Link Speed', data: 'speed', className: 'dt-center font-medium' },
    { title: 'Bandwidth', data: 'bandwidth', className: 'dt-right' },
    {
      title: 'Replay Count',
      data: 'replay_count',
      className: 'dt-right',
      render: (data: number) => {
        const color = data > 100 ? 'text-red-600 font-medium' : data > 10 ? 'text-yellow-600' : ''
        return `<span class="${color}">${data}</span>`
      },
    },
    {
      title: 'L0 to Recovery',
      data: 'l0_to_recovery_count',
      className: 'dt-right',
      render: (data: number) => {
        const color = data > 100 ? 'text-red-600 font-medium' : data > 10 ? 'text-yellow-600' : ''
        return `<span class="${color}">${data}</span>`
      },
    },
    {
      title: 'NAK Sent',
      data: 'nak_sent',
      className: 'dt-right',
      render: (data: number) => {
        const color = data > 100 ? 'text-red-600 font-medium' : data > 10 ? 'text-yellow-600' : ''
        return `<span class="${color}">${data}</span>`
      },
    },
    {
      title: 'NAK Received',
      data: 'nak_received',
      className: 'dt-right',
      render: (data: number) => {
        const color = data > 100 ? 'text-red-600 font-medium' : data > 10 ? 'text-yellow-600' : ''
        return `<span class="${color}">${data}</span>`
      },
    },
  ]

  // XGMI Table columns
  const xgmiColumns = [
    { title: 'Node', data: 'node', className: 'dt-left font-medium' },
    { title: 'GPU ID', data: 'gpu', className: 'dt-left' },
    { title: 'Status', data: 'status', className: 'dt-center' },
    {
      title: 'Error Count',
      data: 'error_count',
      className: 'dt-right',
      render: (data: number) => {
        const color = data > 100 ? 'text-red-600 font-medium' : data > 0 ? 'text-yellow-600' : 'text-green-600'
        return `<span class="${color}">${data}</span>`
      },
    },
  ]

  // ECC Table columns
  const eccColumns = [
    { title: 'Node', data: 'node', className: 'dt-left font-medium' },
    { title: 'GPU ID', data: 'gpu', className: 'dt-left' },
    {
      title: 'Total Correctable',
      data: 'total_correctable',
      className: 'dt-right',
      render: (data: number) => {
        const color = data > 1000 ? 'text-yellow-600' : data > 0 ? 'text-blue-600' : ''
        return `<span class="${color}">${data}</span>`
      },
    },
    {
      title: 'Total Uncorrectable',
      data: 'total_uncorrectable',
      className: 'dt-right',
      render: (data: number) => {
        const color = data > 0 ? 'text-red-600 font-medium' : 'text-green-600'
        return `<span class="${color}">${data}</span>`
      },
    },
    {
      title: 'Cache Correctable',
      data: 'cache_correctable',
      className: 'dt-right',
      render: (data: number) => data > 0 ? `<span class="text-blue-600">${data}</span>` : data,
    },
    {
      title: 'Cache Uncorrectable',
      data: 'cache_uncorrectable',
      className: 'dt-right',
      render: (data: number) => data > 0 ? `<span class="text-red-600">${data}</span>` : data,
    },
    {
      title: 'UMC Correctable',
      data: 'umc_correctable',
      className: 'dt-right',
      render: (data: number) => data > 0 ? `<span class="text-blue-600">${data}</span>` : data,
    },
    {
      title: 'UMC Uncorrectable',
      data: 'umc_uncorrectable',
      className: 'dt-right',
      render: (data: number) => data > 0 ? `<span class="text-red-600 font-medium">${data}</span>` : data,
    },
    {
      title: 'GFX Correctable',
      data: 'gfx_correctable',
      className: 'dt-right',
      render: (data: number) => data > 0 ? `<span class="text-blue-600">${data}</span>` : data,
    },
    {
      title: 'GFX Uncorrectable',
      data: 'gfx_uncorrectable',
      className: 'dt-right',
      render: (data: number) => data > 0 ? `<span class="text-red-600 font-medium">${data}</span>` : data,
    },
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">GPU Metrics</h1>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-4 h-4 text-blue-600 rounded"
            />
            Auto-refresh (60s)
          </label>
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <RefreshCw className="h-4 w-4" />
            Last update: {lastUpdate || 'Never'}
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      {gpuData.length > 0 && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">{gpuData.length}</div>
              <p className="text-xs text-gray-600">Total GPUs</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">
                {formatPercentage(
                  gpuData.reduce((sum, g) => sum + g.utilization, 0) / gpuData.length
                )}
              </div>
              <p className="text-xs text-gray-600">Avg Utilization</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">
                {formatTemperature(
                  gpuData.reduce((sum, g) => sum + g.temperature, 0) / gpuData.length
                )}
              </div>
              <p className="text-xs text-gray-600">Avg Temperature</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">
                {eccData.reduce((sum, e) => sum + e.total_uncorrectable, 0)}
              </div>
              <p className="text-xs text-gray-600">Total ECC Errors</p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* GPU Utilization & Performance Table */}
      <Card>
        <CardHeader>
          <CardTitle>GPU Utilization & Performance</CardTitle>
        </CardHeader>
        <CardContent>
          {gpuData.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <p>No GPU data available</p>
              <p className="text-sm mt-2">
                Configure nodes and SSH access to see GPU metrics
              </p>
            </div>
          ) : (
            <CustomDataTable
              columns={gpuColumns}
              data={gpuData}
              defaultPageLength={50}
              pageLengthOptions={[50, 100, 1000]}
            />
          )}
        </CardContent>
      </Card>

      {/* PCIe Metrics Table */}
      <Card>
        <CardHeader>
          <CardTitle>PCIe Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <CustomDataTable
            columns={pcieColumns}
            data={pcieData}
            defaultPageLength={50}
            pageLengthOptions={[50, 100, 1000]}
          />
        </CardContent>
      </Card>

      {/* XGMI Metrics Table */}
      <Card>
        <CardHeader>
          <CardTitle>XGMI (GPU-to-GPU Interconnect) Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <CustomDataTable
            columns={xgmiColumns}
            data={xgmiData}
            defaultPageLength={50}
            pageLengthOptions={[50, 100, 1000]}
          />
        </CardContent>
      </Card>

      {/* ECC Error Metrics Table */}
      <Card>
        <CardHeader>
          <CardTitle>ECC Memory Error Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <CustomDataTable
            columns={eccColumns}
            data={eccData}
            defaultPageLength={50}
            pageLengthOptions={[50, 100, 1000]}
          />
        </CardContent>
      </Card>
    </div>
  )
}
