import { ClusterOverview } from '@/components/ClusterOverview'
import { NodesGrid } from '@/components/NodesGrid'
import { GenericHeatmap } from '@/components/GenericHeatmap'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { useClusterStore } from '@/stores/clusterStore'
import { Activity, Thermometer, HardDrive, Network } from 'lucide-react'

export function DashboardPage() {
  const latestMetrics = useClusterStore((state) => state.latestMetrics)

  // GPU Utilization Heatmap Data
  const utilizationHeatmapData = (() => {
    const data: any[] = []
    if (!latestMetrics?.gpu) return data

    const utilData = latestMetrics.gpu.utilization || {}
    const memData = latestMetrics.gpu.memory || {}
    const tempData = latestMetrics.gpu.temperature || {}

    Object.entries(utilData).forEach(([node, gpuData]: [string, any]) => {
      if (typeof gpuData === 'object' && !gpuData.error) {
        Object.entries(gpuData).forEach(([gpuId, metrics]: [string, any]) => {
          if (typeof metrics === 'object') {
            const memMetrics = memData[node]?.[gpuId] || {}
            const tempMetrics = tempData[node]?.[gpuId] || {}

            data.push({
              node,
              id: gpuId,
              value: parseFloat(metrics['GPU use (%)']) || 0,
              tooltipData: {
                'Temperature': `${parseFloat(tempMetrics['Temperature (Sensor junction) (C)']) || 0}°C`,
                'Memory': `${(parseInt(memMetrics['VRAM Total Used Memory (B)']) / (1024 * 1024 * 1024)).toFixed(1)} GB`,
              },
            })
          }
        })
      }
    })

    return data
  })()

  // GPU Memory Usage Heatmap Data
  const memoryHeatmapData = (() => {
    const data: any[] = []
    if (!latestMetrics?.gpu?.memory) return data

    const memData = latestMetrics.gpu.memory || {}
    const tempData = latestMetrics.gpu.temperature || {}

    Object.entries(memData).forEach(([node, gpuData]: [string, any]) => {
      if (typeof gpuData === 'object' && !gpuData.error) {
        Object.entries(gpuData).forEach(([gpuId, metrics]: [string, any]) => {
          if (typeof metrics === 'object') {
            const memUsed = parseInt(metrics['VRAM Total Used Memory (B)']) || 0
            const memTotal = parseInt(metrics['VRAM Total Memory (B)']) || 0
            const memPercent = memTotal > 0 ? (memUsed / memTotal) * 100 : 0

            const tempMetrics = tempData[node]?.[gpuId] || {}

            data.push({
              node,
              id: gpuId,
              value: memPercent,
              tooltipData: {
                'Memory Used': `${(memUsed / (1024 * 1024 * 1024)).toFixed(1)} GB`,
                'Memory Total': `${(memTotal / (1024 * 1024 * 1024)).toFixed(0)} GB`,
                'Temperature': `${parseFloat(tempMetrics['Temperature (Sensor junction) (C)']) || 0}°C`,
              },
            })
          }
        })
      }
    })

    return data
  })()

  // GPU Temperature Heatmap Data
  const temperatureHeatmapData = (() => {
    const data: any[] = []
    if (!latestMetrics?.gpu?.temperature) return data

    const tempData = latestMetrics.gpu.temperature || {}
    const utilData = latestMetrics.gpu.utilization || {}
    const memData = latestMetrics.gpu.memory || {}

    Object.entries(tempData).forEach(([node, gpuData]: [string, any]) => {
      if (typeof gpuData === 'object' && !gpuData.error) {
        Object.entries(gpuData).forEach(([gpuId, metrics]: [string, any]) => {
          if (typeof metrics === 'object') {
            const temp = parseFloat(metrics['Temperature (Sensor junction) (C)']) ||
                        parseFloat(metrics['Temperature (Sensor edge) (C)']) || 0

            const utilMetrics = utilData[node]?.[gpuId] || {}
            const memMetrics = memData[node]?.[gpuId] || {}
            const memUsed = parseInt(memMetrics['VRAM Total Used Memory (B)']) || 0

            data.push({
              node,
              id: gpuId,
              value: temp,
              tooltipData: {
                'Utilization': `${parseFloat(utilMetrics['GPU use (%)']) || 0}%`,
                'Memory Used': `${(memUsed / (1024 * 1024 * 1024)).toFixed(1)} GB`,
              },
            })
          }
        })
      }
    })

    return data
  })()

  // NIC Utilization Heatmap Data (for 8 GPU-mapped RDMA NICs: bnxt_re0-7)
  const nicHeatmapData = (() => {
    const data: any[] = []
    if (!latestMetrics?.nic?.rdma_stats) return data

    const rdmaStats = latestMetrics.nic.rdma_stats || {}
    const rdmaLinks = latestMetrics.nic.rdma_links || {}

    // Focus on bnxt_re0 through bnxt_re7 (8 GPU-mapped NICs)
    const gpuMappedNICs = ['bnxt_re0', 'bnxt_re1', 'bnxt_re2', 'bnxt_re3',
                           'bnxt_re4', 'bnxt_re5', 'bnxt_re6', 'bnxt_re7']

    Object.entries(rdmaStats).forEach(([node, devices]: [string, any]) => {
      if (typeof devices === 'object' && !devices.error) {
        gpuMappedNICs.forEach(nicId => {
          const stats = devices[nicId]
          if (stats) {
            const rxBytes = stats.rx_bytes || 0
            const txBytes = stats.tx_bytes || 0
            const totalBytes = Math.abs(rxBytes) + Math.abs(txBytes)

            // Calculate utilization as percentage (assuming 100Gbps = ~12.5GB/s max)
            // Since we don't have time delta, show traffic volume as proxy for utilization
            const trafficGB = totalBytes / (1024 * 1024 * 1024)

            const linkInfo = rdmaLinks[node]?.[`${nicId}/1`] || {}

            data.push({
              node,
              id: nicId,
              value: trafficGB, // Show total GB transferred (not percentage)
              label: `${(trafficGB / 1000).toFixed(1)}TB`,
              tooltipData: {
                'Network Device': linkInfo.netdev || 'N/A',
                'State': linkInfo.state || 'N/A',
                'RX': `${(Math.abs(rxBytes) / (1024 * 1024 * 1024)).toFixed(1)} GB`,
                'TX': `${(Math.abs(txBytes) / (1024 * 1024 * 1024)).toFixed(1)} GB`,
                'RX Packets': stats.rx_pkts?.toLocaleString() || 'N/A',
                'TX Packets': stats.tx_pkts?.toLocaleString() || 'N/A',
              },
            })
          }
        })
      }
    })

    return data
  })()

  // Temperature color stops
  const tempColorStops = [
    { threshold: 85, color: 'bg-red-500' },
    { threshold: 80, color: 'bg-red-400' },
    { threshold: 75, color: 'bg-orange-400' },
    { threshold: 70, color: 'bg-orange-300' },
    { threshold: 65, color: 'bg-yellow-400' },
    { threshold: 60, color: 'bg-yellow-300' },
    { threshold: 50, color: 'bg-green-300' },
    { threshold: 40, color: 'bg-green-200' },
    { threshold: 30, color: 'bg-green-100' },
    { threshold: 0, color: 'bg-blue-100' },
  ]

  // NIC traffic color stops (in GB)
  const nicColorStops = [
    { threshold: 10000, color: 'bg-purple-600' },
    { threshold: 5000, color: 'bg-purple-500' },
    { threshold: 1000, color: 'bg-purple-400' },
    { threshold: 500, color: 'bg-blue-400' },
    { threshold: 100, color: 'bg-blue-300' },
    { threshold: 50, color: 'bg-green-300' },
    { threshold: 10, color: 'bg-green-200' },
    { threshold: 1, color: 'bg-green-100' },
    { threshold: 0, color: 'bg-gray-100' },
  ]

  return (
    <div className="space-y-8">
      <ClusterOverview />

      {/* GPU Utilization Heatmap */}
      {utilizationHeatmapData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              GPU Utilization Heatmap
            </CardTitle>
          </CardHeader>
          <CardContent>
            <GenericHeatmap
              data={utilizationHeatmapData}
              title="Utilization"
              unit="%"
              valueFormatter={(v) => v.toFixed(0)}
            />
          </CardContent>
        </Card>
      )}

      {/* GPU Memory Usage Heatmap */}
      {memoryHeatmapData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <HardDrive className="h-5 w-5" />
              GPU Memory Usage Heatmap
            </CardTitle>
          </CardHeader>
          <CardContent>
            <GenericHeatmap
              data={memoryHeatmapData}
              title="Memory Usage"
              unit="%"
              valueFormatter={(v) => v.toFixed(1)}
            />
          </CardContent>
        </Card>
      )}

      {/* GPU Temperature Heatmap */}
      {temperatureHeatmapData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Thermometer className="h-5 w-5" />
              GPU Temperature Heatmap
            </CardTitle>
          </CardHeader>
          <CardContent>
            <GenericHeatmap
              data={temperatureHeatmapData}
              title="Temperature"
              unit="°C"
              colorStops={tempColorStops}
              valueFormatter={(v) => v.toFixed(0)}
            />
          </CardContent>
        </Card>
      )}

      {/* NIC Aggregate Traffic Heatmap (8 GPU-mapped RDMA NICs) */}
      {nicHeatmapData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Network className="h-5 w-5" />
              RDMA NIC Aggregate Traffic Heatmap
            </CardTitle>
          </CardHeader>
          <CardContent>
            <GenericHeatmap
              data={nicHeatmapData}
              title="Total Traffic"
              unit=" GB"
              colorStops={nicColorStops}
              valueFormatter={(v) => v.toFixed(0)}
            />
            <p className="text-xs text-gray-500 mt-2">
              Shows cumulative RX+TX traffic in GB for the 8 GPU-mapped RDMA NICs (bnxt_re0-7)
            </p>
          </CardContent>
        </Card>
      )}

      <NodesGrid />
    </div>
  )
}
