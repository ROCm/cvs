import { useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card'
import { useClusterStore } from '@/stores/clusterStore'

export function GPUMetricsChart() {
  const latestMetrics = useClusterStore((state) => state.latestMetrics)

  const chartData = useMemo(() => {
    if (!latestMetrics?.gpu?.utilization) return []

    const data: any[] = []

    Object.entries(latestMetrics.gpu.utilization).forEach(([node, gpuData]: [string, any]) => {
      if (typeof gpuData === 'object' && !gpuData.error) {
        Object.entries(gpuData).forEach(([gpuId, metrics]: [string, any]) => {
          if (typeof metrics === 'object') {
            data.push({
              name: `${node.substring(0, 10)}.../${gpuId}`,
              node,
              gpu: gpuId,
              utilization: parseFloat(metrics['GPU use (%)']) || 0,
            })
          }
        })
      }
    })

    return data.slice(0, 20) // Limit to first 20 GPUs for readability
  }, [latestMetrics])

  if (chartData.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>GPU Utilization</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] flex items-center justify-center text-muted-foreground">
            No GPU data available
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>GPU Utilization Across Cluster</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} fontSize={12} />
            <YAxis label={{ value: 'Utilization (%)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="utilization"
              stroke="#8884d8"
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
