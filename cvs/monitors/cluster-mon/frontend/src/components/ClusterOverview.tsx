import { useEffect } from 'react'
import { Activity, Server, HardDrive, Thermometer } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card'
import { useClusterStore } from '@/stores/clusterStore'
import { api } from '@/services/api'
import { formatPercentage } from '@/utils/format'

export function ClusterOverview() {
  const clusterStatus = useClusterStore((state) => state.clusterStatus)
  const setClusterStatus = useClusterStore((state) => state.setClusterStatus)
  const isConnected = useClusterStore((state) => state.isConnected)

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await api.getClusterStatus()
        setClusterStatus(status)
      } catch (error) {
        console.error('Failed to fetch cluster status:', error)
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 60000)
    return () => clearInterval(interval)
  }, [setClusterStatus])

  if (!clusterStatus) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {[...Array(5)].map((_, i) => (
          <Card key={i} className="animate-pulse">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Loading...</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">--</div>
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-500'
      case 'degraded': return 'text-yellow-500'
      case 'critical': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'healthy': return 'Healthy'
      case 'degraded': return 'Degraded'
      case 'critical': return 'Critical'
      default: return 'Unknown'
    }
  }

  const stats = [
    {
      title: 'Total Nodes',
      value: clusterStatus.total_nodes,
      description: `${clusterStatus.healthy_nodes} healthy`,
      icon: Server,
      color: clusterStatus.healthy_nodes === clusterStatus.total_nodes ? 'text-green-500' : 'text-yellow-500',
    },
    {
      title: 'Healthy Nodes',
      value: clusterStatus.healthy_nodes,
      description: `${clusterStatus.unhealthy_nodes} unhealthy, ${clusterStatus.unreachable_nodes} unreachable`,
      icon: Server,
      color: 'text-green-500',
    },
    {
      title: 'Avg GPU Utilization',
      value: formatPercentage(clusterStatus.avg_gpu_utilization),
      description: `${clusterStatus.total_gpus} GPUs total`,
      icon: Activity,
      color:
        clusterStatus.avg_gpu_utilization > 80
          ? 'text-red-500'
          : clusterStatus.avg_gpu_utilization > 50
            ? 'text-yellow-500'
            : 'text-green-500',
    },
    {
      title: 'Avg GPU Memory',
      value: formatPercentage(clusterStatus.avg_gpu_memory_utilization || 0),
      description: 'Memory utilization',
      icon: HardDrive,
      color:
        (clusterStatus.avg_gpu_memory_utilization || 0) > 80
          ? 'text-red-500'
          : (clusterStatus.avg_gpu_memory_utilization || 0) > 50
            ? 'text-yellow-500'
            : 'text-green-500',
    },
    {
      title: 'Avg GPU Temp',
      value: clusterStatus.avg_gpu_temperature ? `${clusterStatus.avg_gpu_temperature.toFixed(0)}°C` : 'N/A',
      description: 'Cluster average',
      icon: Thermometer,
      color:
        (clusterStatus.avg_gpu_temperature || 0) > 80
          ? 'text-red-500'
          : (clusterStatus.avg_gpu_temperature || 0) > 70
            ? 'text-orange-500'
            : (clusterStatus.avg_gpu_temperature || 0) > 60
            ? 'text-yellow-500'
            : 'text-green-500',
    },
  ]

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-3xl font-bold tracking-tight">Cluster Overview</h2>
        <div className="flex items-center gap-2">
          <div
            className={`h-2 w-2 rounded-full ${
              isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
            }`}
          />
          <span className="text-sm text-muted-foreground">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {stats.map((stat) => {
          const Icon = stat.icon
          return (
            <Card key={stat.title}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
                <Icon className={`h-4 w-4 ${stat.color}`} />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stat.value}</div>
                <p className="text-xs text-muted-foreground">{stat.description}</p>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
