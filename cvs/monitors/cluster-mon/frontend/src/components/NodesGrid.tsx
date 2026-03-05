import { useEffect } from 'react'
import { Server, Cpu, Thermometer } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card'
import { useClusterStore } from '@/stores/clusterStore'
import { api } from '@/services/api'
import { formatPercentage, formatTemperature, getStatusColor } from '@/utils/format'

export function NodesGrid() {
  const nodes = useClusterStore((state) => state.nodes)
  const setNodes = useClusterStore((state) => state.setNodes)
  const setSelectedNode = useClusterStore((state) => state.setSelectedNode)

  useEffect(() => {
    const fetchNodes = async () => {
      try {
        const nodesData = await api.getNodes()
        setNodes(nodesData)
      } catch (error) {
        console.error('Failed to fetch nodes:', error)
      }
    }

    fetchNodes()
    const interval = setInterval(fetchNodes, 60000)
    return () => clearInterval(interval)
  }, [setNodes])

  const handleNodeClick = async (node: any) => {
    try {
      const nodeDetails = await api.getNodeDetails(node.hostname)
      setSelectedNode(nodeDetails)
    } catch (error) {
      console.error('Failed to fetch node details:', error)
    }
  }

  if (nodes.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <Server className="h-12 w-12 mx-auto mb-4 opacity-50" />
        <p>No nodes available</p>
      </div>
    )
  }

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Cluster Nodes</h2>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {nodes.map((node) => {
          const getStatusColor = () => {
            switch (node.status) {
              case 'healthy': return 'bg-green-500'
              case 'unhealthy': return 'bg-yellow-500'
              case 'unreachable': return 'bg-red-500'
              default: return 'bg-gray-500'
            }
          }

          const getBorderColor = () => {
            switch (node.status) {
              case 'unhealthy': return 'border-yellow-500'
              case 'unreachable': return 'border-red-500 opacity-50'
              default: return 'hover:border-primary'
            }
          }

          return (
            <Card
              key={node.hostname}
              onClick={() => handleNodeClick(node)}
              className={getBorderColor()}
            >
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium truncate">{node.hostname}</CardTitle>
                <div className={`h-2 w-2 rounded-full ${getStatusColor()}`} />
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-1">
                    <Cpu className="h-3 w-3" />
                    <span className="text-muted-foreground">GPUs</span>
                  </div>
                  <span className="font-medium">{node.gpu_count}</span>
                </div>

                {node.status !== 'unreachable' && (
                  <>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Utilization</span>
                      <span
                        className={`font-medium ${getStatusColor(node.avg_gpu_util, 90)}`}
                      >
                        {formatPercentage(node.avg_gpu_util)}
                      </span>
                    </div>

                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-1">
                        <Thermometer className="h-3 w-3" />
                        <span className="text-muted-foreground">Temp</span>
                      </div>
                      <span
                        className={`font-medium ${getStatusColor(node.avg_gpu_temp, 85)}`}
                      >
                        {formatTemperature(node.avg_gpu_temp)}
                      </span>
                    </div>

                    <div className="mt-3">
                      <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-500 ${
                            node.avg_gpu_util > 90
                              ? 'bg-red-500'
                              : node.avg_gpu_util > 70
                                ? 'bg-yellow-500'
                                : 'bg-green-500'
                          }`}
                          style={{ width: `${Math.min(node.avg_gpu_util, 100)}%` }}
                        />
                      </div>
                    </div>
                  </>
                )}

                {node.status === 'unreachable' && (
                  <div className="text-xs text-red-500 mt-2">Unreachable</div>
                )}

                {node.status === 'unhealthy' && node.health_issues && node.health_issues.length > 0 && (
                  <div className="text-xs text-yellow-600 mt-2 truncate" title={node.health_issues.join(', ')}>
                    ⚠ {node.health_issues.length} issue(s)
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
          )
        })}
      </div>
    </div>
  )
}
