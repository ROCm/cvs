import { X, Cpu, Network, Thermometer, Zap, HardDrive } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from './ui/Card'
import { useClusterStore } from '@/stores/clusterStore'
import { formatPercentage, formatTemperature, formatPower, formatBytes } from '@/utils/format'

export function NodeDetailsModal() {
  const selectedNode = useClusterStore((state) => state.selectedNode)
  const setSelectedNode = useClusterStore((state) => state.setSelectedNode)

  if (!selectedNode) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-background rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
        <div className="sticky top-0 bg-background border-b px-6 py-4 flex items-center justify-between">
          <h2 className="text-2xl font-bold">{selectedNode.hostname}</h2>
          <button
            onClick={() => setSelectedNode(null)}
            className="p-2 hover:bg-muted rounded-lg"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Status */}
          <div className="flex items-center gap-4">
            <div
              className={`h-3 w-3 rounded-full ${
                selectedNode.status === 'reachable' ? 'bg-green-500' : 'bg-red-500'
              }`}
            />
            <span className="font-medium">
              {selectedNode.status === 'reachable' ? 'Online' : 'Offline'}
            </span>
          </div>

          {/* GPUs */}
          {selectedNode.gpus && selectedNode.gpus.length > 0 && (
            <div>
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Cpu className="h-5 w-5" />
                GPUs ({selectedNode.gpus.length})
              </h3>
              <div className="grid gap-4 md:grid-cols-2">
                {selectedNode.gpus.map((gpu) => (
                  <Card key={gpu.id}>
                    <CardHeader>
                      <CardTitle className="text-base">{gpu.id}</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Utilization</span>
                        <span className="font-medium">
                          {formatPercentage(gpu.utilization)}
                        </span>
                      </div>

                      <div className="h-2 w-full bg-muted rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all ${
                            gpu.utilization > 90
                              ? 'bg-red-500'
                              : gpu.utilization > 70
                                ? 'bg-yellow-500'
                                : 'bg-green-500'
                          }`}
                          style={{ width: `${Math.min(gpu.utilization, 100)}%` }}
                        />
                      </div>

                      <div className="flex items-center justify-between text-sm pt-2">
                        <div className="flex items-center gap-1">
                          <HardDrive className="h-3 w-3" />
                          <span className="text-muted-foreground">Memory</span>
                        </div>
                        <span className="font-medium">
                          {formatBytes(gpu.memory_used_mb * 1024 * 1024)} /{' '}
                          {formatBytes(gpu.memory_total_mb * 1024 * 1024)}
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-1">
                          <Thermometer className="h-3 w-3" />
                          <span className="text-muted-foreground">Temperature</span>
                        </div>
                        <span className="font-medium">
                          {formatTemperature(gpu.temperature_c)}
                        </span>
                      </div>

                      {gpu.power_w > 0 && (
                        <div className="flex items-center justify-between text-sm">
                          <div className="flex items-center gap-1">
                            <Zap className="h-3 w-3" />
                            <span className="text-muted-foreground">Power</span>
                          </div>
                          <span className="font-medium">{formatPower(gpu.power_w)}</span>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* NICs */}
          {selectedNode.nics && selectedNode.nics.length > 0 && (
            <div>
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Network className="h-5 w-5" />
                Network Interfaces ({selectedNode.nics.length})
              </h3>
              <div className="grid gap-4 md:grid-cols-2">
                {selectedNode.nics.map((nic) => (
                  <Card key={nic.name}>
                    <CardHeader>
                      <CardTitle className="text-base">{nic.name}</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">State</span>
                        <span
                          className={`font-medium ${
                            nic.state === 'UP' ? 'text-green-500' : 'text-red-500'
                          }`}
                        >
                          {nic.state}
                        </span>
                      </div>

                      <div className="flex justify-between">
                        <span className="text-muted-foreground">MTU</span>
                        <span className="font-medium">{nic.mtu}</span>
                      </div>

                      <div className="flex justify-between">
                        <span className="text-muted-foreground">MAC</span>
                        <span className="font-mono text-xs">{nic.mac_addr}</span>
                      </div>

                      {nic.ipv4_addrs.length > 0 && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">IPv4</span>
                          <span className="font-mono text-xs">{nic.ipv4_addrs[0]}</span>
                        </div>
                      )}

                      {nic.rdma && (
                        <div className="pt-2 border-t">
                          <div className="text-xs font-semibold mb-1">RDMA</div>
                          <div className="flex justify-between text-xs">
                            <span className="text-muted-foreground">Device</span>
                            <span>{nic.rdma.device}</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-muted-foreground">State</span>
                            <span className="text-green-500">{nic.rdma.state}</span>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
