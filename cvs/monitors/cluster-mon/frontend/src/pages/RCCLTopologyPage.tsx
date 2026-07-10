import { useEffect, useState, useCallback } from 'react'
import { RefreshCw, Share2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { api } from '@/services/api'

interface Peer {
  addr: string
  pid: number
  cuda_devs: number
  nvml_devs: number
  is_dead: boolean
}

interface TopologyData {
  state: string
  peers?: Peer[]
  dead_peers?: string[]
  job_summary?: {
    total_nodes: number
    total_gpus: number
  }
}

export function RCCLTopologyPage() {
  const [data, setData] = useState<TopologyData | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchData = useCallback(async () => {
    try {
      setLoading(true)
      const status = await api.getRCCLStatus() as TopologyData
      setData(status)
    } catch {
      // silently handle
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 15000)
    return () => clearInterval(interval)
  }, [fetchData])

  const livePeers = data?.peers?.filter((p) => !p.is_dead) || []
  const deadPeers = data?.peers?.filter((p) => p.is_dead) || []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">RAS Topology</h1>
          <p className="text-sm text-gray-500 mt-1">RCCL peer mesh connectivity view</p>
        </div>
        <button
          onClick={fetchData}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          disabled={loading}
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-gray-900">{data?.job_summary?.total_nodes || 0}</p>
            <p className="text-sm text-gray-500">Nodes</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-gray-900">{data?.job_summary?.total_gpus || 0}</p>
            <p className="text-sm text-gray-500">GPUs</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className="text-3xl font-bold text-green-600">{livePeers.length}</p>
            <p className="text-sm text-gray-500">Live Peers</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <p className={`text-3xl font-bold ${deadPeers.length > 0 ? 'text-red-600' : 'text-gray-400'}`}>
              {deadPeers.length}
            </p>
            <p className="text-sm text-gray-500">Dead Peers</p>
          </CardContent>
        </Card>
      </div>

      {/* Peer Mesh Grid */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Share2 className="h-5 w-5" />
            Peer Mesh
          </CardTitle>
        </CardHeader>
        <CardContent>
          {(data?.peers?.length || 0) === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <Share2 className="h-12 w-12 mx-auto mb-4 opacity-30" />
              <p className="font-medium text-gray-500">Peer mesh not supported in RCCL v2.28.3</p>
              <p className="text-sm mt-1">
                Per-peer connectivity data is not included in the rcclras text output for this version.
                Support is planned with the NCCL Inspector integration (rcclras v2.29+).
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {data?.peers?.map((peer) => (
                <div
                  key={`${peer.addr}-${peer.pid}`}
                  className={`p-3 rounded-lg border-2 text-center transition-colors ${
                    peer.is_dead
                      ? 'border-red-300 bg-red-50'
                      : 'border-green-300 bg-green-50'
                  }`}
                >
                  <p className="font-mono text-xs font-medium truncate" title={peer.addr}>
                    {peer.addr}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">PID: {peer.pid}</p>
                  <p className="text-xs text-gray-500">GPUs: {peer.cuda_devs}</p>
                  <span
                    className={`inline-block mt-2 px-2 py-0.5 rounded-full text-xs font-medium ${
                      peer.is_dead
                        ? 'bg-red-200 text-red-700'
                        : 'bg-green-200 text-green-700'
                    }`}
                  >
                    {peer.is_dead ? 'DEAD' : 'ALIVE'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Dead peers detail */}
      {data?.dead_peers && data.dead_peers.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-red-600">Dead Peer Addresses</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {data.dead_peers.map((addr) => (
                <code key={addr} className="px-3 py-1 bg-red-100 text-red-700 rounded text-sm">
                  {addr}
                </code>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
