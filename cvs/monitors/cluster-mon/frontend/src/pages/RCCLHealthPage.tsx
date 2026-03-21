import { useEffect, useState, useCallback } from 'react'
import { RefreshCw, Activity, AlertTriangle, XCircle, CheckCircle, Radio } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { api } from '@/services/api'

interface RCCLStatus {
  state: string
  timestamp?: number
  job_summary?: {
    total_nodes: number
    total_processes: number
    total_gpus: number
    rccl_version: string
  }
  communicators?: Array<{
    comm_hash: string
    total_ranks: number
    responding_ranks: number
    missing_ranks: number
    health: string
    ranks?: Array<{
      comm_rank: number
      node_addr: string
      pid: number
      cuda_dev: number
    }>
  }>
  dead_peers?: string[]
}

const STATE_CONFIG: Record<string, { label: string; color: string; bg: string; icon: any }> = {
  no_job: { label: 'No RCCL Job Running', color: 'text-gray-500', bg: 'bg-gray-100', icon: Radio },
  unreachable: { label: 'Unreachable', color: 'text-red-600', bg: 'bg-red-50', icon: XCircle },
  healthy: { label: 'Healthy', color: 'text-green-600', bg: 'bg-green-50', icon: CheckCircle },
  degraded: { label: 'Degraded', color: 'text-yellow-600', bg: 'bg-yellow-50', icon: AlertTriangle },
  error: { label: 'Error', color: 'text-red-600', bg: 'bg-red-50', icon: XCircle },
}

export function RCCLHealthPage() {
  const [status, setStatus] = useState<RCCLStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true)
      const data = await api.getRCCLStatus() as RCCLStatus
      setStatus(data)
      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to fetch RCCL status')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 10000) // refresh every 10s
    return () => clearInterval(interval)
  }, [fetchStatus])

  const stateConfig = STATE_CONFIG[status?.state || 'no_job'] || STATE_CONFIG.no_job
  const StateIcon = stateConfig.icon

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">RCCL Health</h1>
          <p className="text-sm text-gray-500 mt-1">Real-time RCCL communicator monitoring via ncclras</p>
        </div>
        <button
          onClick={fetchStatus}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          disabled={loading}
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Job State Banner */}
      <Card>
        <CardContent className="p-6">
          <div className={`flex items-center gap-4 p-4 rounded-lg ${stateConfig.bg}`}>
            <StateIcon className={`h-8 w-8 ${stateConfig.color}`} />
            <div>
              <h2 className={`text-xl font-semibold ${stateConfig.color}`}>
                {stateConfig.label}
              </h2>
              {status?.job_summary && (
                <p className="text-sm text-gray-600 mt-1">
                  {status.job_summary.total_nodes} nodes, {status.job_summary.total_gpus} GPUs,{' '}
                  RCCL {status.job_summary.rccl_version}
                </p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Dead Peers */}
      {status?.dead_peers && status.dead_peers.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-red-600 flex items-center gap-2">
              <XCircle className="h-5 w-5" />
              Dead Peers ({status.dead_peers.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {status.dead_peers.map((peer) => (
                <span key={peer} className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm font-mono">
                  {peer}
                </span>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Communicators */}
      {status?.communicators && status.communicators.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Communicators ({status.communicators.length})
          </h3>
          {status.communicators.map((comm) => {
            const healthColor = comm.health === 'healthy' ? 'green' : comm.health === 'degraded' ? 'yellow' : 'red'
            return (
              <Card key={comm.comm_hash}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span className="font-mono text-sm">{comm.comm_hash}</span>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium bg-${healthColor}-100 text-${healthColor}-700`}>
                      {comm.health}
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <p className="text-2xl font-bold text-gray-900">{comm.total_ranks}</p>
                      <p className="text-xs text-gray-500">Total Ranks</p>
                    </div>
                    <div className="text-center p-3 bg-green-50 rounded-lg">
                      <p className="text-2xl font-bold text-green-600">{comm.responding_ranks}</p>
                      <p className="text-xs text-gray-500">Responding</p>
                    </div>
                    <div className="text-center p-3 bg-red-50 rounded-lg">
                      <p className={`text-2xl font-bold ${comm.missing_ranks > 0 ? 'text-red-600' : 'text-gray-400'}`}>
                        {comm.missing_ranks}
                      </p>
                      <p className="text-xs text-gray-500">Missing</p>
                    </div>
                  </div>

                  {/* Rank grid */}
                  {comm.ranks && comm.ranks.length > 0 && (
                    <div className="overflow-x-auto">
                      <table className="min-w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left py-2 px-3 font-medium text-gray-500">Rank</th>
                            <th className="text-left py-2 px-3 font-medium text-gray-500">Node</th>
                            <th className="text-left py-2 px-3 font-medium text-gray-500">PID</th>
                            <th className="text-left py-2 px-3 font-medium text-gray-500">GPU</th>
                          </tr>
                        </thead>
                        <tbody>
                          {comm.ranks.map((rank) => (
                            <tr key={rank.comm_rank} className="border-b border-gray-100 hover:bg-gray-50">
                              <td className="py-2 px-3 font-mono">{rank.comm_rank}</td>
                              <td className="py-2 px-3 font-mono">{rank.node_addr}</td>
                              <td className="py-2 px-3">{rank.pid}</td>
                              <td className="py-2 px-3">{rank.cuda_dev}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </div>
      )}

      {/* No data message */}
      {!loading && !error && (!status || status.state === 'no_job') && (
        <Card>
          <CardContent className="p-12 text-center">
            <Radio className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-500">No RCCL Job Detected</h3>
            <p className="text-sm text-gray-400 mt-2">
              Start an RCCL job (e.g., rccl-tests) with the RAS subsystem enabled to see health data here.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
