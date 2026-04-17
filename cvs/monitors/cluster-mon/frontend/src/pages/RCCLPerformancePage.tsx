import { useEffect, useState, useCallback, useRef } from 'react'
import { RefreshCw, Zap, TrendingDown, BarChart3, Clock, ChevronDown, ChevronRight, Info, Filter } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { api } from '@/services/api'

interface KernelChannel {
  channel_id: number
  kernel_start_sn: number | null
  kernel_stop_sn: number | null
  kernel_record_sn: number | null
  kernel_start_ts: number | null
  kernel_stop_ts: number | null
  kernel_record_ts: number | null
}

interface EventTrace {
  coll_start_sn: number | null
  coll_stop_sn: number | null
  coll_start_ts: number | null
  coll_stop_ts: number | null
  channels: KernelChannel[]
}

interface CollPerf {
  timestamp: number
  comm_hash: string
  rank: number
  nranks: number
  nnodes: number
  hostname: string
  pid: number
  collective: string
  sequence_num: number
  msg_size_bytes: number
  exec_time_us: number
  timing_source: string
  algo_bw_gbps: number
  bus_bw_gbps: number
  event_trace: EventTrace | null
}

interface InspectorSnapshot {
  timestamp: number
  records: CollPerf[]
  avg_bus_bw_gbps: number | null
  min_bus_bw_gbps: number | null
  max_bus_bw_gbps: number | null
  slowest_rank: number | null
  collective_breakdown: Record<string, number>
}

const POLL_INTERVAL_MS = 15000

function fmt(n: number | null | undefined, decimals = 1): string {
  if (n == null) return '—'
  return n.toFixed(decimals)
}

function fmtBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) return (bytes / (1024 ** 3)).toFixed(1) + ' GB'
  if (bytes >= 1024 * 1024) return (bytes / (1024 ** 2)).toFixed(1) + ' MB'
  if (bytes >= 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return bytes + ' B'
}

function BwBar({ value, max, slowest }: { value: number; max: number; slowest: boolean }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  const color = slowest ? 'bg-red-500' : value / max >= 0.8 ? 'bg-green-500' : 'bg-yellow-500'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-200 rounded-full h-2">
        <div className={`${color} h-2 rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-600 w-16 text-right">{fmt(value)} GB/s</span>
    </div>
  )
}

function ChannelRows({ et }: { et: EventTrace }) {
  return (
    <div className="bg-gray-50 border-t border-gray-100 px-4 py-2">
      <p className="text-xs font-semibold text-gray-500 mb-2 uppercase tracking-wide">
        Per-channel kernel trace ({et.channels.length} channel{et.channels.length !== 1 ? 's' : ''})
      </p>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-gray-400">
            <th className="pb-1 pr-4">channel_idx</th>
            <th className="pb-1 pr-4">start_event</th>
            <th className="pb-1 pr-4">stop_event</th>
            <th className="pb-1 pr-4">record_event</th>
            <th className="pb-1 pr-4">start_timestamp</th>
            <th className="pb-1 pr-4">end_timestamp</th>
            <th className="pb-1 pr-4">duration (µs)</th>
          </tr>
        </thead>
        <tbody>
          {et.channels.map((ch) => {
            // start_ts and stop_ts are CPU epoch timestamps in µs (from inspectorGetTime).
            // Diff is already in µs — no clock-tick conversion needed.
            const diff = ch.kernel_start_ts != null && ch.kernel_stop_ts != null
              ? ch.kernel_stop_ts - ch.kernel_start_ts
              : null
            return (
              <tr key={ch.channel_id} className="border-t border-gray-100">
                <td className="py-1 pr-4 font-mono font-bold">{ch.channel_id}</td>
                <td className="py-1 pr-4 font-mono">{ch.kernel_start_sn ?? '—'}</td>
                <td className="py-1 pr-4 font-mono">{ch.kernel_stop_sn ?? '—'}</td>
                <td className="py-1 pr-4 font-mono">{ch.kernel_record_sn ?? '—'}</td>
                <td className="py-1 pr-4 font-mono text-gray-500">{ch.kernel_start_ts ?? '—'}</td>
                <td className="py-1 pr-4 font-mono text-gray-500">{ch.kernel_stop_ts ?? '—'}</td>
                <td className="py-1 pr-4 font-mono">{diff ?? '—'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export function RCCLPerformancePage() {
  const [snapshot, setSnapshot] = useState<InspectorSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastFetch, setLastFetch] = useState<number | null>(null)
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set())
  const [selectedPids, setSelectedPids] = useState<Set<number>>(new Set())
  const [pidDropdownOpen, setPidDropdownOpen] = useState(false)
  const pidDropdownRef = useRef<HTMLDivElement>(null)
  const [fetchProgress, setFetchProgress] = useState<number | null>(null)
  const progressTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const startProgress = useCallback(() => {
    if (progressTimerRef.current) clearInterval(progressTimerRef.current)
    setFetchProgress(0)
    let current = 0
    progressTimerRef.current = setInterval(() => {
      // Exponential ease-out: fast early, slows near 85% ceiling
      current += (85 - current) * 0.12
      setFetchProgress(current)
    }, 80)
  }, [])

  const completeProgress = useCallback(() => {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current)
      progressTimerRef.current = null
    }
    setFetchProgress(100)
    setTimeout(() => setFetchProgress(null), 350)
  }, [])

  const fetchData = useCallback(async () => {
    startProgress()
    try {
      const data = await api.getRCCLPerformance() as InspectorSnapshot
      setSnapshot(data)
      setError(null)
    } catch (e: any) {
      const msg = e.message || 'Unknown error'
      if (msg.includes('503')) {
        setError('Inspector not active. Ensure rccl.inspector.enabled=true and an RCCL job is running.')
      } else {
        setError(msg)
      }
      setSnapshot(null)
    } finally {
      setLoading(false)
      setLastFetch(Date.now())
      completeProgress()
    }
  }, [startProgress, completeProgress])

  useEffect(() => {
    fetchData()
    const id = setInterval(fetchData, POLL_INTERVAL_MS)
    return () => {
      clearInterval(id)
      if (progressTimerRef.current) clearInterval(progressTimerRef.current)
    }
  }, [fetchData])

  // Reset PID filter when snapshot changes (new job may have different PIDs)
  useEffect(() => {
    setSelectedPids(new Set())
  }, [snapshot?.timestamp])

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (pidDropdownRef.current && !pidDropdownRef.current.contains(e.target as Node)) {
        setPidDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const toggleRow = (key: string) => {
    setExpandedRows(prev => {
      const next = new Set(prev)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })
  }

  const allPids = snapshot
    ? [...new Set(snapshot.records.map(r => r.pid))].sort((a, b) => b - a)
    : []

  const filteredRecords = snapshot
    ? (selectedPids.size === 0 ? snapshot.records : snapshot.records.filter(r => selectedPids.has(r.pid)))
    : []

  const togglePid = (pid: number) => {
    setSelectedPids(prev => {
      const next = new Set(prev)
      next.has(pid) ? next.delete(pid) : next.add(pid)
      return next
    })
  }

  const hasVerbose = snapshot?.records.some(r => r.event_trace !== null) ?? false
  const maxBw = snapshot?.max_bus_bw_gbps ?? 0

  const isFetching = fetchProgress !== null

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">RCCL Performance</h1>
          <p className="text-sm text-gray-500 mt-1">Inspector plugin — per-collective bandwidth & latency</p>
        </div>
        <button
          onClick={fetchData}
          disabled={isFetching}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <RefreshCw className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Progress bar — visible during fetch */}
      <div className="h-1 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-blue-500 rounded-full transition-all ease-out"
          style={{
            width: fetchProgress != null ? `${fetchProgress}%` : '0%',
            transitionDuration: fetchProgress === 100 ? '150ms' : '80ms',
            opacity: fetchProgress != null ? 1 : 0,
          }}
        />
      </div>

      {loading && (
        <div className="text-center py-12 text-gray-500">Loading Inspector data...</div>
      )}

      {error && !loading && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-yellow-800 text-sm">
          {error}
        </div>
      )}

      {snapshot && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <Zap className="h-8 w-8 text-blue-500" />
                  <div>
                    <p className="text-xs text-gray-500">Avg Bus BW</p>
                    <p className="text-2xl font-bold text-gray-900">{fmt(snapshot.avg_bus_bw_gbps)}</p>
                    <p className="text-xs text-gray-400">GB/s</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <TrendingDown className="h-8 w-8 text-red-500" />
                  <div>
                    <p className="text-xs text-gray-500">Min Bus BW</p>
                    <p className="text-2xl font-bold text-gray-900">{fmt(snapshot.min_bus_bw_gbps)}</p>
                    <p className="text-xs text-gray-400">GB/s</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <BarChart3 className="h-8 w-8 text-green-500" />
                  <div>
                    <p className="text-xs text-gray-500">Max Bus BW</p>
                    <p className="text-2xl font-bold text-gray-900">{fmt(snapshot.max_bus_bw_gbps)}</p>
                    <p className="text-xs text-gray-400">GB/s</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="pt-4">
                <div className="flex items-center gap-3">
                  <Clock className="h-8 w-8 text-purple-500" />
                  <div>
                    <p className="text-xs text-gray-500">Slowest Rank</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {snapshot.slowest_rank != null ? `Rank ${snapshot.slowest_rank}` : '—'}
                    </p>
                    <p className="text-xs text-gray-400">straggler</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-2 gap-6">
            {/* Collective breakdown */}
            <Card>
              <CardHeader>
                <CardTitle>Collective Breakdown</CardTitle>
              </CardHeader>
              <CardContent>
                {Object.keys(snapshot.collective_breakdown).length === 0 ? (
                  <p className="text-sm text-gray-500">No data</p>
                ) : (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-left text-xs text-gray-500 border-b">
                        <th className="pb-2">Collective</th>
                        <th className="pb-2 text-right">Count</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(snapshot.collective_breakdown)
                        .sort(([, a], [, b]) => b - a)
                        .map(([coll, count]) => (
                          <tr key={coll} className="border-b border-gray-50">
                            <td className="py-2 font-mono text-xs">{coll}</td>
                            <td className="py-2 text-right text-gray-700">{count}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                )}
              </CardContent>
            </Card>

            {/* Snapshot metadata */}
            <Card>
              <CardHeader>
                <CardTitle>Snapshot Info</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Timestamp</span>
                  <span className="font-mono">{new Date(snapshot.timestamp * 1000).toLocaleTimeString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Total Records</span>
                  <span className="font-mono">{snapshot.records.length}</span>
                </div>
                {snapshot.records.length > 0 && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Ranks Reporting</span>
                      <span className="font-mono">{new Set(snapshot.records.map(r => r.rank)).size}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Nodes</span>
                      <span className="font-mono">{new Set(snapshot.records.map(r => r.hostname)).size}</span>
                    </div>
                  </>
                )}
                {lastFetch && (
                  <div className="flex justify-between">
                    <span className="text-gray-500">Last Fetched</span>
                    <span className="font-mono">{new Date(lastFetch).toLocaleTimeString()}</span>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Per-rank bandwidth table */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Per-Rank Bandwidth</CardTitle>
                {/* PID multi-select filter */}
                {allPids.length > 1 && (
                  <div className="relative" ref={pidDropdownRef}>
                    <button
                      onClick={() => setPidDropdownOpen(o => !o)}
                      className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded border transition-colors ${
                        selectedPids.size > 0
                          ? 'border-blue-400 bg-blue-50 text-blue-700'
                          : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-50'
                      }`}
                    >
                      <Filter className="h-3 w-3" />
                      {selectedPids.size === 0
                        ? `All PIDs (${allPids.length})`
                        : `${selectedPids.size} PID${selectedPids.size > 1 ? 's' : ''} selected`}
                      <ChevronDown className="h-3 w-3 ml-0.5" />
                    </button>

                    {pidDropdownOpen && (
                      <div className="absolute right-0 mt-1 w-52 bg-white border border-gray-200 rounded-lg shadow-lg z-10 py-1">
                        <div className="px-3 py-1.5 border-b border-gray-100 flex items-center justify-between">
                          <span className="text-xs font-medium text-gray-500">Filter by PID</span>
                          {selectedPids.size > 0 && (
                            <button
                              onClick={() => setSelectedPids(new Set())}
                              className="text-xs text-blue-600 hover:text-blue-800"
                            >
                              Clear all
                            </button>
                          )}
                        </div>
                        {allPids.map(pid => (
                          <label
                            key={pid}
                            className="flex items-center gap-2 px-3 py-1.5 hover:bg-gray-50 cursor-pointer"
                          >
                            <input
                              type="checkbox"
                              checked={selectedPids.has(pid)}
                              onChange={() => togglePid(pid)}
                              className="h-3 w-3 rounded border-gray-300 text-blue-600"
                            />
                            <span className="text-xs font-mono text-gray-700">PID {pid}</span>
                          </label>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {/* Verbose tip */}
              {!hasVerbose && (
                <div className="flex items-start gap-2 mb-4 p-3 bg-blue-50 border border-blue-100 rounded-lg text-xs text-blue-700">
                  <Info className="h-4 w-4 mt-0.5 shrink-0" />
                  <span>
                    <strong>Tip:</strong> Set <code className="bg-blue-100 px-1 rounded">NCCL_INSPECTOR_DUMP_VERBOSE=1</code> to
                    enable per-channel kernel timing. Click any row to expand channel data.
                  </span>
                </div>
              )}

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-gray-500 border-b">
                      <th className="pb-2 pr-2 w-6"></th>
                      <th className="pb-2 pr-4">Rank</th>
                      <th className="pb-2 pr-4">PID</th>
                      <th className="pb-2 pr-4">Host</th>
                      <th className="pb-2 pr-4">Collective</th>
                      <th className="pb-2 pr-4">Msg Size</th>
                      <th className="pb-2 pr-4">Exec (µs)</th>
                      <th className="pb-2 pr-4">Timing</th>
                      <th className="pb-2 pr-4">Channels</th>
                      <th className="pb-2 w-48">Bus BW</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRecords
                      .slice()
                      .sort((a, b) => a.rank - b.rank || b.pid - a.pid)
                      .map((r) => {
                        const key = `${r.rank}-${r.comm_hash}-${r.pid}`
                        const expanded = expandedRows.has(key)
                        const canExpand = r.event_trace !== null
                        return (
                          <>
                            <tr
                              key={key}
                              className={`border-b border-gray-50 ${canExpand ? 'cursor-pointer hover:bg-gray-50' : 'hover:bg-gray-50'}`}
                              onClick={() => canExpand && toggleRow(key)}
                            >
                              <td className="py-2 pr-2 text-gray-400">
                                {canExpand
                                  ? (expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />)
                                  : <span className="text-gray-200 text-xs">—</span>
                                }
                              </td>
                              <td className="py-2 pr-4 font-mono">
                                <span className={r.rank === snapshot.slowest_rank ? 'text-red-600 font-bold' : ''}>
                                  {r.rank}
                                </span>
                              </td>
                              <td className="py-2 pr-4 font-mono text-xs text-gray-500">{r.pid}</td>
                              <td className="py-2 pr-4 text-gray-600 text-xs">{r.hostname}</td>
                              <td className="py-2 pr-4 font-mono text-xs">{r.collective}</td>
                              <td className="py-2 pr-4 text-xs text-gray-600">{fmtBytes(r.msg_size_bytes)}</td>
                              <td className="py-2 pr-4 text-xs text-gray-600">{r.exec_time_us}</td>
                              <td className="py-2 pr-4 text-xs text-gray-500">{r.timing_source}</td>
                              <td className="py-2 pr-4 text-xs text-gray-500">
                                {r.event_trace ? r.event_trace.channels.length : <span className="text-gray-300">N/A</span>}
                              </td>
                              <td className="py-2 w-48">
                                <BwBar
                                  value={r.bus_bw_gbps}
                                  max={maxBw}
                                  slowest={r.rank === snapshot.slowest_rank}
                                />
                              </td>
                            </tr>
                            {canExpand && expanded && (
                              <tr key={`${key}-channels`}>
                                <td colSpan={10} className="p-0">
                                  <ChannelRows et={r.event_trace!} />
                                </td>
                              </tr>
                            )}
                          </>
                        )
                      })}
                  </tbody>
                </table>
                {filteredRecords.length === 0 && (
                  <p className="text-center text-sm text-gray-500 py-6">
                    {snapshot.records.length === 0
                      ? 'No rank records in this snapshot'
                      : `No records for selected PID${selectedPids.size > 1 ? 's' : ''}`}
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
