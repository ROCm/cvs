import { useEffect, useState, useCallback } from 'react'
import { RefreshCw, Clock, Filter } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { api } from '@/services/api'

interface RCCLEvent {
  timestamp: number
  event_type: string
  source_node?: string
  details?: string
  peer_addr?: string
  type?: string
  step?: number
  loss?: number
  rank?: number
}

const EVENT_COLORS: Record<string, string> = {
  lifecycle: 'bg-blue-100 text-blue-800',
  trace: 'bg-purple-100 text-purple-800',
  training_marker: 'bg-green-100 text-green-800',
  error: 'bg-red-100 text-red-800',
}

export function RCCLTimelinePage() {
  const [events, setEvents] = useState<RCCLEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [timeRange, setTimeRange] = useState(3600) // 1 hour default
  const [filterType, setFilterType] = useState<string>('')

  const fetchEvents = useCallback(async () => {
    try {
      setLoading(true)
      const now = Date.now() / 1000
      const data = await api.getRCCLEvents(
        now - timeRange,
        now,
        filterType || undefined
      ) as RCCLEvent[]
      setEvents(data.sort((a, b) => b.timestamp - a.timestamp))
    } catch {
      // silently handle
    } finally {
      setLoading(false)
    }
  }, [timeRange, filterType])

  useEffect(() => {
    fetchEvents()
    const interval = setInterval(fetchEvents, 10000)
    return () => clearInterval(interval)
  }, [fetchEvents])

  const formatTime = (ts: number) => {
    return new Date(ts * 1000).toLocaleTimeString()
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Event Timeline</h1>
          <p className="text-sm text-gray-500 mt-1">RCCL events, training markers, and lifecycle changes</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Time range selector */}
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(Number(e.target.value))}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white"
          >
            <option value={300}>Last 5 min</option>
            <option value={900}>Last 15 min</option>
            <option value={3600}>Last 1 hour</option>
            <option value={14400}>Last 4 hours</option>
            <option value={86400}>Last 24 hours</option>
          </select>

          {/* Type filter */}
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white"
          >
            <option value="">All types</option>
            <option value="lifecycle">Lifecycle</option>
            <option value="trace">Trace</option>
            <option value="training_marker">Training</option>
          </select>

          <button
            onClick={fetchEvents}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Event count */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center gap-4">
            <Clock className="h-5 w-5 text-gray-400" />
            <span className="text-sm text-gray-600">
              {events.length} events in the selected time range
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Timeline */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Events
          </CardTitle>
        </CardHeader>
        <CardContent>
          {events.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <Clock className="h-12 w-12 mx-auto mb-4 opacity-30" />
              <p>No events in the selected time range.</p>
              <p className="text-sm mt-1">Events appear when RCCL jobs run and produce lifecycle or trace events.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {events.map((event, idx) => {
                const colorClass = EVENT_COLORS[event.event_type || event.type || ''] || 'bg-gray-100 text-gray-800'
                return (
                  <div key={idx} className="flex items-start gap-4 p-3 rounded-lg hover:bg-gray-50 border border-gray-100">
                    {/* Timestamp */}
                    <div className="flex-shrink-0 w-24 text-right">
                      <span className="text-xs font-mono text-gray-500">
                        {formatTime(event.timestamp)}
                      </span>
                    </div>

                    {/* Timeline dot */}
                    <div className="flex-shrink-0 mt-1">
                      <div className="h-3 w-3 rounded-full bg-blue-400 ring-4 ring-blue-100" />
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colorClass}`}>
                          {event.event_type || event.type || 'unknown'}
                        </span>
                        {event.source_node && (
                          <span className="text-xs text-gray-500 font-mono">{event.source_node}</span>
                        )}
                        {event.rank !== undefined && (
                          <span className="text-xs text-gray-500">Rank {event.rank}</span>
                        )}
                      </div>
                      {event.details && (
                        <p className="text-sm text-gray-700">{event.details}</p>
                      )}
                      {event.step !== undefined && (
                        <p className="text-sm text-gray-600">
                          Step {event.step}
                          {event.loss !== undefined && ` | Loss: ${event.loss.toFixed(4)}`}
                        </p>
                      )}
                      {event.peer_addr && (
                        <p className="text-xs text-gray-400 font-mono mt-1">Peer: {event.peer_addr}</p>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
