/**
 * ScaleUpMetricsPage — Networks → Scale-up → Metrics
 *
 * Interface metrics and counters from SONiC switch trays:
 *   - Interface status
 *   - PFC counters
 *   - Priority-group counters
 *   - Queue counters
 *   - Queue watermarks
 */

import { useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { useIFoEData } from '@/hooks/useIFoEData'

function normalizeRows(rows: Record<string, any>[]): Record<string, any>[] {
  if (!rows.length) return []
  const keys = new Set<string>()
  rows.forEach(r => Object.keys(r).forEach(k => keys.add(k)))
  return rows.map(r => {
    const out: Record<string, any> = {}
    keys.forEach(k => { out[k] = r[k] !== undefined ? r[k] : '' })
    return out
  })
}

function dynamicCols(rows: Record<string, any>[], monoKeys: string[] = []) {
  if (!rows.length) return []
  const keys = Array.from(new Set(rows.flatMap(r => Object.keys(r))))
  const isError = (k: string) => /error|err|drop|discard/i.test(k)
  return keys.map(k => {
    const col: any = {
      title: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      data: k,
      className: monoKeys.includes(k) ? 'dt-left font-mono text-xs' : 'dt-right text-xs',
    }
    if (isError(k)) {
      col.render = (v: any) => {
        const n = Number(v)
        return !isNaN(n) && n > 0
          ? `<span class="text-red-600 font-bold">${v}</span>`
          : `<span class="text-gray-400">${v || 0}</span>`
      }
    }
    return col
  })
}

type MetricTab = 'interfaces' | 'intf_counters' | 'pfc' | 'queue' | 'queue_wm'

const TAB_LABELS: Record<MetricTab, string> = {
  interfaces:    'Interface Status',
  intf_counters: 'Interface Counters',
  pfc:           'PFC Counters',
  queue:         'Queue Counters',
  queue_wm:      'Queue Watermarks',
}

export function ScaleUpMetricsPage() {
  const { data, refreshing, triggerRefresh } = useIFoEData()
  const [tab, setTab] = useState<MetricTab>('interfaces')

  const switchHosts = Object.keys(data?.switch_metrics ?? {})

  const switchHostSet = new Set([
    ...Object.keys(data?.switch_vlan ?? {}),
    ...switchHosts,
  ])
  const switchErrors = Object.fromEntries(
    Object.entries(data?.errors ?? {}).filter(([h]) => switchHostSet.has(h))
  )

  // Flatten all switches into one table per metric type, with a Switch column
  const monoKeys = ['switch', 'interface', 'queue_or_pg', 'port', 'iface']

  const flatRows = (key: keyof SwitchMetrics): Record<string, any>[] => {
    const rows = switchHosts.flatMap(sw =>
      ((data?.switch_metrics?.[sw]?.[key] ?? []) as Record<string, any>[])
        .map(r => ({ switch: sw, ...r }))
    )
    return normalizeRows(rows)
  }

  const getRows = () => {
    switch (tab) {
      case 'interfaces':    return flatRows('interfaces')
      case 'intf_counters': return flatRows('intf_counters')
      case 'pfc':           return flatRows('pfc_counters')
      case 'queue':         return flatRows('queue_counters')
      case 'queue_wm':      return flatRows('queue_wm')
    }
  }

  const rows = getRows()
  const cols = dynamicCols(rows, monoKeys).map(c => ({
    ...c,
    className: monoKeys.includes(c.data) ? 'dt-left font-mono text-xs' : 'dt-right text-xs',
    ...(tab === 'interfaces' && (c.data === 'oper_state' || c.data === 'oper') ? {
      render: (v: string) =>
        /up/i.test(v || '')   ? `<span class="text-green-600 font-semibold">${v}</span>`
        : /down/i.test(v || '') ? `<span class="text-red-600 font-semibold">${v}</span>`
        : v || '—',
    } : {}),
  }))

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Scale-up Fabric — Metrics</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Interface metrics and counter data from SONiC switch trays
            {data?.last_updated && ` · ${new Date(data.last_updated).toLocaleString()}`}
          </p>
        </div>
        <button onClick={triggerRefresh} disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium">
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh Now
        </button>
      </div>

      {data && Object.keys(switchErrors).length > 0 && Object.values(switchErrors).every(e => String(e).startsWith('ABORT')) && (
        <div className="p-4 bg-red-50 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded-lg">
          <p className="text-sm font-semibold text-red-800 dark:text-red-300 mb-1">Switch SSH credentials not available</p>
          <p className="text-xs text-red-700 dark:text-red-400">
            Go to <strong>Configuration → Scale-up Switches</strong>, enter credentials, and click <strong>Save & Apply</strong>.
          </p>
        </div>
      )}

      {!data && (
        <Card><CardContent className="py-12 text-center text-gray-400">
          <p>No switch data. Configure scale-up switches in Configuration and click Save & Apply.</p>
        </CardContent></Card>
      )}

      {data && switchHosts.length === 0 && (
        <Card><CardContent className="py-12 text-center text-gray-400">
          <p>No switch metrics data collected yet. Data collects on the next poll cycle (5 min).</p>
        </CardContent></Card>
      )}

      {data && switchHosts.length > 0 && (
        <>
          {/* Metric type tabs */}
          <div className="flex gap-1 border-b border-gray-200 dark:border-gray-700 flex-wrap">
            {(Object.keys(TAB_LABELS) as MetricTab[]).map(t => (
              <button key={t} onClick={() => setTab(t)}
                className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
                  tab === t
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
                }`}>
                {TAB_LABELS[t]}
              </button>
            ))}
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                {TAB_LABELS[tab]} — All Switch Trays
                {rows.length > 0 && (
                  <span className="ml-2 text-sm font-normal text-gray-500">{rows.length} row(s)</span>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {rows.length === 0 ? (
                <p className="text-sm text-gray-400 italic py-4">
                  No {TAB_LABELS[tab].toLowerCase()} data. Command may not be supported on this SONiC firmware version.
                </p>
              ) : (
                <CustomDataTable
                  key={tab}
                  columns={cols}
                  data={rows}
                  defaultPageLength={50}
                  pageLengthOptions={[50, 100, 500]}
                  scrollX={true}
                />
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
