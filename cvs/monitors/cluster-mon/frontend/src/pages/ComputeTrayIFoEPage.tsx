/**
 * ComputeTrayIFoEPage — Compute → GPUs → IFoE Ports
 *
 * Two tabs:
 *   Port Status   — afmctl show port (all trays, all ports, one table)
 *   Statistics    — afmctl show port statistics mac/fec/ifcp/pfc --json
 *                   Each stat type in its own DataTable, all trays combined.
 */

import { useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { useIFoEData, PPODVPODData } from '@/hooks/useIFoEData'
import {
  DonutChart, GroupedBarChart, VerticalBarChart, COLORS, shortName, compactNum,
} from '@/components/charts/ClusterCharts'
import { ExpandableChartCard } from '@/components/charts/ChartModal'

// ---------------------------------------------------------------------------
// IFoE Station Usability Map
// ---------------------------------------------------------------------------

const STATION_COUNT = 18

function portDotColor(portUp: boolean, stationGloballyUsable: boolean): string {
  if (!portUp)               return '#ef4444'  // red   — port down
  if (stationGloballyUsable) return '#22c55e'  // green — up & globally matched across all trays
  return '#f59e0b'                              // amber — up but not matched everywhere
}

// ---------------------------------------------------------------------------
// Bitmap-based port coloring (from lane_en_bitmap sysfs)
// Each hex digit represents one station (2 ports):
//   f = both UP, 0 = both DOWN, c = P1 UP/P0 DOWN, 3 = P0 UP/P1 DOWN
// ---------------------------------------------------------------------------

interface BitmapPortColors {
  port0: string
  port1: string
}

function getBitmapColor(hexDigit: string): BitmapPortColors {
  const digit = hexDigit.toLowerCase()
  switch (digit) {
    case 'f': return { port0: '#22c55e', port1: '#22c55e' }  // Both green
    case '0': return { port0: '#ef4444', port1: '#ef4444' }  // Both red
    case 'c': return { port0: '#ef4444', port1: '#22c55e' }  // P0 down, P1 up (green)
    case '3': return { port0: '#22c55e', port1: '#ef4444' }  // P0 up (green), P1 down
    // Partial states (other hex values)
    case '7': return { port0: '#22c55e', port1: '#f59e0b' }  // P0 full, P1 partial
    case 'b': return { port0: '#f59e0b', port1: '#22c55e' }  // P0 partial, P1 full
    case 'e': return { port0: '#f59e0b', port1: '#22c55e' }  // P0 partial, P1 full
    case 'd': return { port0: '#22c55e', port1: '#f59e0b' }  // P0 full, P1 partial
    default:  return { port0: '#d1d5db', port1: '#d1d5db' }  // Gray (unknown/no data)
  }
}

// ---------------------------------------------------------------------------
// PPOD/VPOD Topology View
// ---------------------------------------------------------------------------

function PPODTopologyView({
  ppodVpodData,
  computePorts,
}: {
  ppodVpodData: Record<string, PPODVPODData>
  computePorts: Record<string, Record<string, any>[]>
}) {
  const trays = Object.keys(ppodVpodData).sort()
  if (!trays.length) return null

  // Group trays by PPOD ID
  const ppodGroups: Record<string, string[]> = {}
  const notAdmitted: string[] = []

  for (const tray of trays) {
    const data = ppodVpodData[tray]
    const ppodId = data?.ppod_id
    if (ppodId && ppodId !== 'N/A') {
      if (!ppodGroups[ppodId]) ppodGroups[ppodId] = []
      ppodGroups[ppodId].push(tray)
    } else {
      notAdmitted.push(tray)
    }
  }

  // Within each PPOD, group trays by VPOD ID
  const buildVPODGroups = (traysInPPOD: string[]): Record<number | 'none', string[]> => {
    const groups: Record<number | 'none', string[]> = {}
    for (const tray of traysInPPOD) {
      const data = ppodVpodData[tray]
      // Take first VPOD ID (all GPUs on a tray typically share the same VPOD)
      const vpodId = data?.vpod_ids?.[0]
      const key = vpodId !== undefined && vpodId >= 0 ? vpodId : 'none'
      if (!groups[key]) groups[key] = []
      groups[key].push(tray)
    }
    return groups
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          PPOD/VPOD Topology
          <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
            Physical and Virtual Pod structure (AFM-admitted trays)
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Legend */}
        <div className="flex gap-5 text-xs text-gray-500 dark:text-gray-400 flex-wrap">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-blue-500" />
            <span>PPOD (Physical Pod / Rack)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-green-500" />
            <span>VPOD (Virtual Pod / Partition)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-gray-400" />
            <span>Not AFM-Admitted</span>
          </div>
        </div>

        {/* PPOD cards */}
        {Object.entries(ppodGroups).map(([ppodId, traysInPPOD]) => {
          const vpodGroups = buildVPODGroups(traysInPPOD)
          return (
            <div key={ppodId} className="border-2 border-blue-300 dark:border-blue-700 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-3 flex-wrap">
                <div className="w-3 h-3 rounded bg-blue-500 flex-shrink-0" />
                <span className="text-sm font-semibold text-blue-700 dark:text-blue-300">
                  PPOD: <span className="font-mono text-xs break-all">{ppodId}</span>
                </span>
                <span className="text-xs text-gray-500">
                  ({traysInPPOD.length} tray{traysInPPOD.length > 1 ? 's' : ''})
                </span>
              </div>

              <div className="space-y-2 ml-4">
                {Object.entries(vpodGroups)
                  .sort(([a], [b]) => (a === 'none' ? 1 : b === 'none' ? -1 : Number(a) - Number(b)))
                  .map(([vpodId, traysInVPOD]) => (
                    <div key={vpodId} className="border border-green-300 dark:border-green-700 rounded-lg p-2 bg-green-50 dark:bg-green-900/20">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-2.5 h-2.5 rounded bg-green-500" />
                        <span className="text-xs font-semibold text-green-700 dark:text-green-300">
                          {vpodId === 'none' ? 'No VPOD' : `VPOD ${vpodId}`}
                        </span>
                      </div>
                      <div className="space-y-1 ml-3">
                        {traysInVPOD.map(tray => {
                          const data = ppodVpodData[tray]
                          const portCount = (computePorts[tray] ?? []).length
                          const hostname = data?.hostname
                          return (
                            <div key={tray} className="flex items-center gap-3 text-xs flex-wrap">
                              <span className="font-mono text-gray-700 dark:text-gray-300">
                                {tray}
                                {hostname && <span className="text-gray-500 ml-1">({hostname})</span>}
                              </span>
                              {data?.local_accels?.length > 0 && (
                                <span className="text-gray-500">
                                  Accels: <span className="font-mono">{data.local_accels.join(', ')}</span>
                                </span>
                              )}
                              <span className="text-gray-400">
                                {portCount} port{portCount !== 1 ? 's' : ''}
                              </span>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )
        })}

        {/* Not admitted section */}
        {notAdmitted.length > 0 && (
          <div className="border border-gray-300 dark:border-gray-600 rounded-lg p-3 bg-gray-50 dark:bg-gray-800/50">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded bg-gray-400" />
              <span className="text-sm font-semibold text-gray-600 dark:text-gray-400">
                Not AFM-Admitted
              </span>
              <span className="text-xs text-gray-500">
                ({notAdmitted.length} tray{notAdmitted.length > 1 ? 's' : ''})
              </span>
            </div>
            <div className="space-y-1 ml-4">
              {notAdmitted.map(tray => {
                const portCount = (computePorts[tray] ?? []).length
                const hostname = ppodVpodData[tray]?.hostname
                return (
                  <div key={tray} className="flex items-center gap-3 text-xs">
                    <span className="font-mono text-gray-600 dark:text-gray-400">
                      {tray}
                      {hostname && <span className="text-gray-500 ml-1">({hostname})</span>}
                    </span>
                    <span className="text-gray-400">
                      {portCount} port{portCount !== 1 ? 's' : ''} (no AFM data)
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function IFoEStationMap({
  computePorts,
  ppodVpodData,
}: {
  computePorts: Record<string, Record<string, any>[]>
  ppodVpodData?: Record<string, PPODVPODData>
}) {
  const allTrays = Object.keys(computePorts)
  if (!allTrays.length) return null

  // Sort trays by VPOD order: VPOD0 first, then VPOD1, etc., then trays without VPOD at the end
  const sortedTrays = [...allTrays].sort((a, b) => {
    const vpodA = ppodVpodData?.[a]?.vpod_ids?.[0]
    const vpodB = ppodVpodData?.[b]?.vpod_ids?.[0]
    // Trays without VPOD go to the end (use Infinity)
    const valA = vpodA !== undefined && vpodA >= 0 ? vpodA : Infinity
    const valB = vpodB !== undefined && vpodB >= 0 ? vpodB : Infinity
    if (valA !== valB) return valA - valB
    // Within same VPOD, sort by IP/hostname
    return a.localeCompare(b)
  })
  const trays = sortedTrays

  // Check if we have bitmap data available
  const hasBitmapData = ppodVpodData && Object.values(ppodVpodData).some(d => d?.lane_en_bitmaps?.length > 0)

  // Build bdfIndex[tray] = sorted list of unique BDFs → index is GPU number
  const bdfIndex: Record<string, string[]> = {}
  for (const tray of trays) {
    const bdfs = [...new Set((computePorts[tray] ?? []).map((r: any) => r.bdf ?? '').filter(Boolean))].sort()
    bdfIndex[tray] = bdfs
  }
  const gpuCount = Math.max(...trays.map(t => bdfIndex[t].length), 1)

  // portMap[tray][gpuIdx][station][port] = isUp
  type PM = Record<string, Record<number, Record<number, Record<number, boolean>>>>
  const portMap: PM = {}
  for (const tray of trays) {
    portMap[tray] = {}
    for (let g = 0; g < gpuCount; g++) portMap[tray][g] = {}
    for (const row of (computePorts[tray] ?? [])) {
      const bdf = row.bdf ?? ''
      const g   = bdfIndex[tray].indexOf(bdf)
      if (g < 0) continue
      const s = Number(row.station_id ?? 0)
      // port_id is the global port number (0-35 across all stations on one GPU).
      // The local port index within the station is port_id % 2 (0 = P0, 1 = P1).
      const p = Number(row.port_id ?? 0) % 2
      if (!portMap[tray][g][s]) portMap[tray][g][s] = {}
      portMap[tray][g][s][p] = /link_up/i.test(row.link_status ?? '')
    }
  }

  // globallyUsable[gpu][station] = ALL trays have BOTH ports up for that slot
  const globallyUsable: Record<number, Record<number, boolean>> = {}
  for (let g = 0; g < gpuCount; g++) {
    globallyUsable[g] = {}
    for (let s = 0; s < STATION_COUNT; s++) {
      globallyUsable[g][s] = trays.every(tray =>
        portMap[tray][g]?.[s]?.[0] === true &&
        portMap[tray][g]?.[s]?.[1] === true
      )
    }
  }

  const usable = Object.values(globallyUsable).flatMap(g => Object.values(g)).filter(Boolean).length
  const total  = gpuCount * STATION_COUNT

  // Get bitmap hex digit for a specific GPU and station
  const getBitmapDigit = (tray: string, gpuIdx: number, stationIdx: number): string | null => {
    const bitmaps = ppodVpodData?.[tray]?.lane_en_bitmaps ?? []
    const bitmap = bitmaps[gpuIdx]
    if (!bitmap) return null
    // bitmap is a hex string where each digit represents one station
    // Stations are numbered 0-17, so we need to reverse-index the hex string
    // The bitmap is typically written MSB first, so station 0 is the rightmost digit
    const charIdx = bitmap.length - 1 - stationIdx
    if (charIdx < 0 || charIdx >= bitmap.length) return null
    return bitmap[charIdx]
  }

  // Get VPOD ID for a tray (first one if multiple)
  const getVPODId = (tray: string): number | null => {
    const vpodIds = ppodVpodData?.[tray]?.vpod_ids ?? []
    return vpodIds.length > 0 ? vpodIds[0] : null
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          IFoE Station Usability Map
          <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
            {trays.length} tray(s) · {usable}/{total} station slots globally usable
            {hasBitmapData && ' · bitmap-based coloring'}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Legend */}
        <div className="flex gap-5 text-xs text-gray-500 dark:text-gray-400 flex-wrap">
          {hasBitmapData ? (
            // Bitmap-based legend
            <>
              {[
                { color: '#22c55e', label: 'Both ports enabled (bitmap = F)' },
                { color: '#f59e0b', label: 'One port enabled (bitmap = C or 3)' },
                { color: '#ef4444', label: 'Both ports disabled (bitmap = 0)' },
                { color: '#d1d5db', label: 'No bitmap data' },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-1.5">
                  <div style={{ width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0 }} />
                  <span>{label}</span>
                </div>
              ))}
            </>
          ) : (
            // Fallback legend (link_status based)
            <>
              {[
                { color: '#22c55e', label: 'Both ports UP — globally usable (matched on all trays)' },
                { color: '#f59e0b', label: 'Port UP — but station not usable across all trays' },
                { color: '#ef4444', label: 'Port DOWN' },
                { color: '#d1d5db', label: 'No data' },
              ].map(({ color, label }) => (
                <div key={label} className="flex items-center gap-1.5">
                  <div style={{ width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0 }} />
                  <span>{label}</span>
                </div>
              ))}
            </>
          )}
        </div>

        {/* Per-tray grids */}
        <div className="space-y-3">
          {trays.map(tray => {
            const vpodId = getVPODId(tray)
            const hasVpod = vpodId !== null
            const hostname = ppodVpodData?.[tray]?.hostname
            return (
              <div key={tray} className="border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                <div className="flex items-center gap-3 mb-2 flex-wrap">
                  <span className="text-xs font-semibold text-gray-700 dark:text-gray-300 font-mono">
                    {tray}
                    {hostname && <span className="text-gray-500 ml-1">({hostname})</span>}
                  </span>
                  {hasVpod && (
                    <span className="text-xs px-2 py-0.5 rounded bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300 font-medium">
                      VPOD {vpodId}
                    </span>
                  )}
                </div>
                <div className="overflow-x-auto">
                  <table className="border-collapse" style={{ fontSize: 10 }}>
                    <thead>
                      <tr>
                        <th className="pr-3 pb-1 text-gray-400 text-right font-normal whitespace-nowrap" style={{ minWidth: 50 }}>
                          Station →
                        </th>
                        {Array.from({ length: STATION_COUNT }, (_, s) => (
                          <th key={s} className="text-center pb-1 text-gray-400 font-normal" style={{ minWidth: 28, padding: '0 2px' }}>
                            {s}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Array.from({ length: gpuCount }, (_, g) => {
                        const bdf = bdfIndex[tray][g] ?? ''
                        const accelId = ppodVpodData?.[tray]?.accel_ids?.[g]
                        const accelState = ppodVpodData?.[tray]?.accel_states?.[g]
                        // Color code state: green for ready, red for error states, amber for others
                        const stateColor = accelState === 'ready' ? 'text-green-600 dark:text-green-400' :
                                           accelState?.includes('error') ? 'text-red-600 dark:text-red-400' :
                                           'text-amber-600 dark:text-amber-400'
                        return (
                          <tr key={g}>
                            <td className="pr-3 py-1 text-gray-500 dark:text-gray-400 text-right font-medium whitespace-nowrap" title={bdf}>
                              <div className="flex items-center justify-end gap-1">
                                {accelState && (
                                  <span className={`text-[8px] px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-800 ${stateColor}`}>
                                    {accelState}
                                  </span>
                                )}
                                {accelId !== undefined && (
                                  <span className="text-[8px] px-1 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 font-mono">
                                    A{accelId}
                                  </span>
                                )}
                                <span>GPU {g}</span>
                              </div>
                            </td>
                            {Array.from({ length: STATION_COUNT }, (_, s) => {
                              const ports  = portMap[tray][g]?.[s] ?? {}
                              const noData = ports[0] === undefined && ports[1] === undefined
                              const gu     = globallyUsable[g][s]
                              const bitmapDigit = getBitmapDigit(tray, g, s)
                              const useBitmap = hasBitmapData && bitmapDigit !== null
                              const displayDigit = bitmapDigit?.toUpperCase() ?? ''

                              let colors: { port0: string; port1: string }
                              if (useBitmap) {
                                colors = getBitmapColor(bitmapDigit)
                              } else if (noData) {
                                colors = { port0: '#d1d5db', port1: '#d1d5db' }
                              } else {
                                colors = {
                                  port0: portDotColor(ports[0] === true, gu),
                                  port1: portDotColor(ports[1] === true, gu),
                                }
                              }

                              const tip = useBitmap
                                ? `GPU${g} S${s}: bitmap=${displayDigit} (P0=${colors.port0 === '#22c55e' ? 'UP' : colors.port0 === '#ef4444' ? 'DOWN' : 'PARTIAL'}, P1=${colors.port1 === '#22c55e' ? 'UP' : colors.port1 === '#ef4444' ? 'DOWN' : 'PARTIAL'})`
                                : `GPU${g} S${s}: P0=${ports[0]===true?'UP':'DOWN'} P1=${ports[1]===true?'UP':'DOWN'}${gu?' ✓ globally usable':''}`

                              return (
                                <td key={s} style={{ padding: '3px 2px' }} title={tip}>
                                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                                    {/* Show bitmap hex digit above dots if available (uppercase) */}
                                    {useBitmap && (
                                      <div style={{
                                        fontSize: 8,
                                        fontFamily: 'monospace',
                                        color: displayDigit === 'F' ? '#22c55e' : displayDigit === '0' ? '#ef4444' : '#f59e0b',
                                        lineHeight: 1,
                                      }}>
                                        {displayDigit}
                                      </div>
                                    )}
                                    <div style={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                                      <div style={{
                                        width: 9, height: 9, borderRadius: '50%',
                                        background: colors.port0,
                                        flexShrink: 0,
                                      }} />
                                      <div style={{
                                        width: 9, height: 9, borderRadius: '50%',
                                        background: colors.port1,
                                        flexShrink: 0,
                                      }} />
                                    </div>
                                  </div>
                                </td>
                              )
                            })}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}

type MainTab  = 'map' | 'status' | 'stats'
type StatType = 'mac' | 'fec' | 'ifcp' | 'pfc'

const STAT_LABELS: Record<StatType, string> = {
  mac:  'MAC Statistics',
  fec:  'FEC Statistics',
  ifcp: 'IFCP Statistics',
  pfc:  'PFC Statistics',
}

function linkBadge(status: string): string {
  const up   = /link_up/i.test(status)
  const down = /no_phy|no_block|down/i.test(status)
  const cls  = up   ? 'text-green-600 font-semibold' :
               down ? 'text-red-600 font-semibold'   : 'text-gray-500'
  return `<span class="${cls}">${status || '—'}</span>`
}

// Fields that are critical when non-zero — shown red
const CRITICAL_FIELDS = new Set([
  'fec_cw_symbol_errs_uncorrectable',
  'discard_q_rx_dropped_packets',
  'discard_q_rx_dropped_bytes',
  'rx_error', 'tx_error', 'rx_dropped', 'tx_dropped',
])

// Fields that are identifiers (shown monospaced, left-aligned)
const MONO_FIELDS = new Set(['bdf', 'mac_address', 'ifoe_interface', 'name'])

/**
 * Collect ALL unique keys across every row (union), then normalize each row
 * so every row has every key (missing → '' or 0). This prevents DataTables
 * "Requested unknown parameter" warnings when rows have different key sets.
 */
function normalizeRows(rows: Record<string, any>[]): Record<string, any>[] {
  if (!rows.length) return rows
  const allKeys = new Set<string>()
  rows.forEach(r => Object.keys(r).forEach(k => allKeys.add(k)))
  return rows.map(r => {
    const out: Record<string, any> = {}
    allKeys.forEach(k => { out[k] = r[k] !== undefined ? r[k] : '' })
    return out
  })
}

/** Build DataTable columns from the UNION of all row keys (not just first row). */
function buildCols(rows: Record<string, any>[], hostTitle = 'Compute Tray') {
  if (!rows.length) return []
  // Gather all unique non-internal keys across every row
  const keySet = new Set<string>()
  rows.forEach(r => Object.keys(r).filter(k => !k.startsWith('_')).forEach(k => keySet.add(k)))
  const idKeys      = [...keySet].filter(k => ['bdf', 'station', 'port'].includes(k))
  const counterKeys = [...keySet].filter(k => !['bdf', 'station', 'port'].includes(k))
  const orderedKeys = [...idKeys, ...counterKeys]

  // Infer type from first row that has a non-empty value for each key
  const inferType = (k: string) => {
    const val = rows.find(r => r[k] !== '' && r[k] !== undefined)?.[k]
    return typeof val === 'number' ? 'number' : 'string'
  }

  // A counter column is "critical" if it's in the hardcoded set OR its name
  // contains 'error', 'err', or 'bad' — these turn red when > 0.
  const isCritical = (k: string) =>
    CRITICAL_FIELDS.has(k) ||
    /error|err|bad/i.test(k)

  return [
    { title: hostTitle, data: '_host', className: 'dt-left font-medium text-xs' },
    ...orderedKeys.map(k => ({
      title: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      data:  k,
      className: MONO_FIELDS.has(k)
        ? 'dt-left font-mono text-xs'
        : inferType(k) === 'number'
        ? 'dt-right font-mono text-xs'
        : 'dt-left text-xs',
      ...(k === 'link_status' || k === 'fault' ? {
        render: (val: string, _t: string) => linkBadge(val),
      } : isCritical(k) ? {
        render: (val: number, _t: string) =>
          val > 0
            ? `<span class="text-red-600 font-bold">${val.toLocaleString()}</span>`
            : `<span class="text-gray-400">0</span>`,
      } : inferType(k) === 'number' ? {
        render: (val: number, _t: string) =>
          val > 0
            ? val.toLocaleString()
            : `<span class="text-gray-400">0</span>`,
      } : {}),
    })),
  ]
}

export function ComputeTrayIFoEPage() {
  const { data, refreshing, triggerRefresh } = useIFoEData()
  const [mainTab,  setMainTab]  = useState<MainTab>('map')
  const [statType, setStatType] = useState<StatType>('mac')

  // ── Port Status data ─────────────────────────────────────────────────────
  const portRows = Object.entries(data?.compute_ports ?? {}).flatMap(
    ([tray, rows]) => (rows as Record<string, string>[]).map(r => ({
      compute_tray: tray,
      ...r,
    }))
  )
  const upCount   = portRows.filter(r => /link_up/i.test(r.link_status ?? '')).length
  const downCount = portRows.length - upCount
  const trayCount = Object.keys(data?.compute_ports ?? {}).length

  // Normalize port rows so every row has every key (prevents DataTables warnings)
  const normalizedPortRows = normalizeRows(portRows)

  // Collect all unique port keys from ALL rows (not just first row)
  const portKeySet = new Set<string>()
  portRows.forEach(r => Object.keys(r).filter(k => k !== 'compute_tray').forEach(k => portKeySet.add(k)))
  const portExtraKeys = [...portKeySet]

  const portCols = [
    { title: 'Compute Tray', data: 'compute_tray', className: 'dt-left font-medium text-xs' },
    ...portExtraKeys.map(k => ({
      title: k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      data:  k,
      className: (k === 'mac_address' || k === 'bdf' || k === 'name' || k === 'ifoe_interface')
        ? 'dt-left font-mono text-xs' : 'dt-left text-xs',
      ...((k === 'link_status' || k === 'fault') ? {
        render: (val: string, _type: string, _row: any) => linkBadge(val),
      } : {}),
    })),
  ]

  // ── Statistics data ───────────────────────────────────────────────────────
  const statsRaw = data?.compute_port_stats ?? {}

  /** Flatten all trays' stats for a given type into one array. */
  const flatStats = (type: StatType) =>
    Object.entries(statsRaw).flatMap(([host, hostStats]) =>
      (hostStats?.[type] ?? []).map((row: Record<string, any>) => ({
        _host: host,
        ...row,
      }))
    )

  const curStatRows = normalizeRows(flatStats(statType))
  const curStatCols = buildCols(curStatRows)

  // Count how many trays returned data per stat type
  const statCoverage = (type: StatType) =>
    Object.values(statsRaw).filter(h => (h?.[type]?.length ?? 0) > 0).length

  // Count critical FEC errors across all trays for a warning banner
  const fecUncorrectable = flatStats('fec').reduce(
    (sum, r) => sum + (Number(r.fec_cw_symbol_errs_uncorrectable) || 0), 0
  )
  const fecCriticalPorts = flatStats('fec').filter(
    r => (Number(r.fec_cw_symbol_errs_uncorrectable) || 0) > 0
  ).length

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            IFoE Ports — All Compute Trays
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Port status and statistics from afmctl
            {data?.last_updated && ` · ${new Date(data.last_updated).toLocaleString()}`}
          </p>
        </div>
        <button onClick={triggerRefresh} disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium">
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh Now
        </button>
      </div>

      {/* Errors */}
      {data && Object.keys(data.errors).length > 0 && (
        <div className="p-3 bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-700 rounded-lg">
          <p className="text-xs font-semibold text-yellow-800 dark:text-yellow-300 mb-1">Collection errors:</p>
          <ul className="text-xs text-yellow-700 dark:text-yellow-400 space-y-0.5">
            {Object.entries(data.errors).map(([h, e]) => (
              <li key={h}><span className="font-mono">{h}</span>: {String(e)}</li>
            ))}
          </ul>
        </div>
      )}

      {!data && (
        <Card>
          <CardContent className="py-12 text-center text-gray-400">
            <p>No IFoE data. Configure compute trays in Configuration and click Save &amp; Apply.</p>
          </CardContent>
        </Card>
      )}

      {data && (
        <>
          {/* Stats strip */}
          <div className="grid grid-cols-4 gap-3">
            {[
              { label: 'Compute Trays', value: trayCount },
              { label: 'Total Ports',   value: portRows.length },
              { label: 'Link UP',  value: upCount,   color: 'text-green-600 dark:text-green-400' },
              { label: 'Link DOWN',value: downCount, color: downCount > 0 ? 'text-red-600 dark:text-red-400' : '' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 text-center">
                <div className={`text-xl font-bold ${(color as string) || 'text-gray-900 dark:text-gray-100'}`}>{value}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400">{label}</div>
              </div>
            ))}
          </div>

          {/* ── FEC Error chart (shown when stats tab active and data available) ── */}
          {(() => {
            const fecRows = flatStats('fec')
            const errPorts = fecRows
              .filter(r => (Number(r.fec_cw_symbol_errs_uncorrectable) || 0) > 0)
              .map(r => ({
                name: `${shortName(r._host ?? '')}\n${r.bdf ?? ''}\nS${r.station}/P${r.port}`,
                errors: Number(r.fec_cw_symbol_errs_uncorrectable) || 0,
                _color: COLORS.red,
              }))
              .sort((a, b) => b.errors - a.errors)
              .slice(0, 20)
            if (!errPorts.length) return null
            return (
              <ExpandableChartCard
                title={`⚠ FEC Uncorrectable Errors — Top ${errPorts.length} Ports`}
                small={<VerticalBarChart data={errPorts} dataKey="errors" color={COLORS.red} xKey="name" height={200} unit=" err"/>}
                large={<VerticalBarChart data={errPorts} dataKey="errors" color={COLORS.red} xKey="name" height={480} unit=" err"/>}
              />
            )
          })()}

          {/* Main tabs */}
          <div className="flex gap-1 border-b border-gray-200 dark:border-gray-700">
            {([
              { key: 'map',    label: 'Station Map' },
              { key: 'status', label: 'Port Status' },
              { key: 'stats',  label: 'Statistics' },
            ] as { key: MainTab; label: string }[]).map(({ key, label }) => (
              <button key={key} onClick={() => setMainTab(key)}
                className={`px-5 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                  mainTab === key
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                }`}>
                {label}
              </button>
            ))}
          </div>

          {/* Station Map tab */}
          {mainTab === 'map' && (
            <>
              {/* PPOD/VPOD Topology View (only if data is available) */}
              {data?.ppod_vpod && Object.keys(data.ppod_vpod).length > 0 && (
                <PPODTopologyView
                  ppodVpodData={data.ppod_vpod as Record<string, PPODVPODData>}
                  computePorts={data?.compute_ports ?? {}}
                />
              )}
              <IFoEStationMap
                computePorts={data?.compute_ports ?? {}}
                ppodVpodData={data?.ppod_vpod as Record<string, PPODVPODData> | undefined}
              />
            </>
          )}

          {/* Port Status tab */}
          {mainTab === 'status' && (
            <div className="space-y-4">
              {/* Link Status Charts */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <ExpandableChartCard title="Port Link Status"
                  small={<div className="flex justify-center"><DonutChart title="" data={[{name:'Link UP',value:upCount,color:COLORS.green},{name:'Link DOWN',value:downCount,color:COLORS.red}]} centerLabel={`${upCount}/${portRows.length}`} centerSub="ports UP" size={170}/></div>}
                  large={<div className="flex justify-center pt-8"><DonutChart title="" data={[{name:'Link UP',value:upCount,color:COLORS.green},{name:'Link DOWN',value:downCount,color:COLORS.red}]} centerLabel={`${upCount}/${portRows.length}`} centerSub="ports UP" size={340}/></div>}
                />
                <ExpandableChartCard title="Link Status by Compute Tray" className="md:col-span-2"
                  small={<GroupedBarChart data={Object.entries(data?.compute_ports??{}).map(([tray,ports])=>({name:tray,UP:(ports as any[]).filter(p=>/link_up/i.test(p.link_status??'')).length,DOWN:(ports as any[]).filter(p=>!/link_up/i.test(p.link_status??'')).length}))} keys={['UP','DOWN']} colors={[COLORS.green,COLORS.red]} height={180}/>}
                  large={<GroupedBarChart data={Object.entries(data?.compute_ports??{}).map(([tray,ports])=>({name:tray,UP:(ports as any[]).filter(p=>/link_up/i.test(p.link_status??'')).length,DOWN:(ports as any[]).filter(p=>!/link_up/i.test(p.link_status??'')).length}))} keys={['UP','DOWN']} colors={[COLORS.green,COLORS.red]} height={450}/>}
                />
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    Port Status — All Compute Trays
                    <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                      {trayCount} tray(s) · {portRows.length} port(s) · afmctl show port
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {portRows.length === 0 ? (
                    <p className="text-sm text-gray-400 italic py-4">
                      No port data. Nodes with afmctl appear here after detection.
                    </p>
                  ) : (
                  <CustomDataTable
                    columns={portCols}
                    data={normalizedPortRows}
                    defaultPageLength={50}
                    pageLengthOptions={[50, 100, 500]}
                    scrollX={true}
                  />
                  )}
                </CardContent>
              </Card>
            </div>
          )}

          {/* Statistics tab */}
          {mainTab === 'stats' && (
            <div className="space-y-4">
              {/* FEC warning banner */}
              {fecCriticalPorts > 0 && (
                <div className="p-3 bg-red-50 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded-lg">
                  <p className="text-sm font-semibold text-red-800 dark:text-red-300">
                    ⚠ FEC Uncorrectable Errors Detected
                  </p>
                  <p className="text-xs text-red-700 dark:text-red-400 mt-0.5">
                    {fecCriticalPorts} port(s) have uncorrectable FEC codewords
                    (total: {fecUncorrectable.toLocaleString()}).
                    This indicates signal integrity issues — check cable/transceiver health.
                  </p>
                </div>
              )}

              {/* Stat type selector */}
              <div className="flex gap-2 flex-wrap">
                {(Object.keys(STAT_LABELS) as StatType[]).map(t => (
                  <button key={t} onClick={() => setStatType(t)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium border transition-colors ${
                      statType === t
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 border-gray-300 dark:border-gray-600 hover:border-blue-400'
                    }`}>
                    {STAT_LABELS[t]}
                    <span className={`ml-1.5 text-xs px-1.5 py-0.5 rounded-full ${
                      statType === t ? 'bg-blue-500' : 'bg-gray-200 dark:bg-gray-700'
                    }`}>
                      {statCoverage(t)}
                    </span>
                  </button>
                ))}
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    {STAT_LABELS[statType]}
                    <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                      {statCoverage(statType)} tray(s) · afmctl show port statistics {statType} --json
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {curStatRows.length === 0 ? (
                    <div className="py-8 text-center text-gray-400">
                      <p className="font-medium">No {statType.toUpperCase()} statistics data.</p>
                      <p className="text-xs mt-1">
                        {statType === 'ifcp'
                          ? 'IFCP statistics return "Function not implemented" on current firmware — not supported.'
                          : 'Command may not be supported on this firmware version, or no compute trays are connected.'}
                      </p>
                    </div>
                  ) : (
                    <CustomDataTable
                      key={statType}
                      columns={curStatCols}
                      data={curStatRows}
                      defaultPageLength={50}
                      pageLengthOptions={[50, 100, 500]}
                      scrollX={true}
                    />
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </>
      )}
    </div>
  )
}
