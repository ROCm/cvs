/**
 * ScaleUpOverviewPage — Networks → Scale-up → Overview
 *
 * Shows SONiC platform/system information for ALL switch trays in searchable
 * DataTables (one row per switch per entry).  Includes VLAN and MAC tables.
 */

import { RefreshCw, Server, Thermometer, Wind, Zap, Box } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { DonutChart, HorizontalStackedBar, COLORS } from '@/components/charts/ClusterCharts'
import { ExpandableChartCard } from '@/components/charts/ChartModal'
import { useIFoEData } from '@/hooks/useIFoEData'

// ── helpers ──────────────────────────────────────────────────────────────────

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

function dynamicCols(
  rows: Record<string, any>[],
  monoKeys: string[] = [],
  renderOverrides: Record<string, (v: any) => string> = {},
) {
  if (!rows.length) return []
  const keys = Array.from(new Set(rows.flatMap(r => Object.keys(r))))
  return keys.map(k => {
    const col: any = {
      title: k === 'switch' ? 'Switch'
           : k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      data: k,
      className: (k === 'switch' || monoKeys.includes(k))
        ? 'dt-left font-mono text-xs'
        : 'dt-left text-xs',
    }
    if (renderOverrides[k]) col.render = renderOverrides[k]
    return col
  })
}

function flattenObj(obj: Record<string, any>, prefix = ''): Record<string, string> {
  const result: Record<string, string> = {}
  for (const [k, v] of Object.entries(obj ?? {})) {
    const key = prefix ? `${prefix}_${k}` : k
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      Object.assign(result, flattenObj(v, key))
    } else {
      result[key] = String(v ?? '')
    }
  }
  return result
}

// ── section card ─────────────────────────────────────────────────────────────

function SectionCard({
  title,
  icon: Icon,
  iconColor = 'text-blue-600',
  rows,
  monoKeys,
  renderOverrides,
  emptyMsg = 'No data available.',
}: {
  title: string
  icon?: React.ElementType
  iconColor?: string
  rows: Record<string, any>[]
  monoKeys?: string[]
  renderOverrides?: Record<string, (v: any) => string>
  emptyMsg?: string
}) {
  const norm = normalizeRows(rows)
  const cols = dynamicCols(norm, monoKeys ?? [], renderOverrides ?? {})
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          {Icon && <Icon className={`h-4 w-4 ${iconColor}`} />}
          {title}
          {norm.length > 0 && (
            <span className="ml-1 text-sm font-normal text-gray-500 dark:text-gray-400">
              {norm.length} row{norm.length !== 1 ? 's' : ''}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {norm.length === 0
          ? <p className="text-sm text-gray-400 italic py-2">{emptyMsg}</p>
          : <CustomDataTable columns={cols} data={norm}
              defaultPageLength={25} pageLengthOptions={[25, 50, 100]} scrollX={true} />
        }
      </CardContent>
    </Card>
  )
}

// ── status render helpers ─────────────────────────────────────────────────────

const dockerStatusRender = (v: string) =>
  /unhealthy|exited|dead/i.test(v || '')
    ? `<span class="text-red-600 font-semibold">${v}</span>`
    : /up|healthy|running/i.test(v || '')
    ? `<span class="text-green-600">${v}</span>`
    : v || '—'

const sysStatusRender = (v: string) =>
  /not ready|fail|error|down/i.test(v || '')
    ? `<span class="text-red-600 font-semibold">${v}</span>`
    : /ready|ok|up|running/i.test(v || '')
    ? `<span class="text-green-600 font-semibold">${v}</span>`
    : v || '—'

// ── main page ─────────────────────────────────────────────────────────────────

export function ScaleUpOverviewPage() {
  const { data, refreshing, triggerRefresh } = useIFoEData()

  // Only switch tray hosts — avoid showing compute node errors on this page
  const switchHosts = Object.keys(data?.switch_overview ?? {})
  const switchHostSet = new Set([
    ...Object.keys(data?.switch_vlan ?? {}),
    ...switchHosts,
  ])
  const switchErrors = Object.fromEntries(
    Object.entries(data?.errors ?? {}).filter(([h]) => switchHostSet.has(h))
  )

  // ── flatten all switches into combined rows ──────────────────────────────
  // Platform Summary, System Status, Memory: ONE row per switch with fields as columns

  const platformRows = switchHosts.map(sw => {
    const flat = flattenObj(data!.switch_overview[sw]?.platform_summary ?? {})
    const row: Record<string, any> = { switch: sw }
    Object.entries(flat)
      .filter(([, v]) => v && v !== 'undefined')
      .forEach(([k, v]) => { row[k] = v })
    return row
  }).filter(r => Object.keys(r).length > 1)

  // System Status: combine sw_version + system_status into ONE row per switch
  const systemStatusRows = switchHosts.map(sw => {
    const ver = data!.switch_overview[sw]?.sw_version ?? {}
    const health = data!.switch_overview[sw]?.system_status ?? {}
    const row: Record<string, any> = { switch: sw }
    // Key version fields
    const versionFields: Record<string, string> = {
      'SONiC Software Version': 'sonic_version',
      'SONiC OS Version':       'os_version',
      'Distribution':           'distribution',
      'Kernel':                 'kernel',
      'Platform':               'platform',
      'Build commit':           'build_commit',
      'Build date':             'build_date',
    }
    Object.entries(versionFields).forEach(([label, key]) => {
      if (ver[label]) row[key] = ver[label]
    })
    // Health fields
    Object.entries(health).forEach(([k, v]) => { row[k] = String(v) })
    return row
  }).filter(r => Object.keys(r).length > 1)

  const memoryRows = switchHosts.flatMap(sw => {
    const mem = data!.switch_overview[sw]?.memory ?? {}
    return Object.entries(mem).map(([type, stats]: [string, any]) => ({
      switch: sw,
      type:         type.toUpperCase(),
      total_mb:     stats?.total_mb     ?? 0,
      used_mb:      stats?.used_mb      ?? 0,
      free_mb:      stats?.free_mb      ?? 0,
      available_mb: stats?.available_mb ?? 0,
    }))
  })

  const psuRows = switchHosts.flatMap(sw =>
    (data!.switch_overview[sw]?.psu_status ?? []).map((r: any) => ({ switch: sw, ...r }))
  )

  const fanRows = switchHosts.flatMap(sw =>
    (data!.switch_overview[sw]?.fan_status ?? []).map((r: any) => ({ switch: sw, ...r }))
  )

  const tempRows = switchHosts.flatMap(sw =>
    (data!.switch_overview[sw]?.temperature ?? []).map((r: any) => ({ switch: sw, ...r }))
  )

  const dockerPsRows = switchHosts.flatMap(sw =>
    (data!.switch_overview[sw]?.docker_ps ?? []).map((r: any) => ({ switch: sw, ...r }))
  )

  const dockerStatsRows = switchHosts.flatMap(sw =>
    (data!.switch_overview[sw]?.docker_stats ?? []).map((r: any) => ({ switch: sw, ...r }))
  )

  // ── VLAN and MAC from all switches / both ASICs ──────────────────────────

  const vlanRows = Object.entries(data?.switch_vlan ?? {}).flatMap(([sw, asics]) =>
    Object.entries(asics).flatMap(([asic, rows]) =>
      (rows as Record<string, any>[]).map(r => ({ switch: sw, asic, ...r }))
    )
  )

  const macRows = Object.entries(data?.switch_mac ?? {}).flatMap(([sw, asics]) =>
    Object.entries(asics).flatMap(([asic, rows]) =>
      (rows as Record<string, any>[]).map(r => ({ switch: sw, asic, ...r }))
    )
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Scale-up Fabric — Overview</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            SONiC platform, system and docker status — all switch trays
            {data?.last_updated && ` · ${new Date(data.last_updated).toLocaleString()}`}
          </p>
        </div>
        <button onClick={triggerRefresh} disabled={refreshing}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 text-sm font-medium">
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh Now
        </button>
      </div>

      {/* Credential warning — shown when ALL switch hosts are unreachable */}
      {data && Object.keys(switchErrors).length > 0 && Object.values(switchErrors).every(e => String(e).startsWith('ABORT')) && (
        <div className="p-4 bg-red-50 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded-lg">
          <p className="text-sm font-semibold text-red-800 dark:text-red-300 mb-1">
            Switch SSH credentials not available
          </p>
          <p className="text-xs text-red-700 dark:text-red-400">
            After a container restart, switch passwords must be re-entered.
            Go to <strong>Configuration → Scale-up Switches</strong>, enter the username and password, and click <strong>Save & Apply</strong>.
          </p>
        </div>
      )}
      {/* Partial switch errors */}
      {data && Object.keys(switchErrors).length > 0 && !Object.values(switchErrors).every(e => String(e).startsWith('ABORT')) && (
        <div className="p-3 bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-700 rounded-lg">
          <p className="text-xs font-semibold text-yellow-800 dark:text-yellow-300 mb-1">Switch collection errors:</p>
          <ul className="text-xs text-yellow-700 dark:text-yellow-400 space-y-0.5">
            {Object.entries(switchErrors).map(([h, e]) => (
              <li key={h}><span className="font-mono">{h}</span>: {String(e)}</li>
            ))}
          </ul>
        </div>
      )}

      {!data && (
        <Card><CardContent className="py-12 text-center text-gray-400">
          Configure scale-up switches in Configuration and click Save & Apply.
        </CardContent></Card>
      )}

      {data && switchHosts.length === 0 && (
        <Card><CardContent className="py-12 text-center text-gray-400">
          No switch overview data yet. Data collects on the next poll cycle (5 min).
        </CardContent></Card>
      )}

      {data && switchHosts.length > 0 && (
        <>
          {/* ── Summary charts ── */}
          {(() => {
            // Docker container health across all switches
            const allContainers = dockerPsRows
            const running   = allContainers.filter(r => /up|running/i.test(String(r.Status ?? r.status ?? ''))).length
            const unhealthy = allContainers.filter(r => /unhealthy|exited|dead/i.test(String(r.Status ?? r.status ?? ''))).length
            const other     = allContainers.length - running - unhealthy

            // PSU health across all switches
            const allPsu = psuRows
            const psuOk  = allPsu.filter(r => /ok|good|present/i.test(String(r.Status ?? r.status ?? r.State ?? ''))).length
            const psuBad = allPsu.length - psuOk

            // Fan health
            const allFan = fanRows
            const fanOk  = allFan.filter(r => /ok|good/i.test(String(r.Status ?? r.status ?? ''))).length
            const fanBad = allFan.length - fanOk

            if (!allContainers.length && !allPsu.length) return null
            return (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {allContainers.length > 0 && (
                  <ExpandableChartCard title="Container Health"
                    small={<div className="flex justify-center"><DonutChart title="" data={[
                      {name:'Running',   value: running,   color: COLORS.green},
                      {name:'Unhealthy', value: unhealthy, color: COLORS.red},
                      ...(other > 0 ? [{name:'Other', value: other, color: COLORS.gray}] : []),
                    ]} centerLabel={String(allContainers.length)} centerSub="containers" size={160}/></div>}
                    large={<div className="flex justify-center pt-8"><DonutChart title="" data={[
                      {name:'Running',   value: running,   color: COLORS.green},
                      {name:'Unhealthy', value: unhealthy, color: COLORS.red},
                      ...(other > 0 ? [{name:'Other', value: other, color: COLORS.gray}] : []),
                    ]} centerLabel={String(allContainers.length)} centerSub="containers" size={300}/></div>}
                  />
                )}
                {allPsu.length > 0 && (
                  <ExpandableChartCard title="PSU Health"
                    small={<div className="flex justify-center"><DonutChart title="" data={[
                      {name:'OK',   value: psuOk,  color: COLORS.green},
                      {name:'Fail', value: psuBad, color: COLORS.red},
                    ]} centerLabel={`${psuOk}/${allPsu.length}`} centerSub="PSUs OK" size={160}/></div>}
                    large={<div className="flex justify-center pt-8"><DonutChart title="" data={[
                      {name:'OK',   value: psuOk,  color: COLORS.green},
                      {name:'Fail', value: psuBad, color: COLORS.red},
                    ]} centerLabel={`${psuOk}/${allPsu.length}`} centerSub="PSUs OK" size={300}/></div>}
                  />
                )}
                {allFan.length > 0 && (
                  <ExpandableChartCard title="Fan Health"
                    small={<div className="flex justify-center"><DonutChart title="" data={[
                      {name:'OK',   value: fanOk,  color: COLORS.green},
                      {name:'Fail', value: fanBad, color: COLORS.red},
                    ]} centerLabel={`${fanOk}/${allFan.length}`} centerSub="Fans OK" size={160}/></div>}
                    large={<div className="flex justify-center pt-8"><DonutChart title="" data={[
                      {name:'OK',   value: fanOk,  color: COLORS.green},
                      {name:'Fail', value: fanBad, color: COLORS.red},
                    ]} centerLabel={`${fanOk}/${allFan.length}`} centerSub="Fans OK" size={300}/></div>}
                  />
                )}
              </div>
            )
          })()}

          {/* Platform Summary */}
          <SectionCard title="Platform Summary" icon={Server} iconColor="text-blue-600"
            rows={platformRows} monoKeys={['value']}
            emptyMsg="No platform summary data." />

          {/* System Status — SONiC version + health summary, one row per switch */}
          <SectionCard title="System Status &amp; Version" icon={Server} iconColor="text-green-600"
            rows={systemStatusRows} monoKeys={['switch', 'kernel', 'build_commit', 'sonic_version', 'platform']}
            renderOverrides={{
              'Services Status': sysStatusRender,
              'Hardware Status':  sysStatusRender,
              'System LED':       (v: string) =>
                /blue|green/i.test(v || '') ? `<span class="text-green-600 font-semibold">${v}</span>`
                : /red|yellow|amber/i.test(v || '') ? `<span class="text-red-600 font-semibold">${v}</span>`
                : v || '—',
            }}
            emptyMsg="No system status data." />

          {/* Memory */}
          <SectionCard title="Memory Usage" rows={memoryRows}
            emptyMsg="No memory data." />

          {/* PSU / Fan / Temperature */}
          <SectionCard title="PSU Status" icon={Zap} iconColor="text-yellow-500"
            rows={psuRows} monoKeys={['switch']}
            emptyMsg="No PSU data." />

          <SectionCard title="Fan Status" icon={Wind} iconColor="text-cyan-500"
            rows={fanRows} monoKeys={['switch']}
            emptyMsg="No fan data." />

          <SectionCard title="Temperature" icon={Thermometer} iconColor="text-orange-500"
            rows={tempRows} monoKeys={['switch']}
            emptyMsg="No temperature data." />

          {/* Docker */}
          <SectionCard title="Docker Containers" icon={Box} iconColor="text-purple-600"
            rows={dockerPsRows} monoKeys={['switch', 'Names', 'Image', 'ID']}
            renderOverrides={{ Status: dockerStatusRender, status: dockerStatusRender }}
            emptyMsg="No docker container data." />

          {dockerStatsRows.length > 0 && (
            <SectionCard title="Docker Stats" icon={Box} iconColor="text-indigo-600"
              rows={dockerStatsRows} monoKeys={['switch', 'Name', 'Container']}
              emptyMsg="No docker stats data." />
          )}

          {/* VLAN Configuration */}
          <SectionCard title="VLAN Configuration" rows={normalizeRows(vlanRows)}
            monoKeys={['switch', 'ports']}
            emptyMsg="No VLAN data." />

          {/* MAC Table */}
          <SectionCard title="MAC Table" rows={normalizeRows(macRows)}
            monoKeys={['switch', 'mac_address', 'port']}
            emptyMsg="No MAC table data." />

        </>
      )}
    </div>
  )
}
