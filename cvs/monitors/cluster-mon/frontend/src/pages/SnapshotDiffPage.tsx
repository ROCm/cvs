import { useCallback, useEffect, useMemo, useState } from 'react'
import { Camera, GitCompare, Trash2, Loader2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { api } from '@/services/api'

type SnapshotItem = {
  id: string
  captured_at: string
  label: string | null
  degraded: boolean
  failed_nodes: string[]
  categories: Record<string, { nodes: number; devices: number; stats: number }>
}

type SnapshotList = {
  snapshots: SnapshotItem[]
  count: number
  max: number
  in_progress: boolean
}

type DiffRow = {
  severity: string
  category: string
  node: string
  device: string
  stat: string
  before: number | string
  after: number | string
  diff: number
}

type CompareResult = {
  before_id: string
  after_id: string
  before_timestamp: string
  after_timestamp: string
  before_label: string | null
  after_label: string | null
  summary: {
    errors: number
    warnings: number
    threshold_warnings: number
    total_increments: number
  }
  rows: DiffRow[]
  warnings: string[]
}

const CATEGORY_LABELS: Record<string, string> = {
  eth_stats: 'Ethernet',
  rdma_stats: 'RDMA',
  gpu_ras_stats: 'GPU RAS / ECC',
  gpu_pcie_stats: 'GPU PCIe',
}

function snapshotTitle(snap: SnapshotItem) {
  const time = new Date(snap.captured_at).toLocaleString()
  return snap.label ? `${snap.label} (${time})` : time
}

function totalStats(snap: SnapshotItem) {
  return Object.values(snap.categories || {}).reduce((sum, cat) => sum + (cat.stats || 0), 0)
}

function totalNodes(snap: SnapshotItem) {
  const nodes = Object.values(snap.categories || {}).map((cat) => cat.nodes || 0)
  return nodes.length ? Math.max(...nodes) : 0
}

function severityBadgeHtml(severity: string) {
  const styles: Record<string, string> = {
    error: 'background:#fee2e2;color:#991b1b',
    warning: 'background:#fef3c7;color:#92400e',
    threshold_warning: 'background:#ffedd5;color:#9a3412',
    info: 'background:#dbeafe;color:#1e40af',
  }
  const labels: Record<string, string> = {
    error: 'Error',
    warning: 'Warning',
    threshold_warning: 'Threshold',
    info: 'Info',
  }
  const style = styles[severity] || styles.info
  const label = labels[severity] || severity
  return `<span style="${style};border-radius:9999px;padding:2px 8px;font-size:12px;font-weight:600">${label}</span>`
}

export function SnapshotDiffPage() {
  const [list, setList] = useState<SnapshotList | null>(null)
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null)
  const [loading, setLoading] = useState<'capture' | 'diff' | 'list' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [label, setLabel] = useState('')
  const [beforeId, setBeforeId] = useState('')
  const [afterId, setAfterId] = useState('')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [showAllIncrements, setShowAllIncrements] = useState(false)

  const applyList = useCallback((data: SnapshotList, mode: 'init' | 'capture' | 'preserve' = 'preserve') => {
    setList(data)
    const snaps = data.snapshots || []
    const chronological = [...snaps].sort(
      (a, b) => new Date(a.captured_at).getTime() - new Date(b.captured_at).getTime()
    )
    const oldest = chronological[0]?.id || ''
    const newest = chronological[chronological.length - 1]?.id || ''
    const ids = new Set(snaps.map((s) => s.id))

    setBeforeId((prev) => {
      if (mode === 'init' || !prev || !ids.has(prev)) return oldest
      return prev
    })
    setAfterId((prev) => {
      if (mode === 'capture') return newest
      if (mode === 'init' || !prev || !ids.has(prev)) return newest
      return prev
    })
  }, [])

  const fetchList = useCallback(async () => {
    setLoading('list')
    try {
      const data = (await api.listSnapshots()) as SnapshotList
      applyList(data, 'preserve')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load snapshots')
    } finally {
      setLoading(null)
    }
  }, [applyList])

  useEffect(() => {
    fetchList()
  }, [fetchList])

  const handleCapture = async () => {
    setLoading('capture')
    setError(null)
    try {
      await api.captureSnapshot(label.trim() || undefined)
      setLabel('')
      const data = (await api.listSnapshots()) as SnapshotList
      applyList(data, 'capture')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to capture snapshot')
    } finally {
      setLoading(null)
    }
  }

  const handleDiff = async () => {
    if (!beforeId || !afterId || beforeId === afterId) return
    setLoading('diff')
    setError(null)
    try {
      const result = (await api.diffSnapshots(beforeId, afterId)) as CompareResult
      setCompareResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to diff snapshots')
    } finally {
      setLoading(null)
    }
  }

  const handleDelete = async (id: string) => {
    const isSelected = id === beforeId || id === afterId
    if (isSelected && !window.confirm('This snapshot is selected for diff. Delete it anyway?')) {
      return
    }
    setError(null)
    try {
      await api.deleteSnapshot(id)
      if (compareResult && (compareResult.before_id === id || compareResult.after_id === id)) {
        setCompareResult(null)
      }
      const data = (await api.listSnapshots()) as SnapshotList
      applyList(data, 'preserve')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete snapshot')
    }
  }

  const handleClearAll = async () => {
    if (!window.confirm('Delete all captured snapshots?')) return
    setError(null)
    try {
      await api.deleteAllSnapshots()
      setCompareResult(null)
      const data = (await api.listSnapshots()) as SnapshotList
      applyList(data, 'init')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to clear snapshots')
    }
  }

  const filteredRows = useMemo(() => {
    if (!compareResult) return []
    let rows = compareResult.rows
    if (!showAllIncrements) {
      rows = rows.filter((r) => r.severity !== 'info')
    }
    if (categoryFilter !== 'all') {
      rows = rows.filter((r) => r.category === categoryFilter)
    }
    return rows.map((row, idx) => ({
      id: idx,
      severity: row.severity,
      category: CATEGORY_LABELS[row.category] || row.category,
      node: row.node,
      device: row.device,
      stat: row.stat,
      before: row.before,
      after: row.after,
      diff: row.diff,
    }))
  }, [compareResult, categoryFilter, showAllIncrements])

  const tableColumns = [
    {
      title: 'Severity',
      data: 'severity',
      render: (data: string) => severityBadgeHtml(data),
    },
    { title: 'Category', data: 'category' },
    { title: 'Node', data: 'node' },
    { title: 'Device', data: 'device' },
    { title: 'Counter', data: 'stat' },
    { title: 'Before', data: 'before', className: 'dt-right' },
    { title: 'After', data: 'after', className: 'dt-right' },
    {
      title: 'Diff',
      data: 'diff',
      className: 'dt-right',
      render: (data: number) =>
        `<span style="color:${data > 0 ? '#dc2626' : '#374151'}; font-weight:600">${data}</span>`,
    },
  ]

  const count = list?.count ?? 0
  const max = list?.max ?? 5
  const capturing = loading === 'capture' || !!list?.in_progress
  const isFull = count >= max
  const canDiff = count >= 2 && beforeId && afterId && beforeId !== afterId && loading !== 'diff'

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Snapshot Diff</h1>
        <p className="mt-1 text-sm text-gray-600">
          Capture up to {max} cluster counter snapshots, then diff any pair. Capture can take up to 3
          minutes and may slow live metrics. Snapshots are saved under config and survive backend
          restart. Degraded rows mean some nodes returned errors.
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">{error}</div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-2">
              <Camera className="h-5 w-5 text-blue-600" />
              Gallery ({count}/{max})
            </span>
            {list && list.snapshots.length > 0 && (
              <button
                onClick={handleClearAll}
                disabled={capturing}
                className="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
              >
                <Trash2 className="h-4 w-4" />
                Clear All
              </button>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <input
              type="text"
              maxLength={80}
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="Optional label (e.g. before rccl test)"
              className="min-w-[16rem] flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm"
            />
            <button
              onClick={handleCapture}
              disabled={capturing || isFull}
              className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {capturing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Camera className="h-4 w-4" />}
              {isFull ? 'Gallery full' : 'Capture Snapshot'}
            </button>
          </div>

          {list && list.snapshots.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-xs uppercase text-gray-500">
                    <th className="px-3 py-2">Label</th>
                    <th className="px-3 py-2">Captured</th>
                    <th className="px-3 py-2">Nodes</th>
                    <th className="px-3 py-2">Stats</th>
                    <th className="px-3 py-2">Status</th>
                    <th className="px-3 py-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {list.snapshots.map((snap) => (
                    <tr key={snap.id} className="border-b last:border-0">
                      <td className="px-3 py-2 font-medium text-gray-900">{snap.label || '—'}</td>
                      <td className="px-3 py-2 text-gray-700">{new Date(snap.captured_at).toLocaleString()}</td>
                      <td className="px-3 py-2">{totalNodes(snap)}</td>
                      <td className="px-3 py-2">{totalStats(snap)}</td>
                      <td className="px-3 py-2">
                        {snap.degraded ? (
                          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
                            Degraded
                          </span>
                        ) : (
                          <span className="rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
                            OK
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-right">
                        <button
                          onClick={() => handleDelete(snap.id)}
                          className="inline-flex items-center gap-1 text-sm text-red-700 hover:underline"
                        >
                          <Trash2 className="h-4 w-4" />
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-gray-500">No snapshots yet. Capture one before a workload, then another after.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitCompare className="h-5 w-5 text-green-600" />
            Diff
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-gray-700">Before</span>
              <select
                value={beforeId}
                onChange={(e) => setBeforeId(e.target.value)}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm"
              >
                <option value="">Select snapshot</option>
                {(list?.snapshots || []).map((snap) => (
                  <option key={snap.id} value={snap.id}>
                    {snapshotTitle(snap)}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-gray-700">After</span>
              <select
                value={afterId}
                onChange={(e) => setAfterId(e.target.value)}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm"
              >
                <option value="">Select snapshot</option>
                {(list?.snapshots || []).map((snap) => (
                  <option key={snap.id} value={snap.id}>
                    {snapshotTitle(snap)}
                  </option>
                ))}
              </select>
            </label>
            <button
              onClick={handleDiff}
              disabled={!canDiff}
              className="inline-flex items-center gap-2 rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-50"
            >
              {loading === 'diff' ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitCompare className="h-4 w-4" />}
              Run Diff
            </button>
          </div>
        </CardContent>
      </Card>

      {compareResult && (
        <Card>
          <CardHeader>
            <CardTitle>Comparison Results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-4 text-sm text-gray-600">
              <span>
                Before: {compareResult.before_label ? `${compareResult.before_label} · ` : ''}
                {new Date(compareResult.before_timestamp).toLocaleString()}
              </span>
              <span>
                After: {compareResult.after_label ? `${compareResult.after_label} · ` : ''}
                {new Date(compareResult.after_timestamp).toLocaleString()}
              </span>
            </div>

            {compareResult.warnings?.length > 0 && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                {compareResult.warnings.map((w) => (
                  <p key={w}>{w}</p>
                ))}
              </div>
            )}

            <div className="flex flex-wrap gap-3">
              <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-2">
                <p className="text-xs font-medium uppercase text-red-600">Errors</p>
                <p className="text-2xl font-bold text-red-800">{compareResult.summary.errors}</p>
              </div>
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-2">
                <p className="text-xs font-medium uppercase text-amber-600">Warnings</p>
                <p className="text-2xl font-bold text-amber-800">{compareResult.summary.warnings}</p>
              </div>
              <div className="rounded-lg border border-orange-200 bg-orange-50 px-4 py-2">
                <p className="text-xs font-medium uppercase text-orange-600">Threshold</p>
                <p className="text-2xl font-bold text-orange-800">{compareResult.summary.threshold_warnings}</p>
              </div>
              <div className="rounded-lg border border-gray-200 bg-gray-50 px-4 py-2">
                <p className="text-xs font-medium uppercase text-gray-600">Total Increments</p>
                <p className="text-2xl font-bold text-gray-800">{compareResult.summary.total_increments}</p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-4">
              <select
                value={categoryFilter}
                onChange={(e) => setCategoryFilter(e.target.value)}
                className="rounded-lg border border-gray-300 px-3 py-2 text-sm"
              >
                <option value="all">All categories</option>
                {Object.entries(CATEGORY_LABELS).map(([key, catLabel]) => (
                  <option key={key} value={key}>
                    {catLabel}
                  </option>
                ))}
              </select>
              <label className="flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={showAllIncrements}
                  onChange={(e) => setShowAllIncrements(e.target.checked)}
                  className="rounded border-gray-300"
                />
                Show all increments (including info)
              </label>
            </div>

            {filteredRows.length > 0 ? (
              <CustomDataTable columns={tableColumns} data={filteredRows} defaultPageLength={25} />
            ) : (
              <p className="rounded-lg bg-green-50 px-4 py-3 text-sm text-green-800">
                No matching counter increments
                {showAllIncrements ? '' : ' (info rows hidden)'}.
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
