import { useParams, Navigate } from 'react-router-dom'
import { CPUSummaryPage } from './CPUSummaryPage'
import { CPUMetricsPage } from './CPUMetricsPage'
import { ComingSoon } from './ComingSoonPage'

export function CPUsPage() {
  const { tab } = useParams<{ tab: string }>()

  if (!tab) return <Navigate to="/compute/cpus/summary" replace />

  return (
    <div className="space-y-4">
      {tab === 'summary' && <CPUSummaryPage />}
      {tab === 'metrics' && <CPUMetricsPage />}
      {tab !== 'summary' && tab !== 'metrics' && (
        <ComingSoon title={`CPU ${tab}`} />
      )}
    </div>
  )
}
