import { useParams, Navigate } from 'react-router-dom'
import { NICMetricsPage } from './NICMetricsPage'
import { NICSoftwarePage } from './NICSoftwarePage'
import { ComingSoon } from './ComingSoonPage'

export function ScaleOutPage() {
  const { tab } = useParams<{ tab: string }>()

  if (!tab) return <Navigate to="/networks/backend/scale-out/metrics" replace />

  return (
    <div className="space-y-4">
      {tab === 'summary' && <NICSoftwarePage />}
      {tab === 'metrics' && <NICMetricsPage />}
      {tab === 'health' && (
        <>
          <h1 className="text-2xl font-bold text-gray-900">Networks / Backend / Scale-out — Health</h1>
          <ComingSoon title="Scale-out Health" />
        </>
      )}
      {tab !== 'summary' && tab !== 'metrics' && tab !== 'health' && (
        <>
          <h1 className="text-2xl font-bold text-gray-900">Networks / Backend / Scale-out — {tab}</h1>
          <ComingSoon title={`Scale-out ${tab}`} />
        </>
      )}
    </div>
  )
}
