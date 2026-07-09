import { useParams, Navigate } from 'react-router-dom'
import { GPUMetricsPage } from './GPUMetricsPage'
import { GPUSummaryPage } from './GPUSummaryPage'
import { ComputeTrayIFoEPage } from './ComputeTrayIFoEPage'
import { ComingSoon } from './ComingSoonPage'

export function GPUsPage() {
  const { tab } = useParams<{ tab: string }>()

  if (!tab) return <Navigate to="/compute/gpus/summary" replace />

  return (
    <div className="space-y-4">
      {tab === 'summary' && <GPUSummaryPage />}
      {tab === 'metrics' && <GPUMetricsPage />}
      {tab === 'ifoe'    && <ComputeTrayIFoEPage />}
      {!['summary', 'metrics', 'ifoe'].includes(tab) && (
        <>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">GPUs — {tab}</h1>
          <ComingSoon title={`GPU ${tab}`} />
        </>
      )}
    </div>
  )
}
