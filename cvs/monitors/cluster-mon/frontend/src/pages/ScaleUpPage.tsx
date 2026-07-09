import { useParams, Navigate } from 'react-router-dom'
import { ScaleUpOverviewPage } from './ScaleUpOverviewPage'
import { ScaleUpLinksPage }    from './ScaleUpLinksPage'
import { ScaleUpMetricsPage }  from './ScaleUpMetricsPage'

export function ScaleUpPage() {
  const { tab } = useParams<{ tab: string }>()

  if (!tab) return <Navigate to="/networks/backend/scale-up/overview" replace />

  if (tab === 'overview') return <ScaleUpOverviewPage />
  if (tab === 'links')    return <ScaleUpLinksPage />
  if (tab === 'metrics')  return <ScaleUpMetricsPage />

  // Legacy: redirect old tab names
  if (tab === 'health') return <Navigate to="/networks/backend/scale-up/overview" replace />

  return <Navigate to="/networks/backend/scale-up/overview" replace />
}
