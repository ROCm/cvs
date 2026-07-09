import { useParams, Navigate } from 'react-router-dom'
import { ComingSoon } from './ComingSoonPage'

export function FrontendNetworkPage() {
  const { tab } = useParams<{ tab: string }>()

  if (!tab) return <Navigate to="/networks/frontend/health" replace />

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-gray-900">
        Networks / Frontend — {tab.charAt(0).toUpperCase() + tab.slice(1)}
      </h1>
      <ComingSoon title={`Frontend ${tab.charAt(0).toUpperCase() + tab.slice(1)}`} />
    </div>
  )
}
