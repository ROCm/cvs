import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useWebSocket } from './hooks/useWebSocket'
import { MainLayout } from './components/Layout/MainLayout'
import { DashboardPage } from './pages/DashboardPage'
import { ConfigurationPage } from './pages/ConfigurationPage'
import { TopologyPage } from './pages/TopologyPage'
import { LogsPage } from './pages/LogsPage'
import { RCCLHealthPage } from './pages/RCCLHealthPage'
import { RCCLTopologyPage } from './pages/RCCLTopologyPage'
import { RCCLTimelinePage } from './pages/RCCLTimelinePage'
import { RCCLPerformancePage } from './pages/RCCLPerformancePage'
import { NodeDetailsModal } from './components/NodeDetailsModal'

// Compute
import { CPUsPage } from './pages/CPUsPage'
import { GPUsPage } from './pages/GPUsPage'

// Storage
import { StorageSummaryPage } from './pages/StorageSummaryPage'
import { StorageIOPage }      from './pages/StorageIOPage'
import { StorageCachePage }   from './pages/StorageCachePage'

// Networks
import { FrontendNetworkPage } from './pages/FrontendNetworkPage'
import { ScaleOutPage } from './pages/ScaleOutPage'
import { ScaleUpPage } from './pages/ScaleUpPage'

function App() {
  useWebSocket()

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<DashboardPage />} />
          <Route path="config" element={<ConfigurationPage />} />

          {/* Compute */}
          <Route path="compute">
            <Route index element={<Navigate to="/compute/gpus/summary" replace />} />
            <Route path="cpus">
              <Route index element={<Navigate to="/compute/cpus/summary" replace />} />
              <Route path=":tab" element={<CPUsPage />} />
            </Route>
            <Route path="gpus">
              <Route index element={<Navigate to="/compute/gpus/summary" replace />} />
              <Route path=":tab" element={<GPUsPage />} />
            </Route>
          </Route>

          {/* Networks */}
          <Route path="networks">
            <Route index element={<Navigate to="/networks/backend/scale-out/metrics" replace />} />
            <Route path="frontend">
              <Route index element={<Navigate to="/networks/frontend/health" replace />} />
              <Route path=":tab" element={<FrontendNetworkPage />} />
            </Route>
            <Route path="backend">
              <Route index element={<Navigate to="/networks/backend/scale-out/metrics" replace />} />
              <Route path="scale-out">
                <Route index element={<Navigate to="/networks/backend/scale-out/metrics" replace />} />
                <Route path=":tab" element={<ScaleOutPage />} />
              </Route>
              <Route path="scale-up">
                <Route index element={<Navigate to="/networks/backend/scale-up/overview" replace />} />
                <Route path=":tab" element={<ScaleUpPage />} />
              </Route>
            </Route>
          </Route>

          {/* Storage */}
          <Route path="storage">
            <Route index element={<Navigate to="/storage/summary" replace />} />
            <Route path="summary" element={<StorageSummaryPage />} />
            <Route path="io"      element={<StorageIOPage />} />
            <Route path="cache"   element={<StorageCachePage />} />
          </Route>

          {/* Legacy redirects */}
          <Route path="gpu-metrics"   element={<Navigate to="/compute/gpus/metrics"                   replace />} />
          <Route path="nic-metrics"   element={<Navigate to="/networks/backend/scale-out/metrics"     replace />} />
          <Route path="ifoe-details"  element={<Navigate to="/networks/backend/scale-up/overview"     replace />} />
          <Route path="gpu-software"  element={<Navigate to="/compute/gpus/summary"                   replace />} />
          <Route path="nic-software"  element={<Navigate to="/networks/backend/scale-out/summary"     replace />} />

          {/* Unchanged pages */}
          <Route path="topology"        element={<TopologyPage />} />
          <Route path="logs"            element={<LogsPage />} />
          <Route path="rccl-health"     element={<RCCLHealthPage />} />
          <Route path="rccl-topology"   element={<RCCLTopologyPage />} />
          <Route path="rccl-timeline"   element={<RCCLTimelinePage />} />
          <Route path="rccl-performance" element={<RCCLPerformancePage />} />
        </Route>
      </Routes>
      <NodeDetailsModal />
    </BrowserRouter>
  )
}

export default App
