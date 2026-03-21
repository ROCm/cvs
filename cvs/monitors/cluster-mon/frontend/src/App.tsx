import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { useWebSocket } from './hooks/useWebSocket'
import { MainLayout } from './components/Layout/MainLayout'
import { DashboardPage } from './pages/DashboardPage'
import { ConfigurationPage } from './pages/ConfigurationPage'
import { GPUMetricsPage } from './pages/GPUMetricsPage'
import { NICMetricsPage } from './pages/NICMetricsPage'
import { TopologyPage } from './pages/TopologyPage'
import { GPUSoftwarePage } from './pages/GPUSoftwarePage'
import { NICSoftwarePage } from './pages/NICSoftwarePage'
import { LogsPage } from './pages/LogsPage'
import { RCCLHealthPage } from './pages/RCCLHealthPage'
import { RCCLTopologyPage } from './pages/RCCLTopologyPage'
import { RCCLTimelinePage } from './pages/RCCLTimelinePage'
import { NodeDetailsModal } from './components/NodeDetailsModal'

function App() {
  useWebSocket()

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<DashboardPage />} />
          <Route path="config" element={<ConfigurationPage />} />
          <Route path="gpu-metrics" element={<GPUMetricsPage />} />
          <Route path="nic-metrics" element={<NICMetricsPage />} />
          <Route path="topology" element={<TopologyPage />} />
          <Route path="gpu-software" element={<GPUSoftwarePage />} />
          <Route path="nic-software" element={<NICSoftwarePage />} />
          <Route path="logs" element={<LogsPage />} />
          <Route path="rccl-health" element={<RCCLHealthPage />} />
          <Route path="rccl-topology" element={<RCCLTopologyPage />} />
          <Route path="rccl-timeline" element={<RCCLTimelinePage />} />
        </Route>
      </Routes>
      <NodeDetailsModal />
    </BrowserRouter>
  )
}

export default App
