import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import NodeGroups from './pages/NodeGroups'
import NodeGroupDetail from './pages/NodeGroupDetail'
import AddNodeGroup from './pages/AddNodeGroup'
import MonitoringServers from './pages/MonitoringServers'
import MetricGroups from './pages/MetricGroups'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="monitoring" element={<MonitoringServers />} />
          <Route path="metrics" element={<MetricGroups />} />
          <Route path="nodegroups" element={<NodeGroups />} />
          <Route path="nodegroups/new" element={<AddNodeGroup />} />
          <Route path="nodegroups/:id" element={<NodeGroupDetail />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
