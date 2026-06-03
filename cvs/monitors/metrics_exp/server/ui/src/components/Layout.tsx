import { useState } from 'react'
import { Outlet, NavLink } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchHealth, fetchNodeGroups, fetchMonitoringServers } from '../api'
import {
  LayoutDashboard,
  Server,
  Monitor,
  BarChart3,
  Activity,
  ExternalLink,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'

function Layout() {
  const [dashboardsExpanded, setDashboardsExpanded] = useState(true)

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 30000,
  })

  const { data: nodeGroups = [] } = useQuery({
    queryKey: ['nodegroups'],
    queryFn: fetchNodeGroups,
    refetchInterval: 30000,
  })

  const { data: monitoringServers = [] } = useQuery({
    queryKey: ['monitoring-servers'],
    queryFn: fetchMonitoringServers,
    refetchInterval: 30000,
  })

  // Menu order: Dashboard -> Monitoring Servers -> Metric Groups -> Node Groups
  const navItems = [
    { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { to: '/monitoring', icon: Monitor, label: 'Monitoring Servers' },
    { to: '/metrics', icon: BarChart3, label: 'Metric Groups' },
    { to: '/nodegroups', icon: Server, label: 'Node Groups' },
  ]

  // Build dashboard links for each node group with a monitoring server
  const dashboardLinks = nodeGroups
    .filter((ng) => ng.monitoring_server_id)
    .map((ng) => {
      const server = monitoringServers.find((s) => s.id === ng.monitoring_server_id)
      if (!server || !server.server_ip) return null
      const grafanaUrl = `http://${server.server_ip}:${server.grafana_port}`
      const dashboardId = `ng_${ng.name.replace(/[^a-zA-Z0-9]/g, '_')}`
      return {
        href: `${grafanaUrl}/d/${dashboardId}`,
        label: ng.name,
        serverName: server.name,
      }
    })
    .filter(Boolean) as { href: string; label: string; serverName: string }[]

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-64 bg-amd-gray-900 text-white flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-amd-gray-700">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-brand-600 rounded-lg flex items-center justify-center">
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-bold text-lg">GPU Fleet</h1>
              <p className="text-xs text-amd-gray-400">Monitor</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            {navItems.map(({ to, icon: Icon, label }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                      isActive
                        ? 'bg-brand-600 text-white'
                        : 'text-amd-gray-300 hover:bg-amd-gray-800'
                    }`
                  }
                >
                  <Icon className="w-5 h-5" />
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>

          <div className="mt-8">
            <button
              onClick={() => setDashboardsExpanded(!dashboardsExpanded)}
              className="w-full px-4 flex items-center justify-between text-xs font-semibold text-amd-gray-500 uppercase tracking-wider mb-2 hover:text-amd-gray-300"
            >
              <span>Grafana Dashboards</span>
              {dashboardsExpanded ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>
            {dashboardsExpanded && (
              <ul className="space-y-1">
                {dashboardLinks.length > 0 ? (
                  dashboardLinks.map(({ href, label, serverName }) => (
                    <li key={href}>
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-3 px-4 py-2 rounded-lg text-amd-gray-300 hover:bg-amd-gray-800 transition-colors"
                      >
                        <ExternalLink className="w-4 h-4" />
                        <div className="flex flex-col">
                          <span className="text-sm">{label}</span>
                          <span className="text-xs text-amd-gray-500">{serverName}</span>
                        </div>
                      </a>
                    </li>
                  ))
                ) : (
                  <li className="px-4 py-2 text-sm text-amd-gray-500">
                    No dashboards configured
                  </li>
                )}
              </ul>
            )}
          </div>
        </nav>

        {/* Status */}
        <div className="p-4 border-t border-amd-gray-700">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                health?.status === 'healthy' ? 'bg-green-500' : 'bg-yellow-500'
              }`}
            />
            <span className="text-sm text-amd-gray-400">
              {health?.status === 'healthy' ? 'All systems operational' : 'Degraded'}
            </span>
          </div>
          <p className="text-xs text-amd-gray-500 mt-1">v{health?.version || '1.0.0'}</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}

export default Layout
