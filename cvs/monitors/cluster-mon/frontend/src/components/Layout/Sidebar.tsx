import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Settings, Cpu, Network, Activity, Package, HardDrive, Share2, FileText } from 'lucide-react'
import { cn } from '@/utils/cn'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Configuration', href: '/config', icon: Settings },
  { name: 'GPU Metrics', href: '/gpu-metrics', icon: Cpu },
  { name: 'NIC Metrics', href: '/nic-metrics', icon: Network },
  { name: 'Topology', href: '/topology', icon: Share2 },
  { name: 'GPU SW', href: '/gpu-software', icon: Package },
  { name: 'NIC SW', href: '/nic-software', icon: HardDrive },
  { name: 'Logs', href: '/logs', icon: FileText },
]

export function Sidebar() {
  return (
    <div className="flex flex-col w-64 bg-gray-900 text-white">
      {/* Logo/Header */}
      <div className="flex items-center gap-3 px-6 py-6 border-b border-gray-800">
        <Activity className="h-8 w-8 text-blue-400" />
        <div>
          <h1 className="text-lg font-bold">GPU Cluster</h1>
          <p className="text-xs text-gray-400">Monitor</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-2">
        {navigation.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-4 py-3 rounded-lg transition-colors',
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-800 hover:text-white'
                )
              }
            >
              <Icon className="h-5 w-5" />
              <span className="font-medium">{item.name}</span>
            </NavLink>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-gray-800">
        <p className="text-xs text-gray-500">
          Version 0.1.0
        </p>
      </div>
    </div>
  )
}
