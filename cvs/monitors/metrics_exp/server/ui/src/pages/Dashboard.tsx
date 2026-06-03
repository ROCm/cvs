import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { fetchStats, fetchNodeGroups, fetchHealth, fetchMonitoringEndpoints } from '../api'
import {
  Server,
  Cpu,
  Activity,
  AlertTriangle,
  CheckCircle,
  Clock,
  Plus,
} from 'lucide-react'

function Dashboard() {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: fetchStats,
    refetchInterval: 30000,
  })

  const { data: nodeGroups, isLoading: groupsLoading } = useQuery({
    queryKey: ['nodegroups'],
    queryFn: fetchNodeGroups,
    refetchInterval: 30000,
  })

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 30000,
  })

  const { data: endpoints } = useQuery({
    queryKey: ['monitoring-endpoints'],
    queryFn: fetchMonitoringEndpoints,
  })

  if (statsLoading || groupsLoading) {
    return (
      <div className="p-8 flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-4 border-brand-600 border-t-transparent rounded-full" />
      </div>
    )
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-amd-gray-900">Fleet Dashboard</h1>
        <p className="text-amd-gray-500 mt-1">
          Monitor your AMD GPU infrastructure
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Node Groups"
          value={stats?.total_node_groups || 0}
          icon={Server}
          color="blue"
        />
        <StatCard
          title="Total Nodes"
          value={stats?.total_nodes || 0}
          icon={Cpu}
          subtext={`${stats?.active_nodes || 0} active`}
          color="green"
        />
        <StatCard
          title="Total GPUs"
          value={stats?.total_gpus || 0}
          icon={Activity}
          color="purple"
        />
        <StatCard
          title="Issues"
          value={stats?.error_nodes || 0}
          icon={AlertTriangle}
          color={stats?.error_nodes ? 'red' : 'green'}
        />
      </div>

      {/* Service Health */}
      <div className="card mb-8">
        <h2 className="text-lg font-semibold mb-4">Service Health</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {health?.services.map((service) => (
            <div
              key={service.name}
              className="flex items-center gap-3 p-3 bg-amd-gray-50 rounded-lg"
            >
              {service.status === 'healthy' ? (
                <CheckCircle className="w-5 h-5 text-green-500" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-yellow-500" />
              )}
              <div>
                <p className="font-medium capitalize">{service.name}</p>
                <p className="text-sm text-amd-gray-500 capitalize">
                  {service.status}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Node Groups */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Node Groups</h2>
          <Link to="/nodegroups/new" className="btn btn-primary flex items-center gap-2">
            <Plus className="w-4 h-4" />
            Add Node Group
          </Link>
        </div>

        {nodeGroups && nodeGroups.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-amd-gray-200">
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500">
                    Name
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500">
                    Nodes
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500">
                    Status
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500">
                    Dashboard
                  </th>
                </tr>
              </thead>
              <tbody>
                {nodeGroups.map((group) => (
                  <tr
                    key={group.id}
                    className="border-b border-amd-gray-100 hover:bg-amd-gray-50"
                  >
                    <td className="py-3 px-4">
                      <Link
                        to={`/nodegroups/${group.id}`}
                        className="font-medium text-brand-600 hover:underline"
                      >
                        {group.name}
                      </Link>
                      {group.description && (
                        <p className="text-sm text-amd-gray-500">
                          {group.description}
                        </p>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      <span className="font-medium">{group.active_nodes}</span>
                      <span className="text-amd-gray-500">
                        {' '}
                        / {group.node_count}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {group.active_nodes === group.node_count ? (
                        <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-100 text-green-700 rounded-full text-sm">
                          <CheckCircle className="w-4 h-4" />
                          All Active
                        </span>
                      ) : group.active_nodes > 0 ? (
                        <span className="inline-flex items-center gap-1 px-2 py-1 bg-yellow-100 text-yellow-700 rounded-full text-sm">
                          <Clock className="w-4 h-4" />
                          Partial
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-1 bg-amd-gray-100 text-amd-gray-600 rounded-full text-sm">
                          <Clock className="w-4 h-4" />
                          Pending
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      {endpoints?.endpoints?.grafana ? (
                        <a
                          href={`${endpoints.endpoints.grafana}/d/ng_${group.name.replace(/[^a-zA-Z0-9]/g, '_')}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-brand-600 hover:underline text-sm"
                        >
                          Open in Grafana
                        </a>
                      ) : (
                        <span className="text-amd-gray-400 text-sm">
                          Grafana not configured
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12">
            <Server className="w-12 h-12 text-amd-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-amd-gray-900 mb-2">
              No node groups yet
            </h3>
            <p className="text-amd-gray-500 mb-4">
              Get started by creating your first node group
            </p>
            <Link to="/nodegroups/new" className="btn btn-primary">
              Add Node Group
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}

function StatCard({
  title,
  value,
  icon: Icon,
  subtext,
  color,
}: {
  title: string
  value: number
  icon: React.ElementType
  subtext?: string
  color: 'blue' | 'green' | 'purple' | 'red'
}) {
  const colorClasses = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    purple: 'bg-purple-100 text-purple-600',
    red: 'bg-red-100 text-red-600',
  }

  return (
    <div className="card">
      <div className="flex items-center gap-4">
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          <Icon className="w-6 h-6" />
        </div>
        <div>
          <p className="text-sm text-amd-gray-500">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
          {subtext && (
            <p className="text-sm text-amd-gray-500">{subtext}</p>
          )}
        </div>
      </div>
    </div>
  )
}

export default Dashboard
