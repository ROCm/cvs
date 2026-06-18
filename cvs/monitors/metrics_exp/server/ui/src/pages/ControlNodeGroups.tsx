import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  fetchControlNodeGroups,
  deleteControlNodeGroup,
  fetchMonitoringServers,
} from '../api'
import {
  Cpu,
  Plus,
  Trash2,
  CheckCircle,
  Clock,
  AlertTriangle,
  Monitor,
} from 'lucide-react'

function ControlNodeGroups() {
  const queryClient = useQueryClient()
  const [deleteId, setDeleteId] = useState<number | null>(null)

  const { data: groups = [], isLoading } = useQuery({
    queryKey: ['control-nodegroups'],
    queryFn: fetchControlNodeGroups,
  })

  const { data: monitoringServers = [] } = useQuery({
    queryKey: ['monitoring-servers'],
    queryFn: fetchMonitoringServers,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteControlNodeGroup,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['control-nodegroups'] })
      setDeleteId(null)
    },
  })

  const getMonitoringServerName = (id?: number) => {
    if (!id) return null
    return monitoringServers.find((s) => s.id === id)?.name
  }

  const controlTypeBadge = (type: string) => (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
        type === 'slurm'
          ? 'bg-orange-100 text-orange-700'
          : 'bg-blue-100 text-blue-700'
      }`}
    >
      {type === 'slurm' ? 'Slurm' : 'Kubernetes'}
    </span>
  )

  if (isLoading) {
    return (
      <div className="p-8 flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-4 border-brand-600 border-t-transparent rounded-full" />
      </div>
    )
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-amd-gray-900">Control Node Groups</h1>
          <p className="text-amd-gray-500 mt-1">
            Monitor Slurm and Kubernetes control plane nodes
          </p>
        </div>
        <Link
          to="/control-nodegroups/new"
          className="btn btn-primary flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Add Control Group
        </Link>
      </div>

      {groups.length > 0 ? (
        <div className="grid gap-6">
          {groups.map((group) => (
            <div key={group.id} className="card">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-brand-100 rounded-lg flex-shrink-0">
                    <Cpu className="w-6 h-6 text-brand-600" />
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-1">
                      <Link
                        to={`/control-nodegroups/${group.id}`}
                        className="text-xl font-semibold text-brand-600 hover:underline"
                      >
                        {group.name}
                      </Link>
                      {controlTypeBadge(group.control_type)}
                    </div>
                    {group.description && (
                      <p className="text-amd-gray-500 mt-1 text-sm">{group.description}</p>
                    )}
                    <div className="flex items-center gap-4 mt-2 text-sm text-amd-gray-500">
                      <span>
                        SSH: {group.ssh_user}:{group.ssh_port}
                      </span>
                      {group.ssh_key_path ? (
                        <span className="text-green-600">Key configured</span>
                      ) : group.has_ssh_password ? (
                        <span className="text-green-600">Password configured</span>
                      ) : (
                        <span className="text-yellow-600">No credentials</span>
                      )}
                      {group.use_jump_host && group.jump_host && (
                        <span className="text-blue-600">via {group.jump_host}</span>
                      )}
                    </div>
                    {group.monitoring_server_id && (
                      <div className="flex items-center gap-2 mt-2">
                        <span className="inline-flex items-center gap-1 text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                          <Monitor className="w-3 h-3" />
                          {getMonitoringServerName(group.monitoring_server_id) || 'Unknown'}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <div className="flex items-center gap-2">
                      {group.active_node_count === group.node_count && group.node_count > 0 ? (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      ) : group.active_node_count > 0 ? (
                        <Clock className="w-5 h-5 text-yellow-500" />
                      ) : (
                        <AlertTriangle className="w-5 h-5 text-amd-gray-400" />
                      )}
                      <span className="font-medium">
                        {group.active_node_count} / {group.node_count} nodes active
                      </span>
                    </div>
                    <p className="text-sm text-amd-gray-500 mt-1">
                      Created {new Date(group.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={() => setDeleteId(group.id)}
                    className="p-2 text-amd-gray-400 hover:text-red-600 transition-colors"
                    title="Delete group"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="card text-center py-16">
          <Cpu className="w-16 h-16 text-amd-gray-300 mx-auto mb-4" />
          <h3 className="text-xl font-medium text-amd-gray-900 mb-2">
            No control node groups yet
          </h3>
          <p className="text-amd-gray-500 mb-6 max-w-md mx-auto">
            Add a Slurm head node or Kubernetes control plane to start monitoring your
            control infrastructure alongside your GPU fleet.
          </p>
          <Link to="/control-nodegroups/new" className="btn btn-primary">
            Add Control Node Group
          </Link>
        </div>
      )}

      {/* Delete confirmation modal */}
      {deleteId !== null && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 className="text-lg font-semibold mb-2">Delete Control Node Group?</h3>
            <p className="text-amd-gray-500 mb-6">
              This will permanently remove all nodes, Prometheus targets, and the Grafana
              dashboard for this group. This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setDeleteId(null)}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={() => deleteMutation.mutate(deleteId)}
                disabled={deleteMutation.isPending}
                className="btn btn-danger"
              >
                {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ControlNodeGroups
