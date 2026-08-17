import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { fetchNodeGroups, deleteNodeGroup } from '../api'
import api from '../api'
import {
  Server,
  Plus,
  Trash2,
  CheckCircle,
  Clock,
  AlertTriangle,
  Monitor,
  BarChart3,
} from 'lucide-react'
import { useState } from 'react'

function NodeGroups() {
  const queryClient = useQueryClient()
  const [deleteId, setDeleteId] = useState<number | null>(null)

  const { data: nodeGroups, isLoading } = useQuery({
    queryKey: ['nodegroups'],
    queryFn: fetchNodeGroups,
  })

  const { data: monitoringServers = [] } = useQuery({
    queryKey: ['monitoring-servers'],
    queryFn: () => api.get('/monitoring-servers').then(r => r.data),
  })

  const { data: metricGroups = [] } = useQuery({
    queryKey: ['metric-groups'],
    queryFn: () => api.get('/metric-groups').then(r => r.data),
  })

  const deleteMutation = useMutation({
    mutationFn: deleteNodeGroup,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['nodegroups'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      setDeleteId(null)
    },
  })

  if (isLoading) {
    return (
      <div className="p-8 flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-4 border-brand-600 border-t-transparent rounded-full" />
      </div>
    )
  }

  const getMonitoringServerName = (id: number | null | undefined) => {
    if (!id) return null
    const server = monitoringServers.find((s: any) => s.id === id)
    return server?.name
  }

  const getMetricGroupName = (id: number | null | undefined) => {
    if (!id) return null
    const group = metricGroups.find((g: any) => g.id === id)
    return group?.name
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-amd-gray-900">Node Groups</h1>
          <p className="text-amd-gray-500 mt-1">
            Manage your GPU node clusters
          </p>
        </div>
        <Link to="/nodegroups/new" className="btn btn-primary flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Add Node Group
        </Link>
      </div>

      {nodeGroups && nodeGroups.length > 0 ? (
        <div className="grid gap-6">
          {nodeGroups.map((group) => (
            <div key={group.id} className="card">
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4">
                  <div className="p-3 bg-brand-100 rounded-lg">
                    <Server className="w-6 h-6 text-brand-600" />
                  </div>
                  <div>
                    <Link
                      to={`/nodegroups/${group.id}`}
                      className="text-xl font-semibold text-brand-600 hover:underline"
                    >
                      {group.name}
                    </Link>
                    {group.description && (
                      <p className="text-amd-gray-500 mt-1">{group.description}</p>
                    )}
                    <div className="flex items-center gap-4 mt-3 text-sm text-amd-gray-500">
                      <span>SSH User: {group.ssh_user}</span>
                      <span>Port: {group.ssh_port}</span>
                      <span>
                        {group.ssh_key_path ? (
                          <span className="text-green-600">SSH Key configured</span>
                        ) : (
                          <span className="text-yellow-600">No SSH Key</span>
                        )}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 mt-2">
                      {(group as any).monitoring_server_id && (
                        <span className="inline-flex items-center gap-1 text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                          <Monitor className="w-3 h-3" />
                          {getMonitoringServerName((group as any).monitoring_server_id) || 'Unknown'}
                        </span>
                      )}
                      {(group as any).metric_group_id && (
                        <span className="inline-flex items-center gap-1 text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded">
                          <BarChart3 className="w-3 h-3" />
                          {getMetricGroupName((group as any).metric_group_id) || 'Unknown'}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <div className="flex items-center gap-2">
                      {group.active_nodes === group.node_count && group.node_count > 0 ? (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      ) : group.active_nodes > 0 ? (
                        <Clock className="w-5 h-5 text-yellow-500" />
                      ) : (
                        <AlertTriangle className="w-5 h-5 text-amd-gray-400" />
                      )}
                      <span className="font-medium">
                        {group.active_nodes} / {group.node_count} nodes
                      </span>
                    </div>
                    <p className="text-sm text-amd-gray-500 mt-1">
                      Created {new Date(group.created_at).toLocaleDateString()}
                    </p>
                  </div>

                  <button
                    onClick={() => setDeleteId(group.id)}
                    className="p-2 text-amd-gray-400 hover:text-red-600 transition-colors"
                    title="Delete node group"
                  >
                    <Trash2 className="w-5 h-5" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="card text-center py-12">
          <Server className="w-12 h-12 text-amd-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-amd-gray-900 mb-2">
            No node groups yet
          </h3>
          <p className="text-amd-gray-500 mb-4">
            Create your first node group to start monitoring GPU nodes
          </p>
          <Link to="/nodegroups/new" className="btn btn-primary">
            Add Node Group
          </Link>
        </div>
      )}

      {/* Delete confirmation modal */}
      {deleteId && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">Delete Node Group?</h3>
            <p className="text-amd-gray-500 mb-6">
              This will remove all nodes and monitoring configuration for this group.
              This action cannot be undone.
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

export default NodeGroups
