import { useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  fetchNodeGroup,
  uploadSSHKey,
  uploadJumpHostKey,
  verifyConnectivity,
  installExporters,
  installExportersForce,
  addNodes,
  deleteNode,
  fetchMonitoringEndpoints,
  fetchMonitoringServers,
  fetchMetricGroups,
} from '../api'
import {
  ArrowLeft,
  Server,
  Upload,
  Play,
  Trash2,
  CheckCircle,
  Clock,
  AlertTriangle,
  XCircle,
  ExternalLink,
  Plus,
  RefreshCw,
  Monitor,
  BarChart3,
} from 'lucide-react'

function NodeGroupDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const nodeGroupId = parseInt(id!)

  const [showAddNodes, setShowAddNodes] = useState(false)
  const [newIPs, setNewIPs] = useState('')
  const [deleteNodeId, setDeleteNodeId] = useState<number | null>(null)
  const [installError, setInstallError] = useState<string | null>(null)

  const { data: nodeGroup, isLoading, refetch } = useQuery({
    queryKey: ['nodegroup', nodeGroupId],
    queryFn: () => fetchNodeGroup(nodeGroupId),
    refetchInterval: 10000,
  })

  const { data: endpoints } = useQuery({
    queryKey: ['monitoring-endpoints'],
    queryFn: fetchMonitoringEndpoints,
  })

  const { data: monitoringServers = [] } = useQuery({
    queryKey: ['monitoring-servers'],
    queryFn: fetchMonitoringServers,
  })

  const { data: metricGroups = [] } = useQuery({
    queryKey: ['metric-groups'],
    queryFn: fetchMetricGroups,
  })

  const uploadKeyMutation = useMutation({
    mutationFn: (file: File) => uploadSSHKey(nodeGroupId, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['nodegroup', nodeGroupId] })
    },
  })

  const uploadJumpKeyMutation = useMutation({
    mutationFn: (file: File) => uploadJumpHostKey(nodeGroupId, file),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['nodegroup', nodeGroupId] })
    },
  })

  const verifyMutation = useMutation({
    mutationFn: () => verifyConnectivity(nodeGroupId),
    onSuccess: (response) => {
      console.log('Verify response:', response)
      setInstallError(null)
      queryClient.invalidateQueries({ queryKey: ['nodegroup', nodeGroupId] })
    },
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      console.error('Verify error:', err)
      setInstallError(err.response?.data?.detail || err.message || 'Verification failed')
    },
  })

  const installMutation = useMutation({
    mutationFn: (nodeIds?: number[]) => installExporters(nodeGroupId, nodeIds),
    onSuccess: (response) => {
      console.log('Install response:', response)
      setInstallError(null)
      queryClient.invalidateQueries({ queryKey: ['nodegroup', nodeGroupId] })
    },
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      console.error('Install error:', err)
      setInstallError(err.response?.data?.detail || err.message || 'Installation failed')
    },
  })

  const installForceMutation = useMutation({
    mutationFn: (nodeIds?: number[]) => installExportersForce(nodeGroupId, nodeIds),
    onSuccess: (response) => {
      console.log('Install force response:', response)
      setInstallError(null)
      queryClient.invalidateQueries({ queryKey: ['nodegroup', nodeGroupId] })
    },
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      console.error('Install force error:', err)
      setInstallError(err.response?.data?.detail || err.message || 'Installation failed')
    },
  })

  const addNodesMutation = useMutation({
    mutationFn: (ips: string[]) => addNodes(nodeGroupId, ips),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['nodegroup', nodeGroupId] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      setShowAddNodes(false)
      setNewIPs('')
    },
  })

  const deleteNodeMutation = useMutation({
    mutationFn: (nodeId: number) => deleteNode(nodeGroupId, nodeId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['nodegroup', nodeGroupId] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      setDeleteNodeId(null)
    },
  })

  const handleKeyUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      uploadKeyMutation.mutate(file)
    }
  }

  const handleJumpKeyUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      uploadJumpKeyMutation.mutate(file)
    }
  }

  const handleAddNodes = () => {
    const ips = newIPs
      .split(/[\n,;]/)
      .map((ip) => ip.trim())
      .filter((ip) => ip)
    if (ips.length > 0) {
      addNodesMutation.mutate(ips)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="w-5 h-5 text-green-500" />
      case 'connected':
        return <CheckCircle className="w-5 h-5 text-blue-500" />
      case 'installing':
        return <RefreshCw className="w-5 h-5 text-blue-500 animate-spin" />
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-500" />
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />
      case 'unreachable':
        return <XCircle className="w-5 h-5 text-gray-500" />
      default:
        return <AlertTriangle className="w-5 h-5 text-gray-400" />
    }
  }

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      active: 'bg-green-100 text-green-700',
      connected: 'bg-blue-100 text-blue-700',
      installing: 'bg-blue-100 text-blue-700',
      pending: 'bg-yellow-100 text-yellow-700',
      error: 'bg-red-100 text-red-700',
      unreachable: 'bg-gray-100 text-gray-700',
    }
    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${styles[status] || styles.unreachable}`}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    )
  }

  if (isLoading) {
    return (
      <div className="p-8 flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-4 border-brand-600 border-t-transparent rounded-full" />
      </div>
    )
  }

  if (!nodeGroup) {
    return (
      <div className="p-8 text-center">
        <p>Node group not found</p>
        <Link to="/nodegroups" className="text-brand-600 hover:underline">
          Back to Node Groups
        </Link>
      </div>
    )
  }

  const pendingNodes = nodeGroup.nodes.filter((n) => n.status === 'pending')
  const connectedNodes = nodeGroup.nodes.filter((n) => n.status === 'connected')
  const unreachableNodes = nodeGroup.nodes.filter((n) => n.status === 'unreachable')
  const activeNodes = nodeGroup.nodes.filter((n) => n.status === 'active')

  // Find the associated monitoring server and metric group
  const monitoringServer = monitoringServers.find(s => s.id === nodeGroup.monitoring_server_id)
  const metricGroup = metricGroups.find(g => g.id === nodeGroup.metric_group_id)

  // Build Grafana URL based on the monitoring server
  let grafanaBase = ''
  if (monitoringServer && monitoringServer.server_ip) {
    grafanaBase = `http://${monitoringServer.server_ip}:${monitoringServer.grafana_port}`
  } else if (endpoints?.endpoints?.grafana) {
    grafanaBase = endpoints.endpoints.grafana
  }
  const dashboardUrl = grafanaBase ? `${grafanaBase}/d/ng_${nodeGroup.name.replace(/[^a-zA-Z0-9]/g, '_')}` : ''

  return (
    <div className="p-8">
      <button
        onClick={() => navigate('/nodegroups')}
        className="flex items-center gap-2 text-amd-gray-500 hover:text-amd-gray-700 mb-6"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Node Groups
      </button>

      {/* Header */}
      <div className="card mb-6">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-brand-100 rounded-lg">
              <Server className="w-8 h-8 text-brand-600" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">{nodeGroup.name}</h1>
              {nodeGroup.description && (
                <p className="text-amd-gray-500">{nodeGroup.description}</p>
              )}
              <div className="flex items-center gap-4 mt-2 text-sm text-amd-gray-500">
                <span>{nodeGroup.node_count} nodes</span>
                <span>{nodeGroup.active_nodes} active</span>
                <span>SSH: {nodeGroup.ssh_user}@:{nodeGroup.ssh_port}</span>
                {nodeGroup.use_jump_host && (
                  <span className="text-blue-600">via {nodeGroup.jump_host}</span>
                )}
              </div>
              <div className="flex items-center gap-4 mt-1 text-sm">
                {monitoringServer ? (
                  <span className="text-brand-600">
                    <Monitor className="w-4 h-4 inline mr-1" />
                    {monitoringServer.name}
                  </span>
                ) : (
                  <span className="text-amd-gray-400">No monitoring server</span>
                )}
                {metricGroup && (
                  <span className="text-green-600">
                    <BarChart3 className="w-4 h-4 inline mr-1" />
                    {metricGroup.name}
                  </span>
                )}
              </div>
            </div>
          </div>
          {dashboardUrl ? (
            <a
              href={dashboardUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-primary flex items-center gap-2"
            >
              <ExternalLink className="w-4 h-4" />
              Open Dashboard
            </a>
          ) : (
            <span className="text-amd-gray-400 text-sm">
              Configure monitoring server in Monitoring Servers
            </span>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
        {/* SSH Key/Password for direct connection */}
        {!nodeGroup.use_jump_host && (
          <div className="card">
            <h3 className="font-semibold mb-3">SSH Auth (GPU Nodes)</h3>
            {nodeGroup.ssh_auth_type === 'password' ? (
              nodeGroup.has_ssh_password ? (
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="text-sm">Password configured</span>
                </div>
              ) : (
                <p className="text-yellow-600 text-sm">Password not set</p>
              )
            ) : (
              <>
                {nodeGroup.ssh_key_path ? (
                  <div className="flex items-center gap-2 text-green-600">
                    <CheckCircle className="w-5 h-5" />
                    <span className="text-sm">Key configured</span>
                  </div>
                ) : (
                  <label className="btn btn-secondary w-full flex items-center justify-center gap-2 cursor-pointer">
                    <Upload className="w-4 h-4" />
                    Upload SSH Key
                    <input
                      type="file"
                      className="hidden"
                      onChange={handleKeyUpload}
                      disabled={uploadKeyMutation.isPending}
                    />
                  </label>
                )}
              </>
            )}
            {uploadKeyMutation.isError && (
              <p className="text-red-500 text-sm mt-2">Failed to upload key</p>
            )}
          </div>
        )}

        {/* Jump Host Credentials (Step 1) */}
        {nodeGroup.use_jump_host && (
          <div className="card">
            <h3 className="font-semibold mb-3">Step 1: Jump Host Auth</h3>
            <p className="text-xs text-amd-gray-500 mb-2">
              Server → {nodeGroup.jump_host}
            </p>
            {nodeGroup.jump_auth_type === 'password' ? (
              nodeGroup.has_jump_password ? (
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="text-sm">Password configured</span>
                </div>
              ) : (
                <p className="text-yellow-600 text-sm">Password not set</p>
              )
            ) : (
              <>
                {nodeGroup.jump_key_path ? (
                  <div className="flex items-center gap-2 text-green-600">
                    <CheckCircle className="w-5 h-5" />
                    <span className="text-sm">Key uploaded</span>
                  </div>
                ) : (
                  <label className="btn btn-secondary w-full flex items-center justify-center gap-2 cursor-pointer">
                    <Upload className="w-4 h-4" />
                    Upload Key
                    <input
                      type="file"
                      className="hidden"
                      onChange={handleJumpKeyUpload}
                      disabled={uploadJumpKeyMutation.isPending}
                    />
                  </label>
                )}
              </>
            )}
          </div>
        )}

        {/* GPU Node Credentials from Jump Host (Step 2) */}
        {nodeGroup.use_jump_host && (
          <div className="card">
            <h3 className="font-semibold mb-3">Step 2: GPU Node Auth</h3>
            <p className="text-xs text-amd-gray-500 mb-2">
              Jump Host → GPU Nodes
            </p>
            {nodeGroup.remote_auth_type === 'password' ? (
              nodeGroup.has_remote_password ? (
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="text-sm">Password configured</span>
                </div>
              ) : (
                <p className="text-yellow-600 text-sm">Password not set</p>
              )
            ) : (
              nodeGroup.remote_key_path ? (
                <div>
                  <div className="flex items-center gap-2 text-green-600">
                    <CheckCircle className="w-5 h-5" />
                    <span className="text-sm">Key path set</span>
                  </div>
                  <p className="text-xs text-amd-gray-500 mt-1">
                    <code className="bg-amd-gray-100 px-1 rounded">{nodeGroup.remote_key_path}</code>
                  </p>
                </div>
              ) : (
                <p className="text-yellow-600 text-sm">Key path not set</p>
              )
            )}
          </div>
        )}

        {/* Verify & Install */}
        <div className="card">
          <h3 className="font-semibold mb-3">Node Management</h3>
          {(() => {
            // Determine if credentials are configured
            let hasCredentials = false;
            let credentialMessage = '';

            if (nodeGroup.use_jump_host) {
              const hasJumpCreds = nodeGroup.jump_auth_type === 'password'
                ? nodeGroup.has_jump_password
                : !!nodeGroup.jump_key_path;
              const hasRemoteCreds = nodeGroup.remote_auth_type === 'password'
                ? nodeGroup.has_remote_password
                : !!nodeGroup.remote_key_path;

              hasCredentials = hasJumpCreds && hasRemoteCreds;
              if (!hasJumpCreds) {
                credentialMessage = nodeGroup.jump_auth_type === 'password'
                  ? 'Set jump host password first'
                  : 'Upload jump host key first';
              } else if (!hasRemoteCreds) {
                credentialMessage = nodeGroup.remote_auth_type === 'password'
                  ? 'Set GPU node password first'
                  : 'Set GPU node key path first';
              }
            } else {
              hasCredentials = nodeGroup.ssh_auth_type === 'password'
                ? nodeGroup.has_ssh_password
                : !!nodeGroup.ssh_key_path;
              if (!hasCredentials) {
                credentialMessage = nodeGroup.ssh_auth_type === 'password'
                  ? 'Set SSH password first'
                  : 'Upload SSH key first';
              }
            }

            return (
              <>
                {/* Node status summary */}
                <div className="text-sm mb-3 space-y-1">
                  <div className="flex justify-between">
                    <span>Pending:</span>
                    <span className="font-medium">{pendingNodes.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Connected:</span>
                    <span className="font-medium text-blue-600">{connectedNodes.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Active:</span>
                    <span className="font-medium text-green-600">{activeNodes.length}</span>
                  </div>
                  {unreachableNodes.length > 0 && (
                    <div className="flex justify-between">
                      <span>Unreachable:</span>
                      <span className="font-medium text-gray-500">{unreachableNodes.length}</span>
                    </div>
                  )}
                </div>

                {/* Step 1: Verify Connectivity */}
                <button
                  onClick={() => {
                    setInstallError(null)
                    verifyMutation.mutate()
                  }}
                  disabled={!hasCredentials || nodeGroup.nodes.length === 0 || verifyMutation.isPending}
                  className={`btn w-full flex items-center justify-center gap-2 mb-2 ${
                    connectedNodes.length > 0 && pendingNodes.length === 0 && unreachableNodes.length === 0
                      ? 'bg-green-600 text-white hover:bg-green-700'
                      : 'btn-secondary'
                  }`}
                >
                  {connectedNodes.length > 0 && pendingNodes.length === 0 && unreachableNodes.length === 0 ? (
                    <CheckCircle className="w-4 h-4" />
                  ) : (
                    <RefreshCw className={`w-4 h-4 ${verifyMutation.isPending ? 'animate-spin' : ''}`} />
                  )}
                  {verifyMutation.isPending
                    ? 'Verifying...'
                    : connectedNodes.length > 0 && pendingNodes.length === 0 && unreachableNodes.length === 0
                    ? `${connectedNodes.length} Verified`
                    : 'Verify Connectivity'}
                </button>

                {/* Step 2: Install on connected nodes */}
                <button
                  onClick={() => {
                    setInstallError(null)
                    installMutation.mutate(undefined)
                  }}
                  disabled={!hasCredentials || connectedNodes.length === 0 || installMutation.isPending}
                  className={`btn w-full flex items-center justify-center gap-2 mb-2 ${
                    activeNodes.length > 0 && connectedNodes.length === 0
                      ? 'bg-green-600 text-white hover:bg-green-700'
                      : 'btn-primary'
                  }`}
                >
                  {activeNodes.length > 0 && connectedNodes.length === 0 ? (
                    <CheckCircle className="w-4 h-4" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  {installMutation.isPending
                    ? 'Installing...'
                    : activeNodes.length > 0 && connectedNodes.length === 0
                    ? `${activeNodes.length} Installed`
                    : `Install on ${connectedNodes.length} connected nodes`}
                </button>

                {/* Force reinstall option */}
                {(activeNodes.length > 0 || connectedNodes.length > 0) && (
                  <button
                    onClick={() => {
                      setInstallError(null)
                      installForceMutation.mutate(undefined)
                    }}
                    disabled={!hasCredentials || installForceMutation.isPending}
                    className="btn btn-outline w-full flex items-center justify-center gap-2 text-sm"
                  >
                    {installForceMutation.isPending
                      ? 'Reinstalling...'
                      : 'Force Reinstall All'}
                  </button>
                )}

                {!hasCredentials && credentialMessage && (
                  <p className="text-yellow-600 text-sm mt-2">{credentialMessage}</p>
                )}
                {installError && (
                  <p className="text-red-600 text-sm mt-2">{installError}</p>
                )}
              </>
            );
          })()}
        </div>

        {/* Add Nodes */}
        <div className="card">
          <h3 className="font-semibold mb-3">Add Nodes</h3>
          <button
            onClick={() => setShowAddNodes(true)}
            className="btn btn-secondary w-full flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add More Nodes
          </button>
        </div>
      </div>

      {/* Nodes Table */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Nodes ({nodeGroup.nodes.length})</h2>
          <button onClick={() => refetch()} className="text-amd-gray-500 hover:text-amd-gray-700">
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>

        {nodeGroup.nodes.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-amd-gray-200">
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500">Status</th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500">IP Address</th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500">Hostname</th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500">GPUs</th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500">Ports</th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500">Message</th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500"></th>
                </tr>
              </thead>
              <tbody>
                {nodeGroup.nodes.map((node) => (
                  <tr key={node.id} className="border-b border-amd-gray-100 hover:bg-amd-gray-50">
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(node.status)}
                        {getStatusBadge(node.status)}
                      </div>
                    </td>
                    <td className="py-3 px-4 font-mono text-sm">{node.ip_address}</td>
                    <td className="py-3 px-4">{node.hostname || '-'}</td>
                    <td className="py-3 px-4">
                      {node.gpu_count !== null ? (
                        <span>
                          {node.gpu_count} x {node.gpu_model || 'Unknown'}
                        </span>
                      ) : (
                        '-'
                      )}
                    </td>
                    <td className="py-3 px-4 text-sm text-amd-gray-500">
                      GPU: {node.gpu_exporter_port}, Node: {node.node_exporter_port}
                    </td>
                    <td className="py-3 px-4 text-sm text-amd-gray-500 max-w-xs truncate">
                      {node.status_message || '-'}
                    </td>
                    <td className="py-3 px-4">
                      <button
                        onClick={() => setDeleteNodeId(node.id)}
                        className="text-amd-gray-400 hover:text-red-600"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-amd-gray-500">
            No nodes added yet. Click "Add More Nodes" to get started.
          </div>
        )}
      </div>

      {/* Add Nodes Modal */}
      {showAddNodes && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-lg w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">Add Nodes</h3>
            <textarea
              className="input h-40"
              placeholder="Enter IP addresses (one per line, or comma/semicolon separated)"
              value={newIPs}
              onChange={(e) => setNewIPs(e.target.value)}
            />
            <div className="flex justify-end gap-3 mt-4">
              <button
                onClick={() => {
                  setShowAddNodes(false)
                  setNewIPs('')
                }}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleAddNodes}
                disabled={!newIPs.trim() || addNodesMutation.isPending}
                className="btn btn-primary"
              >
                {addNodesMutation.isPending ? 'Adding...' : 'Add Nodes'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Node Modal */}
      {deleteNodeId && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-4">Delete Node?</h3>
            <p className="text-amd-gray-500 mb-6">
              This will remove the node from monitoring. You can re-add it later.
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setDeleteNodeId(null)}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={() => deleteNodeMutation.mutate(deleteNodeId)}
                disabled={deleteNodeMutation.isPending}
                className="btn btn-danger"
              >
                {deleteNodeMutation.isPending ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default NodeGroupDetail
