import { useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  fetchControlNodeGroup,
  uploadControlNodeGroupSSHKey,
  uploadControlNodeGroupJumpKey,
  uploadControlNodeGroupKubeconfig,
  verifyControlNodeConnectivity,
  installControlNodeExporters,
  addControlNodes,
  deleteControlNode,
  fetchMonitoringServers,
} from '../api'
import {
  ArrowLeft,
  Cpu,
  Upload,
  Play,
  Trash2,
  CheckCircle,
  FileKey,
  Clock,
  AlertTriangle,
  XCircle,
  ExternalLink,
  Plus,
  RefreshCw,
  Monitor,
} from 'lucide-react'

function ControlNodeGroupDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const groupId = parseInt(id!)

  const [showAddNodes, setShowAddNodes] = useState(false)
  const [newIPs, setNewIPs] = useState('')
  const [deleteNodeId, setDeleteNodeId] = useState<number | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  const { data: group, isLoading, refetch } = useQuery({
    queryKey: ['control-nodegroup', groupId],
    queryFn: () => fetchControlNodeGroup(groupId),
    refetchInterval: 10000,
  })

  const { data: monitoringServers = [] } = useQuery({
    queryKey: ['monitoring-servers'],
    queryFn: fetchMonitoringServers,
  })

  const uploadKeyMutation = useMutation({
    mutationFn: (file: File) => uploadControlNodeGroupSSHKey(groupId, file),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['control-nodegroup', groupId] }),
  })

  const uploadJumpKeyMutation = useMutation({
    mutationFn: (file: File) => uploadControlNodeGroupJumpKey(groupId, file),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['control-nodegroup', groupId] }),
  })

  const uploadKubeconfigMutation = useMutation({
    mutationFn: (file: File) => uploadControlNodeGroupKubeconfig(groupId, file),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['control-nodegroup', groupId] }),
  })

  type ApiError = Error & { response?: { data?: { detail?: string } } }

  const verifyMutation = useMutation({
    mutationFn: () => verifyControlNodeConnectivity(groupId),
    onSuccess: () => {
      setActionError(null)
      queryClient.invalidateQueries({ queryKey: ['control-nodegroup', groupId] })
    },
    onError: (err: ApiError) => {
      setActionError(err.response?.data?.detail || err.message || 'Verification failed')
    },
  })

  const installMutation = useMutation({
    mutationFn: (force: boolean) => installControlNodeExporters(groupId, force),
    onSuccess: () => {
      setActionError(null)
      queryClient.invalidateQueries({ queryKey: ['control-nodegroup', groupId] })
    },
    onError: (err: ApiError) => {
      setActionError(err.response?.data?.detail || err.message || 'Installation failed')
    },
  })

  const addNodesMutation = useMutation({
    mutationFn: (ips: string[]) => addControlNodes(groupId, ips),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['control-nodegroup', groupId] })
      setShowAddNodes(false)
      setNewIPs('')
    },
  })

  const deleteNodeMutation = useMutation({
    mutationFn: (nodeId: number) => deleteControlNode(groupId, nodeId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['control-nodegroup', groupId] })
      setDeleteNodeId(null)
    },
  })

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
        return <XCircle className="w-5 h-5 text-gray-400" />
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
      <span
        className={`px-2 py-0.5 rounded-full text-xs font-medium ${
          styles[status] || styles.unreachable
        }`}
      >
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

  if (!group) {
    return (
      <div className="p-8 text-center">
        <p className="text-amd-gray-500 mb-4">Control node group not found.</p>
        <Link to="/control-nodegroups" className="text-brand-600 hover:underline">
          Back to Control Node Groups
        </Link>
      </div>
    )
  }

  const connectedNodes = group.nodes.filter((n) => n.status === 'connected')
  const activeNodes = group.nodes.filter((n) => n.status === 'active')
  const pendingNodes = group.nodes.filter((n) => n.status === 'pending')
  const errorNodes = group.nodes.filter((n) => n.status === 'error')

  const monitoringServer = monitoringServers.find((s) => s.id === group.monitoring_server_id)
  const grafanaBase =
    monitoringServer?.server_ip
      ? `http://${monitoringServer.server_ip}:${monitoringServer.grafana_port}`
      : ''
  const safeName = group.name.replace(/[^a-zA-Z0-9]/g, '_')
  const dashboardUrl = grafanaBase ? `${grafanaBase}/d/cng_${safeName}` : ''

  const hasCredentials = (() => {
    if (group.use_jump_host) {
      const jumpOk =
        group.jump_auth_type === 'password'
          ? group.has_jump_password
          : !!group.jump_key_path
      const remoteOk =
        group.remote_auth_type === 'password'
          ? group.has_remote_password
          : !!group.remote_key_path
      return jumpOk && remoteOk
    }
    return group.ssh_auth_type === 'password'
      ? group.has_ssh_password
      : !!group.ssh_key_path
  })()

  const handleAddNodes = () => {
    const ips = newIPs
      .split(/[\n,;]/)
      .map((ip) => ip.trim())
      .filter((ip) => ip)
    if (ips.length > 0) addNodesMutation.mutate(ips)
  }

  const controlTypeBadge = (
    <span
      className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
        group.control_type === 'slurm'
          ? 'bg-orange-100 text-orange-700'
          : 'bg-blue-100 text-blue-700'
      }`}
    >
      {group.control_type === 'slurm' ? 'Slurm Cluster' : 'Kubernetes Control Plane'}
    </span>
  )

  const exporterName =
    group.control_type === 'slurm' ? 'slurm_exporter' : 'k8s_cp_exporter'
  const exporterPort =
    group.custom_exporter_port > 0
      ? group.custom_exporter_port
      : group.control_type === 'slurm'
      ? 9418
      : 9419

  return (
    <div className="p-8">
      <button
        onClick={() => navigate('/control-nodegroups')}
        className="flex items-center gap-2 text-amd-gray-500 hover:text-amd-gray-700 mb-6 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Control Node Groups
      </button>

      {/* Header card */}
      <div className="card mb-6">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-brand-100 rounded-lg">
              <Cpu className="w-8 h-8 text-brand-600" />
            </div>
            <div>
              <div className="flex items-center gap-3 mb-1">
                <h1 className="text-2xl font-bold text-amd-gray-900">{group.name}</h1>
                {controlTypeBadge}
              </div>
              {group.description && (
                <p className="text-amd-gray-500 text-sm">{group.description}</p>
              )}
              <div className="flex items-center gap-4 mt-2 text-sm text-amd-gray-500">
                <span>{group.node_count} nodes total</span>
                <span className="text-green-600">{activeNodes.length} active</span>
                <span>
                  SSH: {group.ssh_user}@:{group.ssh_port}
                </span>
                {group.use_jump_host && group.jump_host && (
                  <span className="text-blue-600">via {group.jump_host}</span>
                )}
              </div>
              <div className="flex items-center gap-4 mt-1 text-sm text-amd-gray-500">
                <span>node_exporter: 9100</span>
                <span>
                  {exporterName}: {exporterPort}
                </span>
              </div>
              {monitoringServer && (
                <div className="flex items-center gap-2 mt-1 text-sm">
                  <Monitor className="w-4 h-4 text-brand-600" />
                  <span className="text-brand-600">{monitoringServer.name}</span>
                </div>
              )}
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
            <p className="text-sm text-amd-gray-400">
              Configure a monitoring server to enable dashboard
            </p>
          )}
        </div>
      </div>

      {/* Actions grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
        {/* Direct SSH auth card */}
        {!group.use_jump_host && (
          <div className="card">
            <h3 className="font-semibold mb-3">SSH Authentication</h3>
            {group.ssh_auth_type === 'password' ? (
              group.has_ssh_password ? (
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="text-sm">Password configured</span>
                </div>
              ) : (
                <p className="text-yellow-600 text-sm">Password not set</p>
              )
            ) : group.ssh_key_path ? (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle className="w-5 h-5" />
                <span className="text-sm">SSH key uploaded</span>
              </div>
            ) : (
              <label className="btn btn-secondary w-full flex items-center justify-center gap-2 cursor-pointer">
                <Upload className="w-4 h-4" />
                Upload SSH Key
                <input
                  type="file"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f) uploadKeyMutation.mutate(f)
                  }}
                  disabled={uploadKeyMutation.isPending}
                />
              </label>
            )}
          </div>
        )}

        {/* Jump host step 1 */}
        {group.use_jump_host && (
          <div className="card">
            <h3 className="font-semibold mb-1">Step 1: Jump Host Auth</h3>
            <p className="text-xs text-amd-gray-500 mb-3">via {group.jump_host}</p>
            {group.jump_auth_type === 'password' ? (
              group.has_jump_password ? (
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="text-sm">Password configured</span>
                </div>
              ) : (
                <p className="text-yellow-600 text-sm">Password not set</p>
              )
            ) : group.jump_key_path ? (
              <div className="flex items-center gap-2 text-green-600">
                <CheckCircle className="w-5 h-5" />
                <span className="text-sm">Jump key uploaded</span>
              </div>
            ) : (
              <label className="btn btn-secondary w-full flex items-center justify-center gap-2 cursor-pointer">
                <Upload className="w-4 h-4" />
                Upload Jump Key
                <input
                  type="file"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f) uploadJumpKeyMutation.mutate(f)
                  }}
                  disabled={uploadJumpKeyMutation.isPending}
                />
              </label>
            )}
          </div>
        )}

        {/* Jump host step 2 */}
        {group.use_jump_host && (
          <div className="card">
            <h3 className="font-semibold mb-3">Step 2: Node Authentication</h3>
            {group.remote_auth_type === 'password' ? (
              group.has_remote_password ? (
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="text-sm">Password configured</span>
                </div>
              ) : (
                <p className="text-yellow-600 text-sm">Password not set</p>
              )
            ) : group.remote_key_path ? (
              <div>
                <div className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="w-5 h-5" />
                  <span className="text-sm">Key path configured</span>
                </div>
                <p className="text-xs text-amd-gray-500 mt-1">
                  <code className="bg-amd-gray-100 px-1 rounded">{group.remote_key_path}</code>
                </p>
              </div>
            ) : (
              <p className="text-yellow-600 text-sm">Remote key path not set</p>
            )}
          </div>
        )}

        {/* Verify & Install card */}
        <div className="card">
          <h3 className="font-semibold mb-3">Node Management</h3>
          <div className="text-sm mb-4 space-y-1 text-amd-gray-600">
            <div className="flex justify-between">
              <span>Pending:</span>
              <span>{pendingNodes.length}</span>
            </div>
            <div className="flex justify-between">
              <span>Connected:</span>
              <span className="text-blue-600 font-medium">{connectedNodes.length}</span>
            </div>
            <div className="flex justify-between">
              <span>Active:</span>
              <span className="text-green-600 font-medium">{activeNodes.length}</span>
            </div>
            {errorNodes.length > 0 && (
              <div className="flex justify-between">
                <span>Error:</span>
                <span className="text-red-600">{errorNodes.length}</span>
              </div>
            )}
          </div>

          <button
            onClick={() => {
              setActionError(null)
              verifyMutation.mutate()
            }}
            disabled={!hasCredentials || group.nodes.length === 0 || verifyMutation.isPending}
            className="btn btn-secondary w-full flex items-center justify-center gap-2 mb-2"
          >
            <RefreshCw
              className={`w-4 h-4 ${verifyMutation.isPending ? 'animate-spin' : ''}`}
            />
            {verifyMutation.isPending ? 'Verifying...' : 'Verify Connectivity'}
          </button>

          <button
            onClick={() => {
              setActionError(null)
              installMutation.mutate(false)
            }}
            disabled={
              !hasCredentials || connectedNodes.length === 0 || installMutation.isPending
            }
            className="btn btn-primary w-full flex items-center justify-center gap-2 mb-2"
          >
            <Play className="w-4 h-4" />
            {installMutation.isPending
              ? 'Installing...'
              : `Install on ${connectedNodes.length} connected`}
          </button>

          {(activeNodes.length > 0 || connectedNodes.length > 0) && (
            <button
              onClick={() => {
                setActionError(null)
                installMutation.mutate(true)
              }}
              disabled={!hasCredentials || installMutation.isPending}
              className="btn btn-outline w-full flex items-center justify-center gap-2 text-sm"
            >
              {installMutation.isPending ? 'Reinstalling...' : 'Force Reinstall All'}
            </button>
          )}

          {actionError && (
            <p className="text-red-600 text-sm mt-2">{actionError}</p>
          )}
        </div>

        {/* Kubeconfig card (Kubernetes only) */}
        {group.control_type === 'kubernetes' && (
          <div className="card border-blue-200">
            <div className="flex items-center gap-2 mb-3">
              <FileKey className="w-5 h-5 text-blue-600" />
              <h3 className="font-semibold">Kubeconfig</h3>
              <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                group.kubeconfig_source === 'auto' ? 'bg-gray-100 text-gray-600' :
                group.has_kubeconfig_upload || group.kubeconfig_remote_path ? 'bg-green-100 text-green-700' :
                'bg-yellow-100 text-yellow-700'
              }`}>
                {group.kubeconfig_source === 'auto' && 'Auto-detect'}
                {group.kubeconfig_source === 'path' && 'Path on node'}
                {group.kubeconfig_source === 'upload' && (group.has_kubeconfig_upload ? 'Uploaded ✓' : 'Upload pending')}
              </span>
            </div>

            {group.kubeconfig_source === 'auto' && (
              <p className="text-sm text-amd-gray-500 mb-3">
                Exporter will auto-detect <code className="bg-amd-gray-100 px-1 rounded text-xs">/etc/kubernetes/admin.conf</code> or <code className="bg-amd-gray-100 px-1 rounded text-xs">~/.kube/config</code>.
              </p>
            )}

            {group.kubeconfig_source === 'path' && (
              <div className="mb-3">
                <p className="text-xs text-amd-gray-500 mb-1">Path on the K8s node:</p>
                <code className="text-xs bg-amd-gray-100 px-2 py-1 rounded block">
                  {group.kubeconfig_remote_path || '(not set)'}
                </code>
              </div>
            )}

            {group.kubeconfig_source === 'upload' && (
              <div className="mb-3">
                {group.has_kubeconfig_upload ? (
                  <div className="flex items-center gap-2 text-green-600 text-sm mb-2">
                    <CheckCircle className="w-4 h-4" />
                    <span>Kubeconfig uploaded. Force Reinstall to deploy it.</span>
                  </div>
                ) : (
                  <label className="btn btn-secondary w-full flex items-center justify-center gap-2 cursor-pointer">
                    <Upload className="w-4 h-4" />
                    Upload Kubeconfig File
                    <input
                      type="file"
                      className="hidden"
                      accept=".yaml,.yml,.conf,"
                      onChange={(e) => {
                        const f = e.target.files?.[0]
                        if (f) uploadKubeconfigMutation.mutate(f)
                      }}
                      disabled={uploadKubeconfigMutation.isPending}
                    />
                  </label>
                )}
                {uploadKubeconfigMutation.isPending && (
                  <p className="text-sm text-blue-600 mt-1">Uploading...</p>
                )}
              </div>
            )}

            {group.has_kubeconfig_upload && (
              <label className="btn btn-outline w-full flex items-center justify-center gap-2 cursor-pointer text-sm mt-1">
                <Upload className="w-4 h-4" />
                Replace Kubeconfig
                <input
                  type="file"
                  className="hidden"
                  accept=".yaml,.yml,.conf,"
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f) uploadKubeconfigMutation.mutate(f)
                  }}
                  disabled={uploadKubeconfigMutation.isPending}
                />
              </label>
            )}
          </div>
        )}

        {/* Add nodes card */}
        <div className="card">
          <h3 className="font-semibold mb-3">Add Nodes</h3>
          <p className="text-sm text-amd-gray-500 mb-3">
            Add more control plane node IPs to this group.
          </p>
          <button
            onClick={() => setShowAddNodes(true)}
            className="btn btn-secondary w-full flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" />
            Add Nodes
          </button>
        </div>
      </div>

      {/* Nodes table */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">
            Nodes ({group.nodes.length})
          </h2>
          <button
            onClick={() => refetch()}
            className="text-amd-gray-400 hover:text-amd-gray-600 transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>

        {group.nodes.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-amd-gray-200">
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500 text-sm">
                    Status
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500 text-sm">
                    IP Address
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500 text-sm">
                    Hostname
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500 text-sm">
                    Message
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-amd-gray-500 text-sm">
                    Role Info
                  </th>
                  <th className="py-3 px-4" />
                </tr>
              </thead>
              <tbody>
                {group.nodes.map((node) => (
                  <tr
                    key={node.id}
                    className="border-b border-amd-gray-100 hover:bg-amd-gray-50"
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(node.status)}
                        {getStatusBadge(node.status)}
                      </div>
                    </td>
                    <td className="py-3 px-4 font-mono text-sm text-amd-gray-700">
                      {node.ip_address}
                    </td>
                    <td className="py-3 px-4 text-sm text-amd-gray-700">
                      {node.hostname || '—'}
                    </td>
                    <td className="py-3 px-4 text-sm text-amd-gray-500 max-w-xs truncate">
                      {node.status_message || '—'}
                    </td>
                    <td className="py-3 px-4 text-sm text-amd-gray-500">
                      {node.role_info ? (
                        <span className="font-mono text-xs bg-amd-gray-100 px-2 py-1 rounded">
                          {JSON.stringify(node.role_info).slice(0, 60)}
                        </span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="py-3 px-4">
                      <button
                        onClick={() => setDeleteNodeId(node.id)}
                        className="text-amd-gray-400 hover:text-red-600 transition-colors"
                        title="Remove node"
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
          <div className="text-center py-10 text-amd-gray-500">
            <Cpu className="w-10 h-10 mx-auto mb-3 text-amd-gray-300" />
            <p>No nodes added yet. Click "Add Nodes" to get started.</p>
          </div>
        )}
      </div>

      {/* Add Nodes Modal */}
      {showAddNodes && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-lg w-full mx-4 shadow-xl">
            <h3 className="text-lg font-semibold mb-4">Add Control Nodes</h3>
            <p className="text-sm text-amd-gray-500 mb-3">
              Enter IP addresses (one per line, or comma/semicolon separated):
            </p>
            <textarea
              className="input h-40 font-mono text-sm"
              placeholder="10.0.0.1&#10;10.0.0.2&#10;10.0.0.3"
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
      {deleteNodeId !== null && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 className="text-lg font-semibold mb-2">Remove Node?</h3>
            <p className="text-amd-gray-500 mb-6">
              Remove this node from the control node group. Prometheus targets will be
              updated automatically.
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
                {deleteNodeMutation.isPending ? 'Removing...' : 'Remove'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ControlNodeGroupDetail
