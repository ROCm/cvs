import { useState } from 'react'
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { createNodeGroup } from '../api'
import api from '../api'
import { Server, ArrowLeft, Plus, X, Monitor, BarChart3 } from 'lucide-react'

function AddNodeGroup() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    ssh_user: 'root',
    ssh_port: 22,
    ssh_auth_type: 'key' as 'key' | 'password',
    ssh_password: '',
    use_jump_host: false,
    jump_host: '',
    jump_port: 22,
    jump_user: '',
    jump_auth_type: 'key' as 'key' | 'password',
    jump_password: '',
    remote_auth_type: 'key' as 'key' | 'password',
    remote_key_path: '',
    remote_password: '',
    monitoring_server_id: null as number | null,
    metric_group_id: null as number | null,
  })

  const { data: monitoringServers = [] } = useQuery({
    queryKey: ['monitoring-servers'],
    queryFn: () => api.get('/monitoring-servers').then(r => r.data),
  })

  const { data: metricGroups = [] } = useQuery({
    queryKey: ['metric-groups'],
    queryFn: () => api.get('/metric-groups').then(r => r.data),
  })
  const [ipAddresses, setIpAddresses] = useState<string[]>([])
  const [ipInput, setIpInput] = useState('')
  const [bulkInput, setBulkInput] = useState('')
  const [showBulkInput, setShowBulkInput] = useState(false)

  const [error, setError] = useState<string | null>(null)

  const createMutation = useMutation({
    mutationFn: createNodeGroup,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['nodegroups'] })
      queryClient.invalidateQueries({ queryKey: ['stats'] })
      navigate(`/nodegroups/${data.id}`)
    },
    onError: (err: Error & { response?: { data?: { detail?: string } } }) => {
      console.error('Create node group error:', err)
      setError(err.response?.data?.detail || err.message || 'Failed to create node group')
    },
  })

  const handleAddIP = () => {
    const ip = ipInput.trim()
    if (ip && !ipAddresses.includes(ip)) {
      setIpAddresses([...ipAddresses, ip])
      setIpInput('')
    }
  }

  const handleRemoveIP = (ip: string) => {
    setIpAddresses(ipAddresses.filter((i) => i !== ip))
  }

  const handleBulkAdd = () => {
    const newIPs = bulkInput
      .split(/[\n,;]/)
      .map((ip) => ip.trim())
      .filter((ip) => ip && !ipAddresses.includes(ip))
    setIpAddresses([...ipAddresses, ...newIPs])
    setBulkInput('')
    setShowBulkInput(false)
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        const content = event.target?.result as string
        const newIPs = content
          .split(/[\n,;]/)
          .map((ip) => ip.trim())
          .filter((ip) => ip && !ipAddresses.includes(ip))
        setIpAddresses([...ipAddresses, ...newIPs])
      }
      reader.readAsText(file)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    // Clean up the payload - convert empty strings to undefined
    const payload = {
      name: formData.name,
      description: formData.description || undefined,
      ssh_user: formData.ssh_user,
      ssh_port: formData.ssh_port,
      ssh_auth_type: formData.ssh_auth_type,
      ssh_password: formData.ssh_password || undefined,
      use_jump_host: formData.use_jump_host,
      jump_host: formData.jump_host || undefined,
      jump_port: formData.jump_port,
      jump_user: formData.jump_user || undefined,
      jump_auth_type: formData.jump_auth_type,
      jump_password: formData.jump_password || undefined,
      remote_auth_type: formData.remote_auth_type,
      remote_key_path: formData.remote_key_path || undefined,
      remote_password: formData.remote_password || undefined,
      ip_addresses: ipAddresses,
      monitoring_server_id: formData.monitoring_server_id || undefined,
      metric_group_id: formData.metric_group_id || undefined,
    }

    console.log('Creating node group with payload:', payload)
    createMutation.mutate(payload)
  }

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <button
        onClick={() => navigate('/nodegroups')}
        className="flex items-center gap-2 text-amd-gray-500 hover:text-amd-gray-700 mb-6"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Node Groups
      </button>

      <div className="card">
        <div className="flex items-center gap-4 mb-6">
          <div className="p-3 bg-brand-100 rounded-lg">
            <Server className="w-6 h-6 text-brand-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">Create Node Group</h1>
            <p className="text-amd-gray-500">
              Add a group of GPU nodes to monitor
            </p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Basic Info */}
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="label">Group Name *</label>
              <input
                type="text"
                required
                className="input"
                placeholder="e.g., production-cluster"
                value={formData.name}
                onChange={(e) =>
                  setFormData({ ...formData, name: e.target.value })
                }
              />
            </div>
            <div>
              <label className="label">Description</label>
              <input
                type="text"
                className="input"
                placeholder="Optional description"
                value={formData.description}
                onChange={(e) =>
                  setFormData({ ...formData, description: e.target.value })
                }
              />
            </div>
          </div>

          {/* Monitoring Configuration */}
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="label flex items-center gap-2">
                <Monitor className="w-4 h-4 text-blue-600" />
                Monitoring Server
              </label>
              <select
                className="input"
                value={formData.monitoring_server_id || ''}
                onChange={(e) =>
                  setFormData({ ...formData, monitoring_server_id: e.target.value ? parseInt(e.target.value) : null })
                }
              >
                <option value="">Select a monitoring server...</option>
                {monitoringServers.map((server: any) => (
                  <option key={server.id} value={server.id}>
                    {server.name} ({server.server_ip || 'Not configured'})
                  </option>
                ))}
              </select>
              {monitoringServers.length === 0 && (
                <p className="text-sm text-yellow-600 mt-1">
                  No monitoring servers configured. Create one first.
                </p>
              )}
            </div>
            <div>
              <label className="label flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-purple-600" />
                Metric Group
              </label>
              <select
                className="input"
                value={formData.metric_group_id || ''}
                onChange={(e) =>
                  setFormData({ ...formData, metric_group_id: e.target.value ? parseInt(e.target.value) : null })
                }
              >
                <option value="">Select a metric group...</option>
                {metricGroups.map((group: any) => (
                  <option key={group.id} value={group.id}>
                    {group.name}
                  </option>
                ))}
              </select>
              {metricGroups.length === 0 && (
                <p className="text-sm text-yellow-600 mt-1">
                  No metric groups configured. Create one first.
                </p>
              )}
            </div>
          </div>

          {/* SSH Config */}
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="label">SSH Username (GPU Nodes)</label>
              <input
                type="text"
                className="input"
                value={formData.ssh_user}
                onChange={(e) =>
                  setFormData({ ...formData, ssh_user: e.target.value })
                }
              />
            </div>
            <div>
              <label className="label">SSH Port (GPU Nodes)</label>
              <input
                type="number"
                className="input"
                value={formData.ssh_port}
                onChange={(e) =>
                  setFormData({ ...formData, ssh_port: parseInt(e.target.value) })
                }
              />
            </div>
          </div>

          {/* Direct SSH Auth (when not using jump host) */}
          {!formData.use_jump_host && (
            <div className="border border-amd-gray-200 rounded-lg p-4">
              <h4 className="font-medium text-amd-gray-700 mb-3">Direct SSH Authentication</h4>
              <div className="flex items-center gap-4 mb-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="ssh_auth_type"
                    value="key"
                    checked={formData.ssh_auth_type === 'key'}
                    onChange={() => setFormData({ ...formData, ssh_auth_type: 'key', ssh_password: '' })}
                    className="w-4 h-4 text-brand-600"
                  />
                  <span>SSH Key</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="ssh_auth_type"
                    value="password"
                    checked={formData.ssh_auth_type === 'password'}
                    onChange={() => setFormData({ ...formData, ssh_auth_type: 'password' })}
                    className="w-4 h-4 text-brand-600"
                  />
                  <span>Password</span>
                </label>
              </div>
              {formData.ssh_auth_type === 'password' && (
                <div>
                  <label className="label">SSH Password</label>
                  <input
                    type="password"
                    className="input"
                    placeholder="Enter password for GPU nodes"
                    value={formData.ssh_password}
                    onChange={(e) => setFormData({ ...formData, ssh_password: e.target.value })}
                  />
                </div>
              )}
              {formData.ssh_auth_type === 'key' && (
                <p className="text-sm text-amd-gray-500">
                  You'll upload the SSH key after creating the node group.
                </p>
              )}
            </div>
          )}

          {/* Jump Host Config */}
          <div className="border border-amd-gray-200 rounded-lg p-4">
            <div className="flex items-center gap-3 mb-4">
              <input
                type="checkbox"
                id="use_jump_host"
                checked={formData.use_jump_host}
                onChange={(e) =>
                  setFormData({ ...formData, use_jump_host: e.target.checked })
                }
                className="w-4 h-4 text-brand-600 rounded"
              />
              <label htmlFor="use_jump_host" className="font-medium">
                Use Jump Host (Bastion)
              </label>
            </div>

            {formData.use_jump_host && (
              <div className="space-y-4 pl-7">
                <div className="p-3 bg-blue-50 rounded-lg text-sm text-blue-800 mb-4">
                  <strong>Connection flow:</strong> Monitoring Server → Jump Host → GPU Nodes
                </div>

                {/* Jump Host Access (Step 1) */}
                <div className="border-l-4 border-blue-400 pl-4">
                  <h4 className="font-medium text-amd-gray-700 mb-3">Step 1: Access Jump Host</h4>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <label className="label">Jump Host Address *</label>
                      <input
                        type="text"
                        className="input"
                        placeholder="bastion.example.com"
                        value={formData.jump_host}
                        onChange={(e) =>
                          setFormData({ ...formData, jump_host: e.target.value })
                        }
                      />
                    </div>
                    <div>
                      <label className="label">Jump Host Port</label>
                      <input
                        type="number"
                        className="input"
                        value={formData.jump_port}
                        onChange={(e) =>
                          setFormData({ ...formData, jump_port: parseInt(e.target.value) })
                        }
                      />
                    </div>
                  </div>
                  <div className="mt-3">
                    <label className="label">Jump Host Username *</label>
                    <input
                      type="text"
                      className="input"
                      placeholder="your_username"
                      value={formData.jump_user}
                      onChange={(e) =>
                        setFormData({ ...formData, jump_user: e.target.value })
                      }
                    />
                  </div>

                  {/* Jump Host Auth Type */}
                  <div className="mt-4">
                    <label className="label">Authentication Method</label>
                    <div className="flex items-center gap-4 mb-3">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="jump_auth_type"
                          value="key"
                          checked={formData.jump_auth_type === 'key'}
                          onChange={() => setFormData({ ...formData, jump_auth_type: 'key', jump_password: '' })}
                          className="w-4 h-4 text-brand-600"
                        />
                        <span>SSH Key</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="jump_auth_type"
                          value="password"
                          checked={formData.jump_auth_type === 'password'}
                          onChange={() => setFormData({ ...formData, jump_auth_type: 'password' })}
                          className="w-4 h-4 text-brand-600"
                        />
                        <span>Password</span>
                      </label>
                    </div>
                    {formData.jump_auth_type === 'password' && (
                      <div>
                        <input
                          type="password"
                          className="input"
                          placeholder="Enter password for jump host"
                          value={formData.jump_password}
                          onChange={(e) => setFormData({ ...formData, jump_password: e.target.value })}
                        />
                      </div>
                    )}
                    {formData.jump_auth_type === 'key' && (
                      <p className="text-sm text-amd-gray-500">
                        You'll upload the SSH key to access the jump host after creating the node group.
                      </p>
                    )}
                  </div>
                </div>

                {/* GPU Node Access from Jump Host (Step 2) */}
                <div className="border-l-4 border-green-400 pl-4 mt-4">
                  <h4 className="font-medium text-amd-gray-700 mb-3">Step 2: Access GPU Nodes (from Jump Host)</h4>

                  {/* Remote Auth Type */}
                  <div className="mb-4">
                    <label className="label">Authentication Method</label>
                    <div className="flex items-center gap-4 mb-3">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="remote_auth_type"
                          value="key"
                          checked={formData.remote_auth_type === 'key'}
                          onChange={() => setFormData({ ...formData, remote_auth_type: 'key', remote_password: '' })}
                          className="w-4 h-4 text-brand-600"
                        />
                        <span>SSH Key (on Jump Host)</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="remote_auth_type"
                          value="password"
                          checked={formData.remote_auth_type === 'password'}
                          onChange={() => setFormData({ ...formData, remote_auth_type: 'password', remote_key_path: '' })}
                          className="w-4 h-4 text-brand-600"
                        />
                        <span>Password</span>
                      </label>
                    </div>
                    {formData.remote_auth_type === 'key' && (
                      <div>
                        <label className="label">SSH Key Path on Jump Host *</label>
                        <input
                          type="text"
                          className="input"
                          placeholder="~/.ssh/gpu_nodes_key"
                          value={formData.remote_key_path}
                          onChange={(e) =>
                            setFormData({ ...formData, remote_key_path: e.target.value })
                          }
                        />
                        <p className="text-sm text-amd-gray-500 mt-1">
                          Full path to the SSH private key that exists on the jump host for accessing GPU nodes
                        </p>
                      </div>
                    )}
                    {formData.remote_auth_type === 'password' && (
                      <div>
                        <label className="label">GPU Node Password *</label>
                        <input
                          type="password"
                          className="input"
                          placeholder="Enter password for GPU nodes"
                          value={formData.remote_password}
                          onChange={(e) => setFormData({ ...formData, remote_password: e.target.value })}
                        />
                        <p className="text-sm text-amd-gray-500 mt-1">
                          Password to authenticate to GPU nodes from the jump host
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* IP Addresses */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="label mb-0">Node IP Addresses</label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setShowBulkInput(!showBulkInput)}
                  className="text-sm text-brand-600 hover:underline"
                >
                  Bulk Add
                </button>
                <label className="text-sm text-brand-600 hover:underline cursor-pointer">
                  Upload File
                  <input
                    type="file"
                    accept=".txt,.csv"
                    className="hidden"
                    onChange={handleFileUpload}
                  />
                </label>
              </div>
            </div>

            {showBulkInput ? (
              <div className="space-y-2">
                <textarea
                  className="input h-32"
                  placeholder="Enter IP addresses (one per line, or comma/semicolon separated)"
                  value={bulkInput}
                  onChange={(e) => setBulkInput(e.target.value)}
                />
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={handleBulkAdd}
                    className="btn btn-primary"
                  >
                    Add IPs
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowBulkInput(false)}
                    className="btn btn-secondary"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex gap-2">
                <input
                  type="text"
                  className="input flex-1"
                  placeholder="Enter IP address"
                  value={ipInput}
                  onChange={(e) => setIpInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), handleAddIP())}
                />
                <button
                  type="button"
                  onClick={handleAddIP}
                  className="btn btn-secondary"
                >
                  <Plus className="w-5 h-5" />
                </button>
              </div>
            )}

            {/* IP List */}
            {ipAddresses.length > 0 && (
              <div className="mt-4 p-4 bg-amd-gray-50 rounded-lg">
                <p className="text-sm text-amd-gray-500 mb-2">
                  {ipAddresses.length} IP address(es) added
                </p>
                <div className="flex flex-wrap gap-2">
                  {ipAddresses.map((ip) => (
                    <span
                      key={ip}
                      className="inline-flex items-center gap-1 px-3 py-1 bg-white border border-amd-gray-200 rounded-full text-sm"
                    >
                      {ip}
                      <button
                        type="button"
                        onClick={() => handleRemoveIP(ip)}
                        className="text-amd-gray-400 hover:text-red-600"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
              <p className="font-medium">Error creating node group:</p>
              <p className="text-sm">{error}</p>
            </div>
          )}

          {/* Submit */}
          <div className="flex justify-end gap-3 pt-4 border-t">
            <button
              type="button"
              onClick={() => navigate('/nodegroups')}
              className="btn btn-secondary"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!formData.name || createMutation.isPending}
              className="btn btn-primary"
            >
              {createMutation.isPending ? 'Creating...' : 'Create Node Group'}
            </button>
          </div>
        </form>
      </div>

      <div className="card mt-6 bg-blue-50 border-blue-200">
        <h3 className="font-medium text-blue-900 mb-2">Next Steps</h3>
        <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
          <li>Create the node group with IP addresses</li>
          <li>Upload an SSH private key for authentication</li>
          <li>Click "Install Exporters" to deploy monitoring agents</li>
          <li>View your metrics in the auto-generated Grafana dashboard</li>
        </ol>
      </div>
    </div>
  )
}

export default AddNodeGroup
