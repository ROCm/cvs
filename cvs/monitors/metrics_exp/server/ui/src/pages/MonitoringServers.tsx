import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Plus,
  Monitor,
  Trash2,
  CheckCircle,
  XCircle,
  RefreshCw,
  Play,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import api from '../api'

interface MonitoringServer {
  id: number
  name: string
  description?: string
  server_ip?: string
  server_hostname?: string
  prometheus_port: number
  loki_port: number
  grafana_port: number
  prometheus_retention_time: string
  prometheus_retention_size: string
  prometheus_scrape_interval: string
  loki_retention_days: number
  grafana_admin_user: string
  setup_monitoring_stack: boolean
  ssh_user?: string
  ssh_port: number
  ssh_auth_type: string
  has_ssh_key: boolean
  has_ssh_password: boolean
  use_jump_host: boolean
  jump_host?: string
  jump_port: number
  jump_user?: string
  jump_auth_type: string
  has_jump_key: boolean
  has_jump_password: boolean
  remote_auth_type: string
  remote_key_path?: string
  has_remote_password: boolean
  use_push_gateway: boolean
  push_gateway_port: number
  stack_installed: boolean
  last_install_at?: string
  created_at: string
  updated_at: string
  node_group_count: number
}

function MonitoringServers() {
  const queryClient = useQueryClient()
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [expandedServer, setExpandedServer] = useState<number | null>(null)
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    server_ip: '',
    server_hostname: '',
    prometheus_port: 30090,
    loki_port: 30100,
    grafana_port: 30030,
    prometheus_retention_time: '15d',
    prometheus_retention_size: '50GB',
    prometheus_scrape_interval: '15s',
    loki_retention_days: 7,
    grafana_admin_user: 'admin',
    grafana_admin_password: 'admin',
    setup_monitoring_stack: false,
    ssh_user: '',
    ssh_port: 22,
    ssh_auth_type: 'password',
    ssh_password: '',
    use_jump_host: false,
    jump_host: '',
    jump_port: 22,
    jump_user: '',
    jump_auth_type: 'key',
    jump_password: '',
    remote_auth_type: 'key',
    remote_key_path: '',
    remote_password: '',
    use_push_gateway: false,
    push_gateway_port: 9091,
  })

  const [installStatus, setInstallStatus] = useState<{ [key: number]: any }>({})
  const [serviceStatus, setServiceStatus] = useState<{ [key: number]: any }>({})
  const [uploadingKey, setUploadingKey] = useState<{ serverId: number; type: 'ssh' | 'jump' } | null>(null)
  const [sshKeyFile, setSshKeyFile] = useState<File | null>(null)
  const [jumpKeyFile, setJumpKeyFile] = useState<File | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [syncingTargets, setSyncingTargets] = useState<{ [key: number]: boolean }>({})
  const [checkingServices, setCheckingServices] = useState<{ [key: number]: boolean }>({})
  const [targetsSynced, setTargetsSynced] = useState<{ [key: number]: boolean }>({})

  const { data: servers = [], isLoading } = useQuery({
    queryKey: ['monitoring-servers'],
    queryFn: () => api.get<MonitoringServer[]>('/monitoring-servers').then(r => r.data),
  })

  const createMutation = useMutation({
    mutationFn: async (data: typeof formData) => {
      // First create the server
      const response = await api.post('/monitoring-servers', data)
      const serverId = response.data.id

      // Then upload SSH key if provided
      if (sshKeyFile && data.ssh_auth_type === 'key') {
        const keyFormData = new FormData()
        keyFormData.append('key_file', sshKeyFile)
        await api.post(`/monitoring-servers/${serverId}/ssh-key`, keyFormData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
      }

      // Upload jump host key if provided
      if (jumpKeyFile && data.use_jump_host && data.jump_auth_type === 'key') {
        const keyFormData = new FormData()
        keyFormData.append('key_file', jumpKeyFile)
        await api.post(`/monitoring-servers/${serverId}/jump-key`, keyFormData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        })
      }

      return response
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['monitoring-servers'] })
      setShowCreateForm(false)
      setSshKeyFile(null)
      setJumpKeyFile(null)
      resetForm()
      setSuccessMessage('Monitoring server created successfully!')
      setErrorMessage(null)
      // Clear success message after 5 seconds
      setTimeout(() => setSuccessMessage(null), 5000)
    },
    onError: (error: Error & { response?: { data?: { detail?: string } } }) => {
      setErrorMessage(error.response?.data?.detail || error.message || 'Failed to create monitoring server')
      setSuccessMessage(null)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) => api.delete(`/monitoring-servers/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['monitoring-servers'] })
    },
  })

  const uploadSshKey = async (serverId: number, file: File, type: 'ssh' | 'jump') => {
    const formData = new FormData()
    formData.append('key_file', file)
    const endpoint = type === 'ssh' ? 'ssh-key' : 'jump-key'
    try {
      setUploadingKey({ serverId, type })
      await api.post(`/monitoring-servers/${serverId}/${endpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      queryClient.invalidateQueries({ queryKey: ['monitoring-servers'] })
    } finally {
      setUploadingKey(null)
    }
  }

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      server_ip: '',
      server_hostname: '',
      prometheus_port: 9090,
      loki_port: 3100,
      grafana_port: 3000,
      prometheus_retention_time: '15d',
      prometheus_retention_size: '50GB',
      prometheus_scrape_interval: '15s',
      loki_retention_days: 7,
      grafana_admin_user: 'admin',
      grafana_admin_password: 'admin',
      setup_monitoring_stack: false,
      ssh_user: '',
      ssh_port: 22,
      ssh_auth_type: 'password',
      ssh_password: '',
      use_jump_host: false,
      jump_host: '',
      jump_port: 22,
      jump_user: '',
      jump_auth_type: 'key',
      jump_password: '',
      remote_auth_type: 'key',
      remote_key_path: '',
      remote_password: '',
      use_push_gateway: false,
      push_gateway_port: 9091,
    })
    setSshKeyFile(null)
    setJumpKeyFile(null)
  }

  const checkServices = async (serverId: number) => {
    try {
      setCheckingServices(prev => ({ ...prev, [serverId]: true }))
      const response = await api.post(`/monitoring-servers/${serverId}/check-services`)
      setServiceStatus(prev => ({ ...prev, [serverId]: response.data }))
    } catch (error: any) {
      setServiceStatus(prev => ({ ...prev, [serverId]: { error: error.message } }))
    } finally {
      setCheckingServices(prev => ({ ...prev, [serverId]: false }))
    }
  }

  const installStack = async (serverId: number) => {
    try {
      const response = await api.post(`/monitoring-servers/${serverId}/install-stack`)
      const jobId = response.data.job_id

      // Poll for status
      const pollStatus = async () => {
        try {
          const statusResponse = await api.get(`/monitoring-servers/${serverId}/install-status/${jobId}`)
          setInstallStatus(prev => ({ ...prev, [serverId]: statusResponse.data }))

          if (!statusResponse.data.completed) {
            setTimeout(pollStatus, 2000)
          } else {
            queryClient.invalidateQueries({ queryKey: ['monitoring-servers'] })
          }
        } catch (error) {
          console.error('Failed to get install status:', error)
        }
      }

      pollStatus()
    } catch (error: any) {
      setInstallStatus(prev => ({ ...prev, [serverId]: { error: error.message } }))
    }
  }

  const syncTargets = async (serverId: number) => {
    try {
      setSyncingTargets(prev => ({ ...prev, [serverId]: true }))
      const response = await api.post(`/monitoring-servers/${serverId}/sync-targets`)
      // Track targets_synced separately so it persists across other button clicks
      setTargetsSynced(prev => ({ ...prev, [serverId]: true }))
      setSuccessMessage(`Synced ${response.data.targets_count} targets successfully!`)
      setTimeout(() => setSuccessMessage(null), 5000)
    } catch (error: any) {
      setErrorMessage('Failed to sync targets: ' + error.message)
      setTimeout(() => setErrorMessage(null), 5000)
    } finally {
      setSyncingTargets(prev => ({ ...prev, [serverId]: false }))
    }
  }

  if (isLoading) {
    return (
      <div className="p-8">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="w-8 h-8 animate-spin text-brand-600" />
        </div>
      </div>
    )
  }

  return (
    <div className="p-8">
      {/* Success Message */}
      {successMessage && (
        <div className="mb-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded-lg flex items-center gap-2">
          <CheckCircle className="w-5 h-5" />
          {successMessage}
        </div>
      )}

      {/* Error Message */}
      {errorMessage && (
        <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg flex items-center gap-2">
          <XCircle className="w-5 h-5" />
          {errorMessage}
          <button onClick={() => setErrorMessage(null)} className="ml-auto text-red-700 hover:text-red-900">
            &times;
          </button>
        </div>
      )}

      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-amd-gray-900">Monitoring Servers</h1>
          <p className="text-amd-gray-600 mt-1">
            Configure Prometheus, Grafana, and Loki instances for monitoring your GPU fleet
          </p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="btn btn-primary flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          Add Server
        </button>
      </div>

      {/* Create Form Modal */}
      {showCreateForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 overflow-auto py-8">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            <h2 className="text-xl font-bold mb-4">Add Monitoring Server</h2>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Name *</label>
                  <input
                    type="text"
                    className="input"
                    value={formData.name}
                    onChange={e => setFormData(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="production-monitoring"
                  />
                </div>
                <div>
                  <label className="label">Description</label>
                  <input
                    type="text"
                    className="input"
                    value={formData.description}
                    onChange={e => setFormData(prev => ({ ...prev, description: e.target.value }))}
                    placeholder="Production monitoring stack"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="label">Server IP *</label>
                  <input
                    type="text"
                    className="input"
                    value={formData.server_ip}
                    onChange={e => setFormData(prev => ({ ...prev, server_ip: e.target.value }))}
                    placeholder="10.0.0.1"
                  />
                </div>
                <div>
                  <label className="label">Hostname (optional)</label>
                  <input
                    type="text"
                    className="input"
                    value={formData.server_hostname}
                    onChange={e => setFormData(prev => ({ ...prev, server_hostname: e.target.value }))}
                    placeholder="monitoring.example.com"
                  />
                </div>
              </div>

              <div className="border-t pt-4 mt-4">
                <h3 className="font-semibold mb-3">Service Ports</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="label">Prometheus Port</label>
                    <input
                      type="number"
                      className="input"
                      value={formData.prometheus_port}
                      onChange={e => setFormData(prev => ({ ...prev, prometheus_port: parseInt(e.target.value) }))}
                    />
                  </div>
                  <div>
                    <label className="label">Loki Port</label>
                    <input
                      type="number"
                      className="input"
                      value={formData.loki_port}
                      onChange={e => setFormData(prev => ({ ...prev, loki_port: parseInt(e.target.value) }))}
                    />
                  </div>
                  <div>
                    <label className="label">Grafana Port</label>
                    <input
                      type="number"
                      className="input"
                      value={formData.grafana_port}
                      onChange={e => setFormData(prev => ({ ...prev, grafana_port: parseInt(e.target.value) }))}
                    />
                  </div>
                </div>
              </div>

              <div className="border-t pt-4 mt-4">
                <label className="flex items-center gap-2 mb-4">
                  <input
                    type="checkbox"
                    checked={formData.setup_monitoring_stack}
                    onChange={e => setFormData(prev => ({ ...prev, setup_monitoring_stack: e.target.checked }))}
                    className="rounded border-amd-gray-300"
                  />
                  <span className="font-semibold">Install monitoring stack on remote server</span>
                </label>

                {formData.setup_monitoring_stack && (
                  <div className="bg-amd-gray-50 p-4 rounded-lg space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="label">SSH User</label>
                        <input
                          type="text"
                          className="input"
                          value={formData.ssh_user}
                          onChange={e => setFormData(prev => ({ ...prev, ssh_user: e.target.value }))}
                          placeholder="root"
                        />
                      </div>
                      <div>
                        <label className="label">SSH Port</label>
                        <input
                          type="number"
                          className="input"
                          value={formData.ssh_port}
                          onChange={e => setFormData(prev => ({ ...prev, ssh_port: parseInt(e.target.value) }))}
                        />
                      </div>
                    </div>

                    <div>
                      <label className="label">SSH Authentication</label>
                      <select
                        className="input"
                        value={formData.ssh_auth_type}
                        onChange={e => setFormData(prev => ({ ...prev, ssh_auth_type: e.target.value }))}
                      >
                        <option value="password">Password</option>
                        <option value="key">SSH Key</option>
                      </select>
                    </div>

                    {formData.ssh_auth_type === 'password' && (
                      <div>
                        <label className="label">SSH Password</label>
                        <input
                          type="password"
                          className="input"
                          value={formData.ssh_password}
                          onChange={e => setFormData(prev => ({ ...prev, ssh_password: e.target.value }))}
                        />
                      </div>
                    )}

                    {formData.ssh_auth_type === 'key' && (
                      <div>
                        <label className="label">SSH Private Key File</label>
                        <div className="flex items-center gap-3">
                          <label className="flex-1 flex items-center justify-center px-4 py-2 border-2 border-dashed border-amd-gray-300 rounded-lg cursor-pointer hover:border-brand-500 transition-colors">
                            <input
                              type="file"
                              className="hidden"
                              onChange={(e) => setSshKeyFile(e.target.files?.[0] || null)}
                            />
                            <span className="text-sm text-amd-gray-600">
                              {sshKeyFile ? sshKeyFile.name : 'Click to select SSH key file...'}
                            </span>
                          </label>
                          {sshKeyFile && (
                            <button
                              type="button"
                              onClick={() => setSshKeyFile(null)}
                              className="text-red-500 hover:text-red-700"
                            >
                              <XCircle className="w-5 h-5" />
                            </button>
                          )}
                        </div>
                      </div>
                    )}

                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={formData.use_jump_host}
                        onChange={e => setFormData(prev => ({ ...prev, use_jump_host: e.target.checked }))}
                        className="rounded border-amd-gray-300"
                      />
                      <span>Use Jump Host (Bastion)</span>
                    </label>

                    {formData.use_jump_host && (
                      <div className="bg-white p-4 rounded-lg space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <label className="label">Jump Host</label>
                            <input
                              type="text"
                              className="input"
                              value={formData.jump_host}
                              onChange={e => setFormData(prev => ({ ...prev, jump_host: e.target.value }))}
                              placeholder="bastion.example.com"
                            />
                          </div>
                          <div>
                            <label className="label">Jump Port</label>
                            <input
                              type="number"
                              className="input"
                              value={formData.jump_port}
                              onChange={e => setFormData(prev => ({ ...prev, jump_port: parseInt(e.target.value) }))}
                            />
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <label className="label">Jump User</label>
                            <input
                              type="text"
                              className="input"
                              value={formData.jump_user}
                              onChange={e => setFormData(prev => ({ ...prev, jump_user: e.target.value }))}
                            />
                          </div>
                          <div>
                            <label className="label">Jump Auth Type</label>
                            <select
                              className="input"
                              value={formData.jump_auth_type}
                              onChange={e => setFormData(prev => ({ ...prev, jump_auth_type: e.target.value }))}
                            >
                              <option value="key">SSH Key</option>
                              <option value="password">Password</option>
                            </select>
                          </div>
                        </div>
                        {formData.jump_auth_type === 'password' && (
                          <div>
                            <label className="label">Jump Host Password</label>
                            <input
                              type="password"
                              className="input"
                              value={formData.jump_password}
                              onChange={e => setFormData(prev => ({ ...prev, jump_password: e.target.value }))}
                            />
                          </div>
                        )}
                        {formData.jump_auth_type === 'key' && (
                          <div>
                            <label className="label">Jump Host SSH Key File</label>
                            <div className="flex items-center gap-3">
                              <label className="flex-1 flex items-center justify-center px-4 py-2 border-2 border-dashed border-amd-gray-300 rounded-lg cursor-pointer hover:border-brand-500 transition-colors">
                                <input
                                  type="file"
                                  className="hidden"
                                  onChange={(e) => setJumpKeyFile(e.target.files?.[0] || null)}
                                />
                                <span className="text-sm text-amd-gray-600">
                                  {jumpKeyFile ? jumpKeyFile.name : 'Click to select SSH key file...'}
                                </span>
                              </label>
                              {jumpKeyFile && (
                                <button
                                  type="button"
                                  onClick={() => setJumpKeyFile(null)}
                                  className="text-red-500 hover:text-red-700"
                                >
                                  <XCircle className="w-5 h-5" />
                                </button>
                              )}
                            </div>
                          </div>
                        )}
                        <div>
                          <label className="label">Remote Auth Type (from Jump to Target)</label>
                          <select
                            className="input"
                            value={formData.remote_auth_type}
                            onChange={e => setFormData(prev => ({ ...prev, remote_auth_type: e.target.value }))}
                          >
                            <option value="key">SSH Key (on jump host)</option>
                            <option value="password">Password</option>
                          </select>
                        </div>
                        {formData.remote_auth_type === 'key' && (
                          <div>
                            <label className="label">Remote Key Path (on jump host)</label>
                            <input
                              type="text"
                              className="input"
                              value={formData.remote_key_path}
                              onChange={e => setFormData(prev => ({ ...prev, remote_key_path: e.target.value }))}
                              placeholder="~/.ssh/id_rsa"
                            />
                          </div>
                        )}
                        {formData.remote_auth_type === 'password' && (
                          <div>
                            <label className="label">Remote Password</label>
                            <input
                              type="password"
                              className="input"
                              value={formData.remote_password}
                              onChange={e => setFormData(prev => ({ ...prev, remote_password: e.target.value }))}
                            />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => {
                  setShowCreateForm(false)
                  resetForm()
                }}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={() => createMutation.mutate(formData)}
                disabled={!formData.name || !formData.server_ip || createMutation.isPending}
                className="btn btn-primary"
              >
                {createMutation.isPending ? 'Creating...' : 'Create Server'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Server List */}
      {servers.length === 0 ? (
        <div className="card text-center py-12">
          <Monitor className="w-16 h-16 mx-auto text-amd-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-amd-gray-900 mb-2">No Monitoring Servers</h3>
          <p className="text-amd-gray-600 mb-4">
            Add a monitoring server to start collecting metrics from your GPU fleet.
          </p>
          <button
            onClick={() => setShowCreateForm(true)}
            className="btn btn-primary inline-flex items-center gap-2"
          >
            <Plus className="w-5 h-5" />
            Add Your First Server
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {servers.map(server => (
            <div key={server.id} className="card">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                    server.stack_installed ? 'bg-green-100' : 'bg-amd-gray-100'
                  }`}>
                    <Monitor className={`w-6 h-6 ${
                      server.stack_installed ? 'text-green-600' : 'text-amd-gray-600'
                    }`} />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-amd-gray-900">{server.name}</h3>
                    <p className="text-sm text-amd-gray-600">
                      {server.server_ip || 'No IP configured'}
                      {server.node_group_count > 0 && (
                        <span className="ml-2 text-brand-600">
                          • {server.node_group_count} node group{server.node_group_count !== 1 ? 's' : ''}
                        </span>
                      )}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {server.stack_installed ? (
                    <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-100 text-green-700 text-sm rounded-full">
                      <CheckCircle className="w-4 h-4" />
                      Installed
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 px-2 py-1 bg-amd-gray-100 text-amd-gray-600 text-sm rounded-full">
                      <XCircle className="w-4 h-4" />
                      Not Installed
                    </span>
                  )}
                  <button
                    onClick={() => setExpandedServer(expandedServer === server.id ? null : server.id)}
                    className="p-2 hover:bg-amd-gray-100 rounded-lg"
                  >
                    {expandedServer === server.id ? (
                      <ChevronUp className="w-5 h-5" />
                    ) : (
                      <ChevronDown className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>

              {expandedServer === server.id && (
                <div className="mt-4 pt-4 border-t border-amd-gray-200">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div>
                      <p className="text-sm text-amd-gray-500">Prometheus</p>
                      <p className="font-medium">:{server.prometheus_port}</p>
                    </div>
                    <div>
                      <p className="text-sm text-amd-gray-500">Loki</p>
                      <p className="font-medium">:{server.loki_port}</p>
                    </div>
                    <div>
                      <p className="text-sm text-amd-gray-500">Grafana</p>
                      <p className="font-medium">:{server.grafana_port}</p>
                    </div>
                    <div>
                      <p className="text-sm text-amd-gray-500">Retention</p>
                      <p className="font-medium">{server.prometheus_retention_time}</p>
                    </div>
                  </div>

                  {/* SSH Credentials Section */}
                  {server.setup_monitoring_stack && (
                    <div className="bg-amd-gray-50 p-4 rounded-lg mb-4">
                      <h4 className="font-medium mb-3">SSH Configuration</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <p className="text-sm text-amd-gray-500">SSH User</p>
                          <p className="font-medium">{server.ssh_user || 'Not set'}</p>
                        </div>
                        <div>
                          <p className="text-sm text-amd-gray-500">SSH Port</p>
                          <p className="font-medium">{server.ssh_port}</p>
                        </div>
                        <div>
                          <p className="text-sm text-amd-gray-500">Auth Type</p>
                          <p className="font-medium capitalize">{server.ssh_auth_type}</p>
                        </div>
                        <div>
                          <p className="text-sm text-amd-gray-500">Credentials</p>
                          {server.ssh_auth_type === 'key' ? (
                            server.has_ssh_key ? (
                              <span className="inline-flex items-center gap-1 text-green-600 text-sm">
                                <CheckCircle className="w-4 h-4" />
                                Key uploaded
                              </span>
                            ) : (
                              <div className="flex items-center gap-2">
                                <span className="text-yellow-600 text-sm">No key</span>
                                <label className="btn btn-secondary text-xs py-1 px-2 cursor-pointer">
                                  {uploadingKey?.serverId === server.id && uploadingKey?.type === 'ssh' ? (
                                    <RefreshCw className="w-3 h-3 animate-spin" />
                                  ) : (
                                    'Upload'
                                  )}
                                  <input
                                    type="file"
                                    className="hidden"
                                    onChange={(e) => {
                                      const file = e.target.files?.[0]
                                      if (file) uploadSshKey(server.id, file, 'ssh')
                                    }}
                                  />
                                </label>
                              </div>
                            )
                          ) : (
                            server.has_ssh_password ? (
                              <span className="inline-flex items-center gap-1 text-green-600 text-sm">
                                <CheckCircle className="w-4 h-4" />
                                Password set
                              </span>
                            ) : (
                              <span className="text-yellow-600 text-sm">No password</span>
                            )
                          )}
                        </div>
                      </div>

                      {/* Jump Host Section */}
                      {server.use_jump_host && (
                        <div className="mt-4 pt-4 border-t border-amd-gray-200">
                          <h5 className="text-sm font-medium mb-2">Jump Host</h5>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div>
                              <p className="text-sm text-amd-gray-500">Host</p>
                              <p className="font-medium">{server.jump_host || 'Not set'}</p>
                            </div>
                            <div>
                              <p className="text-sm text-amd-gray-500">User</p>
                              <p className="font-medium">{server.jump_user || 'Not set'}</p>
                            </div>
                            <div>
                              <p className="text-sm text-amd-gray-500">Auth Type</p>
                              <p className="font-medium capitalize">{server.jump_auth_type}</p>
                            </div>
                            <div>
                              <p className="text-sm text-amd-gray-500">Credentials</p>
                              {server.jump_auth_type === 'key' ? (
                                server.has_jump_key ? (
                                  <span className="inline-flex items-center gap-1 text-green-600 text-sm">
                                    <CheckCircle className="w-4 h-4" />
                                    Key uploaded
                                  </span>
                                ) : (
                                  <div className="flex items-center gap-2">
                                    <span className="text-yellow-600 text-sm">No key</span>
                                    <label className="btn btn-secondary text-xs py-1 px-2 cursor-pointer">
                                      {uploadingKey?.serverId === server.id && uploadingKey?.type === 'jump' ? (
                                        <RefreshCw className="w-3 h-3 animate-spin" />
                                      ) : (
                                        'Upload'
                                      )}
                                      <input
                                        type="file"
                                        className="hidden"
                                        onChange={(e) => {
                                          const file = e.target.files?.[0]
                                          if (file) uploadSshKey(server.id, file, 'jump')
                                        }}
                                      />
                                    </label>
                                  </div>
                                )
                              ) : (
                                server.has_jump_password ? (
                                  <span className="inline-flex items-center gap-1 text-green-600 text-sm">
                                    <CheckCircle className="w-4 h-4" />
                                    Password set
                                  </span>
                                ) : (
                                  <span className="text-yellow-600 text-sm">No password</span>
                                )
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Actions - Order: Install Stack -> Sync Targets -> Check Services -> Delete */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {server.setup_monitoring_stack && (
                      <button
                        onClick={() => installStack(server.id)}
                        className={`btn flex items-center gap-2 ${
                          server.stack_installed
                            ? 'bg-green-600 text-white hover:bg-green-700'
                            : 'btn-primary'
                        }`}
                        disabled={installStatus[server.id]?.status === 'running'}
                      >
                        {server.stack_installed ? (
                          <CheckCircle className="w-4 h-4" />
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                        {server.stack_installed ? 'Stack Installed' : 'Install Stack'}
                      </button>
                    )}
                    {server.stack_installed && (
                      <button
                        onClick={() => syncTargets(server.id)}
                        disabled={syncingTargets[server.id]}
                        className={`btn flex items-center gap-2 ${
                          targetsSynced[server.id]
                            ? 'bg-green-600 text-white hover:bg-green-700'
                            : 'btn-secondary'
                        }`}
                      >
                        {syncingTargets[server.id] ? (
                          <RefreshCw className="w-4 h-4 animate-spin" />
                        ) : targetsSynced[server.id] ? (
                          <CheckCircle className="w-4 h-4" />
                        ) : (
                          <RefreshCw className="w-4 h-4" />
                        )}
                        {syncingTargets[server.id] ? 'Syncing...' : 'Sync Targets'}
                      </button>
                    )}
                    <button
                      onClick={() => checkServices(server.id)}
                      disabled={checkingServices[server.id]}
                      className={`btn flex items-center gap-2 ${
                        serviceStatus[server.id]?.ready_for_install ||
                        (serviceStatus[server.id]?.services?.prometheus?.running &&
                         serviceStatus[server.id]?.services?.grafana?.running)
                          ? 'bg-green-600 text-white hover:bg-green-700'
                          : 'btn-secondary'
                      }`}
                    >
                      {checkingServices[server.id] ? (
                        <RefreshCw className="w-4 h-4 animate-spin" />
                      ) : serviceStatus[server.id]?.ready_for_install ||
                         (serviceStatus[server.id]?.services?.prometheus?.running &&
                          serviceStatus[server.id]?.services?.grafana?.running) ? (
                        <CheckCircle className="w-4 h-4" />
                      ) : (
                        <RefreshCw className="w-4 h-4" />
                      )}
                      {checkingServices[server.id] ? 'Checking...' : 'Check Services'}
                    </button>
                    <button
                      onClick={() => {
                        if (confirm(`Delete monitoring server "${server.name}"?`)) {
                          deleteMutation.mutate(server.id)
                        }
                      }}
                      disabled={server.node_group_count > 0}
                      className="btn btn-danger flex items-center gap-2"
                      title={server.node_group_count > 0 ? 'Cannot delete - has associated node groups' : ''}
                    >
                      <Trash2 className="w-4 h-4" />
                      Delete
                    </button>
                  </div>

                  {/* Service Status */}
                  {serviceStatus[server.id] && (
                    <div className="bg-amd-gray-50 p-4 rounded-lg mb-4">
                      <h4 className="font-medium mb-2">Service Status</h4>
                      {serviceStatus[server.id].error ? (
                        <p className="text-red-600">{serviceStatus[server.id].error}</p>
                      ) : (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          {Object.entries(serviceStatus[server.id].services || {}).map(([name, status]: [string, any]) => (
                            <div key={name} className="flex items-center gap-2">
                              {status.running || status.installed || status.accessible ? (
                                <CheckCircle className="w-4 h-4 text-green-600" />
                              ) : (
                                <XCircle className="w-4 h-4 text-red-600" />
                              )}
                              <span className="text-sm capitalize">{name.replace('_', ' ')}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Install Status */}
                  {installStatus[server.id] && (
                    <div className="bg-amd-gray-50 p-4 rounded-lg">
                      <h4 className="font-medium mb-2">Installation Status</h4>
                      <div className="flex items-center gap-2 mb-2">
                        {installStatus[server.id].status === 'running' && (
                          <RefreshCw className="w-4 h-4 animate-spin text-brand-600" />
                        )}
                        {installStatus[server.id].status === 'completed' && (
                          <CheckCircle className="w-4 h-4 text-green-600" />
                        )}
                        {installStatus[server.id].status === 'failed' && (
                          <XCircle className="w-4 h-4 text-red-600" />
                        )}
                        <span className="font-medium capitalize">{installStatus[server.id].status}</span>
                        {installStatus[server.id].current_step && (
                          <span className="text-amd-gray-600">- {installStatus[server.id].current_step}</span>
                        )}
                      </div>
                      {installStatus[server.id].logs && installStatus[server.id].logs.length > 0 && (
                        <div className="bg-amd-gray-900 text-amd-gray-100 p-3 rounded text-sm font-mono max-h-40 overflow-y-auto">
                          {installStatus[server.id].logs.map((log: any, i: number) => (
                            <div key={i} className={log.level === 'error' ? 'text-red-400' : ''}>
                              [{new Date(log.timestamp).toLocaleTimeString()}] {log.message}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default MonitoringServers
