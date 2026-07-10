import { useState } from 'react'
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { createControlNodeGroup, fetchMonitoringServers } from '../api'
import { Cpu, ArrowLeft, Plus, X, Monitor, FileKey } from 'lucide-react'

type ControlType = 'slurm' | 'kubernetes'
type AuthType = 'key' | 'password'
type KubeconfigSource = 'auto' | 'path' | 'upload'

function AddControlNodeGroup() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    control_type: 'slurm' as ControlType,
    custom_exporter_port: 0,
    ssh_user: 'root',
    ssh_port: 22,
    ssh_auth_type: 'key' as AuthType,
    ssh_password: '',
    use_jump_host: false,
    jump_host: '',
    jump_port: 22,
    jump_user: '',
    jump_auth_type: 'key' as AuthType,
    jump_password: '',
    remote_auth_type: 'key' as AuthType,
    remote_key_path: '',
    // Kubeconfig (Kubernetes only)
    kubeconfig_source: 'auto' as KubeconfigSource,
    kubeconfig_remote_path: '',
    remote_password: '',
    monitoring_server_id: null as number | null,
  })

  const { data: monitoringServers = [] } = useQuery({
    queryKey: ['monitoring-servers'],
    queryFn: fetchMonitoringServers,
  })

  const [ipAddresses, setIpAddresses] = useState<string[]>([])
  const [ipInput, setIpInput] = useState('')
  const [bulkInput, setBulkInput] = useState('')
  const [showBulkInput, setShowBulkInput] = useState(false)
  const [error, setError] = useState<string | null>(null)

  type ApiError = Error & { response?: { data?: { detail?: string } } }

  const createMutation = useMutation({
    mutationFn: createControlNodeGroup,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['control-nodegroups'] })
      navigate(`/control-nodegroups/${data.id}`)
    },
    onError: (err: ApiError) => {
      setError(
        err.response?.data?.detail ||
          err.message ||
          'Failed to create control node group'
      )
    },
  })

  const handleAddIP = () => {
    const ip = ipInput.trim()
    if (ip && !ipAddresses.includes(ip)) {
      setIpAddresses([...ipAddresses, ip])
      setIpInput('')
    }
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

  const set = (field: string, value: unknown) =>
    setFormData((prev) => ({ ...prev, [field]: value }))

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    const payload: Parameters<typeof createControlNodeGroup>[0] = {
      name: formData.name,
      control_type: formData.control_type,
      ssh_user: formData.ssh_user,
      ssh_port: formData.ssh_port,
    }

    if (formData.description) payload.description = formData.description
    if (formData.custom_exporter_port > 0)
      payload.custom_exporter_port = formData.custom_exporter_port
    if (formData.monitoring_server_id)
      payload.monitoring_server_id = formData.monitoring_server_id

    payload.ssh_auth_type = formData.ssh_auth_type
    if (formData.ssh_auth_type === 'password' && formData.ssh_password)
      payload.ssh_password = formData.ssh_password

    payload.use_jump_host = formData.use_jump_host
    if (formData.use_jump_host) {
      if (formData.jump_host) payload.jump_host = formData.jump_host
      payload.jump_port = formData.jump_port
      if (formData.jump_user) payload.jump_user = formData.jump_user
      payload.jump_auth_type = formData.jump_auth_type
      if (formData.jump_auth_type === 'password' && formData.jump_password)
        payload.jump_password = formData.jump_password

      payload.remote_auth_type = formData.remote_auth_type
      if (formData.remote_auth_type === 'key' && formData.remote_key_path)
        payload.remote_key_path = formData.remote_key_path
      if (formData.remote_auth_type === 'password' && formData.remote_password)
        payload.remote_password = formData.remote_password
    }

    // Kubeconfig (Kubernetes only)
    if (formData.control_type === 'kubernetes') {
      payload.kubeconfig_source = formData.kubeconfig_source
      if (formData.kubeconfig_source === 'path' && formData.kubeconfig_remote_path)
        payload.kubeconfig_remote_path = formData.kubeconfig_remote_path
      // "upload" source: file is uploaded separately after group creation
    }

    // Auto-include any IP that's been typed but not yet added via + or Enter
    const finalIPs = [...ipAddresses]
    const pendingIP = ipInput.trim()
    if (pendingIP && !finalIPs.includes(pendingIP)) finalIPs.push(pendingIP)
    if (finalIPs.length > 0) payload.ip_addresses = finalIPs

    createMutation.mutate(payload)
  }

  const defaultPort = formData.control_type === 'slurm' ? 9418 : 9419
  const typeLabel =
    formData.control_type === 'slurm' ? 'Slurm Head Node' : 'K8s Control Plane'

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <button
        onClick={() => navigate('/control-nodegroups')}
        className="flex items-center gap-2 text-amd-gray-500 hover:text-amd-gray-700 mb-6 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Control Node Groups
      </button>

      <div className="card">
        <div className="flex items-center gap-4 mb-8">
          <div className="p-3 bg-brand-100 rounded-lg">
            <Cpu className="w-6 h-6 text-brand-600" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-amd-gray-900">
              Create Control Node Group
            </h1>
            <p className="text-amd-gray-500">
              Monitor Slurm or Kubernetes control plane nodes
            </p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* ---- Control Plane Type ---- */}
          <div className="border border-amd-gray-200 rounded-lg p-4">
            <label className="label mb-3 block font-semibold">
              Control Plane Type *
            </label>
            <div className="grid grid-cols-2 gap-4">
              {(['slurm', 'kubernetes'] as const).map((type) => (
                <label
                  key={type}
                  className={`flex items-start gap-3 p-4 border-2 rounded-lg cursor-pointer transition-colors ${
                    formData.control_type === type
                      ? 'border-brand-600 bg-brand-50'
                      : 'border-amd-gray-200 hover:border-amd-gray-400'
                  }`}
                >
                  <input
                    type="radio"
                    name="control_type"
                    value={type}
                    checked={formData.control_type === type}
                    onChange={() => set('control_type', type)}
                    className="w-4 h-4 text-brand-600 mt-0.5"
                  />
                  <div>
                    <p className="font-medium">
                      {type === 'slurm' ? 'Slurm' : 'Kubernetes'}
                    </p>
                    <p className="text-xs text-amd-gray-500 mt-1">
                      {type === 'slurm'
                        ? `slurm_exporter on port ${9418}`
                        : `k8s_cp_exporter on port ${9419}`}
                    </p>
                    <p className="text-xs text-amd-gray-400 mt-0.5">
                      {type === 'slurm'
                        ? 'Collects job queue, node states, scheduler stats'
                        : 'Collects apiserver, etcd, scheduler, controller-manager metrics'}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* ---- Basic Info ---- */}
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="label">Group Name *</label>
              <input
                type="text"
                required
                className="input"
                placeholder={
                  formData.control_type === 'slurm'
                    ? 'e.g., hpc-cluster-head'
                    : 'e.g., k8s-prod-control-plane'
                }
                value={formData.name}
                onChange={(e) => set('name', e.target.value)}
              />
            </div>
            <div>
              <label className="label">Description</label>
              <input
                type="text"
                className="input"
                placeholder="Optional description"
                value={formData.description}
                onChange={(e) => set('description', e.target.value)}
              />
            </div>
          </div>

          {/* ---- Monitoring Server ---- */}
          <div>
            <label className="label flex items-center gap-2">
              <Monitor className="w-4 h-4 text-blue-600" />
              Monitoring Server
            </label>
            <select
              className="input"
              value={formData.monitoring_server_id || ''}
              onChange={(e) =>
                set(
                  'monitoring_server_id',
                  e.target.value ? parseInt(e.target.value) : null
                )
              }
            >
              <option value="">Select monitoring server (optional)…</option>
              {monitoringServers.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.server_ip || 'IP not set'})
                </option>
              ))}
            </select>
            <p className="text-xs text-amd-gray-400 mt-1">
              Required to enable Grafana dashboards and Prometheus scraping.
            </p>
          </div>

          {/* ---- Custom Exporter Port ---- */}
          <div>
            <label className="label">
              Custom Exporter Port
              <span className="ml-2 text-xs text-amd-gray-400 font-normal">
                (0 = default: {defaultPort} for {typeLabel})
              </span>
            </label>
            <input
              type="number"
              className="input"
              min={0}
              max={65535}
              value={formData.custom_exporter_port}
              onChange={(e) =>
                set('custom_exporter_port', parseInt(e.target.value) || 0)
              }
            />
          </div>

          {/* ---- SSH Config ---- */}
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="label">SSH Username</label>
              <input
                type="text"
                className="input"
                value={formData.ssh_user}
                onChange={(e) => set('ssh_user', e.target.value)}
              />
            </div>
            <div>
              <label className="label">SSH Port</label>
              <input
                type="number"
                className="input"
                value={formData.ssh_port}
                onChange={(e) => set('ssh_port', parseInt(e.target.value) || 22)}
              />
            </div>
          </div>

          {/* ---- Direct SSH Auth (only when no jump host) ---- */}
          {!formData.use_jump_host && (
            <div className="border border-amd-gray-200 rounded-lg p-4">
              <h4 className="font-medium text-amd-gray-700 mb-3">
                Direct SSH Authentication
              </h4>
              <div className="flex items-center gap-6 mb-3">
                {(['key', 'password'] as const).map((type) => (
                  <label
                    key={type}
                    className="flex items-center gap-2 cursor-pointer"
                  >
                    <input
                      type="radio"
                      name="ssh_auth_type"
                      value={type}
                      checked={formData.ssh_auth_type === type}
                      onChange={() => set('ssh_auth_type', type)}
                      className="w-4 h-4 text-brand-600"
                    />
                    <span>{type === 'key' ? 'SSH Key' : 'Password'}</span>
                  </label>
                ))}
              </div>
              {formData.ssh_auth_type === 'password' ? (
                <input
                  type="password"
                  className="input"
                  placeholder="SSH password"
                  value={formData.ssh_password}
                  onChange={(e) => set('ssh_password', e.target.value)}
                />
              ) : (
                <p className="text-sm text-amd-gray-400">
                  You will upload the SSH private key after creating the group.
                </p>
              )}
            </div>
          )}

          {/* ---- Jump Host Toggle ---- */}
          <div className="border border-amd-gray-200 rounded-lg p-4">
            <div className="flex items-center gap-3 mb-4">
              <input
                type="checkbox"
                id="use_jump_host"
                checked={formData.use_jump_host}
                onChange={(e) => set('use_jump_host', e.target.checked)}
                className="w-4 h-4 text-brand-600 rounded"
              />
              <label htmlFor="use_jump_host" className="font-medium cursor-pointer">
                Use Jump Host / Bastion
              </label>
            </div>

            {formData.use_jump_host && (
              <div className="space-y-4 pl-7">
                <div className="p-3 bg-blue-50 rounded-lg text-sm text-blue-800">
                  <strong>Connection path:</strong> Fleet Manager → Jump Host → Control
                  Node(s)
                </div>

                {/* Jump host address/port */}
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label className="label">Jump Host Address *</label>
                    <input
                      type="text"
                      className="input"
                      placeholder="bastion.example.com or 10.0.0.100"
                      value={formData.jump_host}
                      onChange={(e) => set('jump_host', e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="label">Jump Host Port</label>
                    <input
                      type="number"
                      className="input"
                      value={formData.jump_port}
                      onChange={(e) =>
                        set('jump_port', parseInt(e.target.value) || 22)
                      }
                    />
                  </div>
                </div>

                <div>
                  <label className="label">Jump Host Username</label>
                  <input
                    type="text"
                    className="input"
                    placeholder="root"
                    value={formData.jump_user}
                    onChange={(e) => set('jump_user', e.target.value)}
                  />
                </div>

                <div>
                  <label className="label">Jump Host Auth Method</label>
                  <div className="flex items-center gap-6 mb-3">
                    {(['key', 'password'] as const).map((type) => (
                      <label
                        key={type}
                        className="flex items-center gap-2 cursor-pointer"
                      >
                        <input
                          type="radio"
                          name="jump_auth_type"
                          value={type}
                          checked={formData.jump_auth_type === type}
                          onChange={() => {
                            set('jump_auth_type', type)
                            set('jump_password', '')
                          }}
                          className="w-4 h-4 text-brand-600"
                        />
                        <span>{type === 'key' ? 'SSH Key' : 'Password'}</span>
                      </label>
                    ))}
                  </div>
                  {formData.jump_auth_type === 'password' ? (
                    <input
                      type="password"
                      className="input"
                      placeholder="Jump host password"
                      value={formData.jump_password}
                      onChange={(e) => set('jump_password', e.target.value)}
                    />
                  ) : (
                    <p className="text-sm text-amd-gray-400">
                      You will upload the jump host key after creating the group.
                    </p>
                  )}
                </div>

                {/* Remote node auth from jump host */}
                <div className="border-t border-amd-gray-200 pt-4">
                  <h4 className="font-medium text-amd-gray-700 mb-3">
                    Control Node Auth (from Jump Host)
                  </h4>
                  <div className="flex items-center gap-6 mb-3">
                    {(['key', 'password'] as const).map((type) => (
                      <label
                        key={type}
                        className="flex items-center gap-2 cursor-pointer"
                      >
                        <input
                          type="radio"
                          name="remote_auth_type"
                          value={type}
                          checked={formData.remote_auth_type === type}
                          onChange={() => {
                            set('remote_auth_type', type)
                            set('remote_key_path', '')
                            set('remote_password', '')
                          }}
                          className="w-4 h-4 text-brand-600"
                        />
                        <span>
                          {type === 'key' ? 'SSH Key (on jump host)' : 'Password'}
                        </span>
                      </label>
                    ))}
                  </div>
                  {formData.remote_auth_type === 'key' ? (
                    <input
                      type="text"
                      className="input font-mono text-sm"
                      placeholder="~/.ssh/control_node_key (path on jump host)"
                      value={formData.remote_key_path}
                      onChange={(e) => set('remote_key_path', e.target.value)}
                    />
                  ) : (
                    <input
                      type="password"
                      className="input"
                      placeholder="Control node password"
                      value={formData.remote_password}
                      onChange={(e) => set('remote_password', e.target.value)}
                    />
                  )}
                </div>
              </div>
            )}
          </div>

          {/* ---- Kubeconfig (Kubernetes only) ---- */}
          {formData.control_type === 'kubernetes' && (
            <div className="border border-blue-200 rounded-lg p-4 bg-blue-50">
              <div className="flex items-center gap-2 mb-3">
                <FileKey className="w-4 h-4 text-blue-600" />
                <h4 className="font-medium text-blue-900">Kubeconfig (Kubernetes)</h4>
              </div>
              <p className="text-xs text-blue-700 mb-3">
                The exporter needs access to a kubeconfig to call <code className="bg-blue-100 px-1 rounded">kubectl</code> and scrape component metrics.
              </p>
              <div className="space-y-3">
                {(['auto', 'path', 'upload'] as const).map((src) => (
                  <label key={src} className={`flex items-start gap-3 p-3 border-2 rounded-lg cursor-pointer transition-colors ${
                    formData.kubeconfig_source === src
                      ? 'border-blue-500 bg-white'
                      : 'border-blue-200 bg-white hover:border-blue-400'
                  }`}>
                    <input
                      type="radio"
                      name="kubeconfig_source"
                      value={src}
                      checked={formData.kubeconfig_source === src}
                      onChange={() => set('kubeconfig_source', src)}
                      className="w-4 h-4 text-blue-600 mt-0.5"
                    />
                    <div>
                      {src === 'auto' && (
                        <>
                          <p className="font-medium text-sm">Auto-detect (recommended)</p>
                          <p className="text-xs text-amd-gray-500 mt-0.5">
                            Exporter looks for <code>/etc/kubernetes/admin.conf</code> then <code>~/.kube/config</code> on the K8s node. Works on standard kubeadm clusters.
                          </p>
                        </>
                      )}
                      {src === 'path' && (
                        <>
                          <p className="font-medium text-sm">Path on the K8s node</p>
                          <p className="text-xs text-amd-gray-500 mt-0.5">
                            Specify the full path of an existing kubeconfig file on the control plane node.
                          </p>
                          {formData.kubeconfig_source === 'path' && (
                            <input
                              type="text"
                              className="input mt-2 font-mono text-sm"
                              placeholder="/etc/kubernetes/admin.conf"
                              value={formData.kubeconfig_remote_path}
                              onChange={(e) => set('kubeconfig_remote_path', e.target.value)}
                            />
                          )}
                        </>
                      )}
                      {src === 'upload' && (
                        <>
                          <p className="font-medium text-sm">Upload kubeconfig file</p>
                          <p className="text-xs text-amd-gray-500 mt-0.5">
                            Upload a kubeconfig file from your machine. It will be stored securely and deployed to the K8s node when you click Install.
                          </p>
                          {formData.kubeconfig_source === 'upload' && (
                            <p className="text-xs text-blue-600 mt-1 font-medium">
                              ↳ Upload the file on the Control Node Group detail page after creation.
                            </p>
                          )}
                        </>
                      )}
                    </div>
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* ---- IP Addresses ---- */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="label mb-0">Control Node IP Addresses</label>
              <button
                type="button"
                onClick={() => setShowBulkInput(!showBulkInput)}
                className="text-sm text-brand-600 hover:underline"
              >
                {showBulkInput ? 'Single Add' : 'Bulk Add'}
              </button>
            </div>

            {showBulkInput ? (
              <div className="space-y-2">
                <textarea
                  className="input h-32 font-mono text-sm"
                  placeholder="One IP per line, or comma/semicolon separated&#10;10.0.0.1&#10;10.0.0.2"
                  value={bulkInput}
                  onChange={(e) => setBulkInput(e.target.value)}
                />
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={handleBulkAdd}
                    className="btn btn-primary text-sm"
                  >
                    Add IPs
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowBulkInput(false)}
                    className="btn btn-secondary text-sm"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex gap-2">
                <input
                  type="text"
                  className="input flex-1 font-mono text-sm"
                  placeholder="Enter IP address (e.g., 10.0.0.1)"
                  value={ipInput}
                  onChange={(e) => setIpInput(e.target.value)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      handleAddIP()
                    }
                  }}
                />
                <button
                  type="button"
                  onClick={handleAddIP}
                  className="btn btn-secondary px-3"
                >
                  <Plus className="w-5 h-5" />
                </button>
              </div>
            )}

            {ipAddresses.length > 0 && (
              <div className="mt-3 p-3 bg-amd-gray-50 rounded-lg">
                <p className="text-sm text-amd-gray-500 mb-2">
                  {ipAddresses.length} IP{ipAddresses.length !== 1 ? 's' : ''} added:
                </p>
                <div className="flex flex-wrap gap-2">
                  {ipAddresses.map((ip) => (
                    <span
                      key={ip}
                      className="inline-flex items-center gap-1 px-3 py-1 bg-white border border-amd-gray-200 rounded-full text-sm font-mono"
                    >
                      {ip}
                      <button
                        type="button"
                        onClick={() =>
                          setIpAddresses(ipAddresses.filter((i) => i !== ip))
                        }
                        className="text-amd-gray-400 hover:text-red-600 transition-colors"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="font-medium text-red-700">Error</p>
              <p className="text-sm text-red-600 mt-1">{error}</p>
            </div>
          )}

          {/* Submit */}
          <div className="flex justify-end gap-3 pt-4 border-t border-amd-gray-200">
            <button
              type="button"
              onClick={() => navigate('/control-nodegroups')}
              className="btn btn-secondary"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!formData.name || createMutation.isPending}
              className="btn btn-primary"
            >
              {createMutation.isPending
                ? 'Creating…'
                : 'Create Control Node Group'}
            </button>
          </div>
        </form>
      </div>

      {/* Next steps info */}
      <div className="card mt-6 bg-blue-50 border border-blue-200">
        <h3 className="font-semibold text-blue-900 mb-2">Next Steps</h3>
        <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
          <li>Create the group with control node IP addresses</li>
          <li>Upload an SSH key for authentication</li>
          {formData.control_type === 'kubernetes' && formData.kubeconfig_source === 'upload' && (
            <li>Upload the kubeconfig file using the <strong>Upload Kubeconfig</strong> button on the detail page</li>
          )}
          <li>Click "Verify Connectivity" to test SSH access</li>
          <li>
            Click "Install" to deploy{' '}
            <code className="bg-blue-100 px-1 rounded">node_exporter</code>,{' '}
            <code className="bg-blue-100 px-1 rounded">
              {formData.control_type === 'slurm'
                ? 'slurm_exporter'
                : 'k8s_cp_exporter'}
            </code>
            , and <code className="bg-blue-100 px-1 rounded">promtail</code>
          </li>
          <li>View metrics in the auto-generated Grafana dashboard</li>
        </ol>
      </div>
    </div>
  )
}

export default AddControlNodeGroup
