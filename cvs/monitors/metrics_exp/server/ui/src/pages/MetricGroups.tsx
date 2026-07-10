import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Plus,
  BarChart3,
  Edit2,
  Trash2,
  CheckCircle,
  XCircle,
  RefreshCw,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import api from '../api'

interface MetricGroup {
  id: number
  name: string
  description?: string
  gpu_utilization: boolean
  gpu_memory: boolean
  gpu_temperature: boolean
  gpu_power: boolean
  gpu_fan: boolean
  gpu_clocks: boolean
  gpu_pcie: boolean
  gpu_ecc: boolean
  node_cpu: boolean
  node_memory: boolean
  node_disk: boolean
  node_network: boolean
  collect_logs: boolean
  log_patterns: string[]
  node_group_count: number
}

function MetricGroups() {
  const queryClient = useQueryClient()
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [expandedGroup, setExpandedGroup] = useState<number | null>(null)
  const [editingGroup, setEditingGroup] = useState<number | null>(null)
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    gpu_utilization: true,
    gpu_memory: true,
    gpu_temperature: true,
    gpu_power: true,
    gpu_fan: false,
    gpu_clocks: false,
    gpu_pcie: false,
    gpu_ecc: false,
    node_cpu: true,
    node_memory: true,
    node_disk: true,
    node_network: false,
    collect_logs: true,
    log_patterns: [] as string[],
  })

  const { data: groups = [], isLoading } = useQuery({
    queryKey: ['metric-groups'],
    queryFn: () => api.get<MetricGroup[]>('/metric-groups').then(r => r.data),
  })

  const createMutation = useMutation({
    mutationFn: (data: typeof formData) => api.post('/metric-groups', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['metric-groups'] })
      setShowCreateForm(false)
      resetForm()
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: Partial<typeof formData> }) =>
      api.put(`/metric-groups/${id}`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['metric-groups'] })
      setEditingGroup(null)
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) => api.delete(`/metric-groups/${id}?force=true`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['metric-groups'] })
      queryClient.invalidateQueries({ queryKey: ['nodegroups'] })
    },
  })

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      gpu_utilization: true,
      gpu_memory: true,
      gpu_temperature: true,
      gpu_power: true,
      gpu_fan: false,
      gpu_clocks: false,
      gpu_pcie: false,
      gpu_ecc: false,
      node_cpu: true,
      node_memory: true,
      node_disk: true,
      node_network: false,
      collect_logs: true,
      log_patterns: [],
    })
  }

  const startEditing = (group: MetricGroup) => {
    setFormData({
      name: group.name,
      description: group.description || '',
      gpu_utilization: group.gpu_utilization,
      gpu_memory: group.gpu_memory,
      gpu_temperature: group.gpu_temperature,
      gpu_power: group.gpu_power,
      gpu_fan: group.gpu_fan,
      gpu_clocks: group.gpu_clocks,
      gpu_pcie: group.gpu_pcie,
      gpu_ecc: group.gpu_ecc,
      node_cpu: group.node_cpu,
      node_memory: group.node_memory,
      node_disk: group.node_disk,
      node_network: group.node_network,
      collect_logs: group.collect_logs,
      log_patterns: group.log_patterns || [],
    })
    setEditingGroup(group.id)
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

  const MetricCheckbox = ({ label, checked, onChange, description }: {
    label: string
    checked: boolean
    onChange: (checked: boolean) => void
    description?: string
  }) => (
    <label className="flex items-start gap-3 p-3 bg-white rounded-lg border border-amd-gray-200 hover:border-brand-300 cursor-pointer transition-colors">
      <input
        type="checkbox"
        checked={checked}
        onChange={e => onChange(e.target.checked)}
        className="mt-0.5 rounded border-amd-gray-300 text-brand-600 focus:ring-brand-500"
      />
      <div>
        <span className="font-medium text-amd-gray-900">{label}</span>
        {description && <p className="text-sm text-amd-gray-500">{description}</p>}
      </div>
    </label>
  )

  const renderForm = (isEdit: boolean = false) => (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="label">Name *</label>
          <input
            type="text"
            className="input"
            value={formData.name}
            onChange={e => setFormData(prev => ({ ...prev, name: e.target.value }))}
            placeholder="gpu-full-metrics"
            disabled={isEdit && formData.name === 'default'}
          />
        </div>
        <div>
          <label className="label">Description</label>
          <input
            type="text"
            className="input"
            value={formData.description}
            onChange={e => setFormData(prev => ({ ...prev, description: e.target.value }))}
            placeholder="Full GPU metrics collection"
          />
        </div>
      </div>

      <div>
        <h3 className="font-semibold text-amd-gray-900 mb-3">GPU Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCheckbox
            label="Utilization"
            checked={formData.gpu_utilization}
            onChange={checked => setFormData(prev => ({ ...prev, gpu_utilization: checked }))}
            description="GPU usage %"
          />
          <MetricCheckbox
            label="Memory"
            checked={formData.gpu_memory}
            onChange={checked => setFormData(prev => ({ ...prev, gpu_memory: checked }))}
            description="VRAM usage"
          />
          <MetricCheckbox
            label="Temperature"
            checked={formData.gpu_temperature}
            onChange={checked => setFormData(prev => ({ ...prev, gpu_temperature: checked }))}
            description="Junction temp"
          />
          <MetricCheckbox
            label="Power"
            checked={formData.gpu_power}
            onChange={checked => setFormData(prev => ({ ...prev, gpu_power: checked }))}
            description="Power draw"
          />
          <MetricCheckbox
            label="Fan Speed"
            checked={formData.gpu_fan}
            onChange={checked => setFormData(prev => ({ ...prev, gpu_fan: checked }))}
            description="RPM & %"
          />
          <MetricCheckbox
            label="Clocks"
            checked={formData.gpu_clocks}
            onChange={checked => setFormData(prev => ({ ...prev, gpu_clocks: checked }))}
            description="SCLK & MCLK"
          />
          <MetricCheckbox
            label="PCIe"
            checked={formData.gpu_pcie}
            onChange={checked => setFormData(prev => ({ ...prev, gpu_pcie: checked }))}
            description="Bandwidth"
          />
          <MetricCheckbox
            label="ECC Errors"
            checked={formData.gpu_ecc}
            onChange={checked => setFormData(prev => ({ ...prev, gpu_ecc: checked }))}
            description="Error counts"
          />
        </div>
      </div>

      <div>
        <h3 className="font-semibold text-amd-gray-900 mb-3">Node Metrics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCheckbox
            label="CPU"
            checked={formData.node_cpu}
            onChange={checked => setFormData(prev => ({ ...prev, node_cpu: checked }))}
            description="CPU usage"
          />
          <MetricCheckbox
            label="Memory"
            checked={formData.node_memory}
            onChange={checked => setFormData(prev => ({ ...prev, node_memory: checked }))}
            description="RAM usage"
          />
          <MetricCheckbox
            label="Disk"
            checked={formData.node_disk}
            onChange={checked => setFormData(prev => ({ ...prev, node_disk: checked }))}
            description="I/O & space"
          />
          <MetricCheckbox
            label="Network"
            checked={formData.node_network}
            onChange={checked => setFormData(prev => ({ ...prev, node_network: checked }))}
            description="Traffic"
          />
        </div>
      </div>

      <div>
        <h3 className="font-semibold text-amd-gray-900 mb-3">Log Collection</h3>
        <MetricCheckbox
          label="Collect System Logs"
          checked={formData.collect_logs}
          onChange={checked => setFormData(prev => ({ ...prev, collect_logs: checked }))}
          description="Collect dmesg, journalctl, and syslog entries"
        />
      </div>

      <div className="flex justify-end gap-3">
        <button
          onClick={() => {
            if (isEdit) {
              setEditingGroup(null)
            } else {
              setShowCreateForm(false)
            }
            resetForm()
          }}
          className="btn btn-secondary"
        >
          Cancel
        </button>
        <button
          onClick={() => {
            if (isEdit && editingGroup) {
              updateMutation.mutate({ id: editingGroup, data: formData })
            } else {
              createMutation.mutate(formData)
            }
          }}
          disabled={!formData.name || createMutation.isPending || updateMutation.isPending}
          className="btn btn-primary"
        >
          {isEdit ? (updateMutation.isPending ? 'Saving...' : 'Save Changes') : (createMutation.isPending ? 'Creating...' : 'Create Group')}
        </button>
      </div>
    </div>
  )

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-amd-gray-900">Metric Groups</h1>
          <p className="text-amd-gray-600 mt-1">
            Configure which metrics to collect and display for your GPU node groups
          </p>
        </div>
        <button
          onClick={() => setShowCreateForm(true)}
          className="btn btn-primary flex items-center gap-2"
        >
          <Plus className="w-5 h-5" />
          Create Group
        </button>
      </div>

      {/* Create Form */}
      {showCreateForm && (
        <div className="card mb-6">
          <h2 className="text-xl font-bold mb-4">Create Metric Group</h2>
          {renderForm(false)}
        </div>
      )}

      {/* Group List */}
      {groups.length === 0 ? (
        <div className="card text-center py-12">
          <BarChart3 className="w-16 h-16 mx-auto text-amd-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-amd-gray-900 mb-2">No Metric Groups</h3>
          <p className="text-amd-gray-600 mb-4">
            Create a metric group to define which metrics to collect.
          </p>
          <button
            onClick={() => setShowCreateForm(true)}
            className="btn btn-primary inline-flex items-center gap-2"
          >
            <Plus className="w-5 h-5" />
            Create Your First Group
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {groups.map(group => (
            <div key={group.id} className="card">
              {editingGroup === group.id ? (
                <div>
                  <h2 className="text-xl font-bold mb-4">Edit Metric Group</h2>
                  {renderForm(true)}
                </div>
              ) : (
                <>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 bg-brand-100 rounded-lg flex items-center justify-center">
                        <BarChart3 className="w-6 h-6 text-brand-600" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-amd-gray-900">
                          {group.name}
                          {group.name === 'default' && (
                            <span className="ml-2 text-xs bg-amd-gray-200 text-amd-gray-600 px-2 py-0.5 rounded">
                              Default
                            </span>
                          )}
                        </h3>
                        <p className="text-sm text-amd-gray-600">
                          {group.description || 'No description'}
                          {group.node_group_count > 0 && (
                            <span className="ml-2 text-brand-600">
                              • {group.node_group_count} node group{group.node_group_count !== 1 ? 's' : ''}
                            </span>
                          )}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => startEditing(group)}
                        className="p-2 hover:bg-amd-gray-100 rounded-lg text-amd-gray-600"
                      >
                        <Edit2 className="w-5 h-5" />
                      </button>
                      <button
                        onClick={() => {
                          const message = group.node_group_count > 0
                            ? `Delete metric group "${group.name}"? This will unassign it from ${group.node_group_count} node group(s).`
                            : `Delete metric group "${group.name}"?`
                          if (confirm(message)) {
                            deleteMutation.mutate(group.id)
                          }
                        }}
                        disabled={group.name === 'default'}
                        className="p-2 hover:bg-red-100 rounded-lg text-amd-gray-600 hover:text-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
                        title={group.name === 'default' ? 'Cannot delete default group' : ''}
                      >
                        <Trash2 className="w-5 h-5" />
                      </button>
                      <button
                        onClick={() => setExpandedGroup(expandedGroup === group.id ? null : group.id)}
                        className="p-2 hover:bg-amd-gray-100 rounded-lg"
                      >
                        {expandedGroup === group.id ? (
                          <ChevronUp className="w-5 h-5" />
                        ) : (
                          <ChevronDown className="w-5 h-5" />
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Quick preview of enabled metrics */}
                  <div className="mt-4 flex flex-wrap gap-2">
                    {group.gpu_utilization && <MetricBadge label="GPU Util" enabled />}
                    {group.gpu_memory && <MetricBadge label="GPU Mem" enabled />}
                    {group.gpu_temperature && <MetricBadge label="Temp" enabled />}
                    {group.gpu_power && <MetricBadge label="Power" enabled />}
                    {group.gpu_fan && <MetricBadge label="Fan" enabled />}
                    {group.gpu_clocks && <MetricBadge label="Clocks" enabled />}
                    {group.gpu_pcie && <MetricBadge label="PCIe" enabled />}
                    {group.gpu_ecc && <MetricBadge label="ECC" enabled />}
                    {group.node_cpu && <MetricBadge label="CPU" enabled />}
                    {group.node_memory && <MetricBadge label="RAM" enabled />}
                    {group.node_disk && <MetricBadge label="Disk" enabled />}
                    {group.node_network && <MetricBadge label="Network" enabled />}
                  </div>

                  {expandedGroup === group.id && (
                    <div className="mt-4 pt-4 border-t border-amd-gray-200">
                      <div className="grid grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-medium text-amd-gray-900 mb-2">GPU Metrics</h4>
                          <div className="space-y-1 text-sm">
                            <MetricRow label="Utilization" enabled={group.gpu_utilization} />
                            <MetricRow label="Memory" enabled={group.gpu_memory} />
                            <MetricRow label="Temperature" enabled={group.gpu_temperature} />
                            <MetricRow label="Power" enabled={group.gpu_power} />
                            <MetricRow label="Fan Speed" enabled={group.gpu_fan} />
                            <MetricRow label="Clocks" enabled={group.gpu_clocks} />
                            <MetricRow label="PCIe" enabled={group.gpu_pcie} />
                            <MetricRow label="ECC Errors" enabled={group.gpu_ecc} />
                          </div>
                        </div>
                        <div>
                          <h4 className="font-medium text-amd-gray-900 mb-2">Node Metrics</h4>
                          <div className="space-y-1 text-sm">
                            <MetricRow label="CPU" enabled={group.node_cpu} />
                            <MetricRow label="Memory" enabled={group.node_memory} />
                            <MetricRow label="Disk" enabled={group.node_disk} />
                            <MetricRow label="Network" enabled={group.node_network} />
                          </div>
                          <h4 className="font-medium text-amd-gray-900 mt-4 mb-2">Log Collection</h4>
                          <MetricRow label="System Logs" enabled={group.collect_logs} />
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function MetricBadge({ label, enabled }: { label: string; enabled: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-1 text-xs rounded-full ${
      enabled ? 'bg-green-100 text-green-700' : 'bg-amd-gray-100 text-amd-gray-500'
    }`}>
      {enabled ? <CheckCircle className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
      {label}
    </span>
  )
}

function MetricRow({ label, enabled }: { label: string; enabled: boolean }) {
  return (
    <div className="flex items-center gap-2">
      {enabled ? (
        <CheckCircle className="w-4 h-4 text-green-600" />
      ) : (
        <XCircle className="w-4 h-4 text-amd-gray-400" />
      )}
      <span className={enabled ? 'text-amd-gray-900' : 'text-amd-gray-400'}>{label}</span>
    </div>
  )
}

export default MetricGroups
