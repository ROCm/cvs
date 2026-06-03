import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Types
export interface NodeGroup {
  id: number
  name: string
  description?: string
  ssh_user: string
  ssh_port: number
  ssh_auth_type: 'key' | 'password'
  ssh_key_path?: string
  metric_config_id?: number
  monitoring_server_id?: number
  metric_group_id?: number
  node_count: number
  active_nodes: number
  created_at: string
  updated_at: string
  // Jump host config
  use_jump_host: boolean
  jump_host?: string
  jump_port: number
  jump_user?: string
  jump_auth_type: 'key' | 'password'
  jump_key_path?: string
  remote_auth_type: 'key' | 'password'
  remote_key_path?: string
  // Password indicators (not the actual passwords)
  has_ssh_password: boolean
  has_jump_password: boolean
  has_remote_password: boolean
}

export interface Node {
  id: number
  node_group_id: number
  ip_address: string
  hostname?: string
  status: 'pending' | 'connected' | 'installing' | 'active' | 'error' | 'unreachable'
  status_message?: string
  last_seen?: string
  gpu_count?: number
  gpu_model?: string
  gpu_exporter_port: number
  node_exporter_port: number
  created_at: string
  updated_at: string
}

export interface NodeGroupDetail extends NodeGroup {
  nodes: Node[]
  metric_config?: MetricConfig
  monitoring_server_id?: number
  metric_group_id?: number
}

export interface MonitoringServer {
  id: number
  name: string
  description?: string
  server_ip?: string
  server_hostname?: string
  prometheus_port: number
  loki_port: number
  grafana_port: number
  grafana_admin_user: string
  stack_installed: boolean
  node_group_count: number
}

export interface MetricGroup {
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
}

export interface MetricConfig {
  id: number
  name: string
  fleet_health: boolean
  thermal_power: boolean
  pcie: boolean
  xgmi: boolean
  ecc: boolean
  ras: boolean
  utilization: boolean
  memory: boolean
  cpu_metrics: boolean
  memory_metrics: boolean
  disk_metrics: boolean
  network_metrics: boolean
  collect_dmesg: boolean
  collect_journalctl: boolean
  log_patterns: string[]
  custom_metrics: string[]
}

export interface FleetStats {
  total_node_groups: number
  total_nodes: number
  active_nodes: number
  pending_nodes: number
  error_nodes: number
  total_gpus: number
}

export interface HealthResponse {
  status: string
  version: string
  services: { name: string; status: string; message?: string }[]
}

// API functions
export const fetchStats = () => api.get<FleetStats>('/stats').then(r => r.data)

export const fetchHealth = () => api.get<HealthResponse>('/health').then(r => r.data)

export const fetchNodeGroups = () => api.get<NodeGroup[]>('/nodegroups').then(r => r.data)

export const fetchNodeGroup = (id: number) =>
  api.get<NodeGroupDetail>(`/nodegroups/${id}`).then(r => r.data)

export const createNodeGroup = (data: {
  name: string
  description?: string
  ssh_user: string
  ssh_port: number
  ssh_auth_type?: 'key' | 'password'
  ssh_password?: string
  use_jump_host?: boolean
  jump_host?: string
  jump_port?: number
  jump_user?: string
  jump_auth_type?: 'key' | 'password'
  jump_password?: string
  remote_auth_type?: 'key' | 'password'
  remote_key_path?: string
  remote_password?: string
  ip_addresses?: string[]
  monitoring_server_id?: number
  metric_group_id?: number
}) => api.post<NodeGroup>('/nodegroups/with-nodes', data).then(r => r.data)

export const deleteNodeGroup = (id: number) =>
  api.delete(`/nodegroups/${id}`)

export const uploadSSHKey = (nodeGroupId: number, file: File) => {
  const formData = new FormData()
  formData.append('key_file', file)
  return api.post(`/nodegroups/${nodeGroupId}/ssh-key`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export const uploadJumpHostKey = (nodeGroupId: number, file: File) => {
  const formData = new FormData()
  formData.append('key_file', file)
  return api.post(`/nodegroups/${nodeGroupId}/jump-key`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export const verifyConnectivity = (nodeGroupId: number) =>
  api.post(`/nodegroups/${nodeGroupId}/verify`)

export const installExporters = (nodeGroupId: number, nodeIds?: number[]) =>
  api.post(`/nodegroups/${nodeGroupId}/install`, { node_ids: nodeIds, force: false })

export const installExportersForce = (nodeGroupId: number, nodeIds?: number[]) =>
  api.post(`/nodegroups/${nodeGroupId}/install`, { node_ids: nodeIds, force: true })

export const fetchNodes = (nodeGroupId: number) =>
  api.get<Node[]>(`/nodegroups/${nodeGroupId}/nodes`).then(r => r.data)

export const addNodes = (nodeGroupId: number, ipAddresses: string[]) =>
  api.post<Node[]>(`/nodegroups/${nodeGroupId}/nodes/bulk`, {
    ip_addresses: ipAddresses,
  }).then(r => r.data)

export const deleteNode = (nodeGroupId: number, nodeId: number) =>
  api.delete(`/nodegroups/${nodeGroupId}/nodes/${nodeId}`)

export const fetchMetricConfigs = () =>
  api.get<MetricConfig[]>('/metrics/configs').then(r => r.data)

export const fetchMetricCategories = () =>
  api.get('/metrics/categories').then(r => r.data)

// Monitoring Servers
export const fetchMonitoringServers = () =>
  api.get<MonitoringServer[]>('/monitoring-servers').then(r => r.data)

export const fetchMonitoringServer = (id: number) =>
  api.get<MonitoringServer>(`/monitoring-servers/${id}`).then(r => r.data)

// Metric Groups
export const fetchMetricGroups = () =>
  api.get<MetricGroup[]>('/metric-groups').then(r => r.data)

export const fetchMetricGroup = (id: number) =>
  api.get<MetricGroup>(`/metric-groups/${id}`).then(r => r.data)

// Node Groups with their associated servers
export const fetchNodeGroupsWithServers = () =>
  api.get<(NodeGroup & { monitoring_server?: MonitoringServer })[]>('/nodegroups').then(r => r.data)

// Monitoring Configuration
export interface MonitoringConfig {
  id: number
  monitoring_server_ip?: string
  monitoring_server_hostname?: string
  prometheus_port: number
  loki_port: number
  grafana_port: number
  grafana_admin_user: string
  prometheus_retention_time: string
  prometheus_retention_size: string
  prometheus_scrape_interval: string
  loki_retention_days: number
  use_push_gateway: boolean
  push_gateway_port: number
  setup_monitoring_stack: boolean
  monitoring_ssh_user?: string
  monitoring_ssh_port: number
  monitoring_ssh_auth_type: 'key' | 'password'
  has_monitoring_ssh_key: boolean
  has_monitoring_ssh_password: boolean
  // Jump host fields
  monitoring_use_jump_host: boolean
  monitoring_jump_host?: string
  monitoring_jump_port: number
  monitoring_jump_user?: string
  monitoring_jump_auth_type: 'key' | 'password'
  has_monitoring_jump_key: boolean
  has_monitoring_jump_password: boolean
  monitoring_remote_auth_type: 'key' | 'password'
  monitoring_remote_key_path?: string
  has_monitoring_remote_password: boolean
  created_at: string
  updated_at: string
}

export interface MonitoringEndpoints {
  configured: boolean
  server?: string
  message?: string
  endpoints: {
    prometheus?: string
    loki?: string
    grafana?: string
    push_gateway?: string
  }
  retention?: {
    prometheus_time: string
    prometheus_size: string
    loki_days: number
  }
  scrape_interval?: string
}

export const fetchMonitoringConfig = () =>
  api.get<MonitoringConfig>('/monitoring/config').then(r => r.data)

export const updateMonitoringConfig = (config: Partial<MonitoringConfig>) =>
  api.put<MonitoringConfig>('/monitoring/config', config).then(r => r.data)

export const fetchMonitoringEndpoints = () =>
  api.get<MonitoringEndpoints>('/monitoring/endpoints').then(r => r.data)

export const testMonitoringConnection = () =>
  api.post('/monitoring/test-connection').then(r => r.data)

export const checkMonitoringServices = () =>
  api.post('/monitoring/check-services').then(r => r.data)

export const installMonitoringStack = () =>
  api.post('/monitoring/install-stack').then(r => r.data)

export const deleteMonitoringConfig = () =>
  api.delete('/monitoring/config').then(r => r.data)

export const syncPrometheusTargets = () =>
  api.post('/monitoring/sync-targets').then(r => r.data)

// Installation status
export interface InstallationLog {
  timestamp: string
  level: string
  message: string
}

export interface InstallationStatus {
  status: 'starting' | 'running' | 'completed' | 'failed'
  started_at: string
  server: string
  logs: InstallationLog[]
  current_step: string
  completed: boolean
  error?: string
  completed_at?: string
}

export const fetchInstallationStatus = (jobId: string) =>
  api.get<InstallationStatus>(`/monitoring/install-status/${jobId}`).then(r => r.data)

export default api
