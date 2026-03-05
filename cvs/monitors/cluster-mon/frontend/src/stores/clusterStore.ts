import { create } from 'zustand'

export interface GPUMetrics {
  id: string
  utilization: number
  memory_used_mb: number
  memory_total_mb: number
  memory_util_percent: number
  temperature_c: number
  power_w: number
}

export interface NICMetrics {
  name: string
  state: string
  mtu: string
  mac_addr: string
  ipv4_addrs: string[]
  rdma?: {
    device: string
    state: string
    physical_state: string
  }
}

export interface NodeInfo {
  hostname: string
  status: 'healthy' | 'unhealthy' | 'unreachable'
  gpu_count: number
  avg_gpu_util: number
  avg_gpu_temp: number
  health_issues?: string[]
  gpus?: GPUMetrics[]
  nics?: NICMetrics[]
}

export interface ClusterStatus {
  total_nodes: number
  healthy_nodes: number
  unhealthy_nodes: number
  unreachable_nodes: number
  total_gpus: number
  avg_gpu_utilization: number
  status: 'healthy' | 'degraded' | 'critical' | 'no_data'
  last_update?: string
}

export interface MetricsData {
  timestamp: string
  gpu: any
  nic: any
}

interface ClusterStore {
  // State
  clusterStatus: ClusterStatus | null
  nodes: NodeInfo[]
  selectedNode: NodeInfo | null
  latestMetrics: MetricsData | null
  gpuSoftwareData: any | null
  nicSoftwareData: any | null
  nicAdvancedData: any | null
  isConnected: boolean
  error: string | null

  // Actions
  setClusterStatus: (status: ClusterStatus) => void
  setNodes: (nodes: NodeInfo[]) => void
  setSelectedNode: (node: NodeInfo | null) => void
  setLatestMetrics: (metrics: MetricsData) => void
  setGPUSoftwareData: (data: any) => void
  setNICSoftwareData: (data: any) => void
  setNICAdvancedData: (data: any) => void
  setConnected: (connected: boolean) => void
  setError: (error: string | null) => void
  updateFromWebSocket: (data: any) => void
}

export const useClusterStore = create<ClusterStore>((set) => ({
  // Initial state
  clusterStatus: null,
  nodes: [],
  selectedNode: null,
  latestMetrics: null,
  gpuSoftwareData: null,
  nicSoftwareData: null,
  nicAdvancedData: null,
  isConnected: false,
  error: null,

  // Actions
  setClusterStatus: (status) => set({ clusterStatus: status }),

  setNodes: (nodes) => set({ nodes }),

  setSelectedNode: (node) => set({ selectedNode: node }),

  setLatestMetrics: (metrics) => set({ latestMetrics: metrics }),

  setGPUSoftwareData: (data) => set({ gpuSoftwareData: data }),

  setNICSoftwareData: (data) => set({ nicSoftwareData: data }),

  setNICAdvancedData: (data) => set({ nicAdvancedData: data }),

  setConnected: (connected) => set({ isConnected: connected }),

  setError: (error) => set({ error }),

  updateFromWebSocket: (data) => {
    if (data.type === 'metrics' && data.data) {
      set({ latestMetrics: data.data, isConnected: true, error: null })
    }
  },
}))
