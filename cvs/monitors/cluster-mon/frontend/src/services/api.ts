// Use relative URL when served from same origin (Docker single container)
// Or use environment variable for development
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl
  }

  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, options)
    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`API error: ${response.statusText} - ${errorText}`)
    }
    return response.json()
  }

  private async post<T>(endpoint: string, data: any): Promise<T> {
    return this.request(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    })
  }

  async getClusterStatus() {
    return this.request('/cluster/status')
  }

  async getClusterHealth() {
    return this.request('/cluster/health')
  }

  async getNodes() {
    return this.request('/nodes')
  }

  async getNodeDetails(nodeId: string) {
    return this.request(`/nodes/${nodeId}`)
  }

  async getLatestMetrics() {
    return this.request('/metrics/latest')
  }

  async getMetricsHistory(params?: {
    node?: string
    metric_type?: string
    duration?: number
  }) {
    const queryParams = new URLSearchParams()
    if (params?.node) queryParams.append('node', params.node)
    if (params?.metric_type) queryParams.append('metric_type', params.metric_type)
    if (params?.duration) queryParams.append('duration', params.duration.toString())

    const query = queryParams.toString() ? `?${queryParams.toString()}` : ''
    return this.request(`/metrics/history${query}`)
  }

  async updateConfiguration(config: any) {
    return this.post('/config/update', config)
  }

  async getCurrentConfiguration() {
    return this.request('/config/current')
  }

  async reloadConfiguration() {
    return this.post('/config/reload', {})
  }

  async uploadSshKey(file: File) {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${this.baseUrl}/ssh-keys/upload`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Failed to upload SSH key: ${response.statusText} - ${errorText}`)
    }

    return response.json()
  }

  async listSshKeys() {
    return this.request('/ssh-keys/list')
  }

  // Package management
  async getPackageList() {
    return this.request('/packages/list')
  }

  async getPackageStatus(packageId: string) {
    return this.request(`/packages/status/${packageId}`)
  }

  async installPackage(packageId: string) {
    return this.post('/packages/install', { package: packageId })
  }

  async getRCCLStatus() {
    return this.request('/rccl/status')
  }

  async getRCCLCommunicators() {
    return this.request('/rccl/communicators')
  }

  async getRCCLCommunicator(commHash: string) {
    return this.request(`/rccl/communicators/${commHash}`)
  }

  async getRCCLEvents(since?: number, until?: number, type?: string) {
    const params = new URLSearchParams()
    if (since) params.set('since', since.toString())
    if (until) params.set('until', until.toString())
    if (type) params.set('type', type)
    const query = params.toString()
    return this.request(`/rccl/events${query ? '?' + query : ''}`)
  }

  async getCollectorsStatus() {
    return this.request('/collectors/status')
  }
}

export const api = new ApiClient(API_BASE_URL)
