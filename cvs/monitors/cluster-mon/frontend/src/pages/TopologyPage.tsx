import { useState, useEffect } from 'react'
import { Network, RefreshCw, AlertCircle, CheckCircle, Table, Share2 } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { api } from '@/services/api'
import { useClusterStore } from '@/stores/clusterStore'
import { NetworkTopologyGraph } from '@/components/NetworkTopologyGraph'

interface LLDPNeighbor {
  chassis?: {
    name?: string
    descr?: string
    mgmt_ip?: string
  }
  port?: {
    id?: string
    descr?: string
  }
}

interface LLDPData {
  [node: string]: {
    lldp?: {
      interface?: {
        [ifname: string]: {
          chassis?: any
          port?: any
          [key: string]: any
        }
      }
    }
  }
}

export function TopologyPage() {
  const nodes = useClusterStore((state) => state.nodes)
  const [lldpData, setLldpData] = useState<any>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'graph' | 'table'>('graph')

  const loadTopologyData = async () => {
    try {
      setLoading(true)
      setError(null)
      const metrics = await api.getLatestMetrics()

      // Extract LLDP data from metrics
      // API returns: { timestamp, gpu, nic: { lldp, ... } }
      if (metrics && metrics.nic && metrics.nic.lldp) {
        setLldpData(metrics.nic.lldp)
      } else {
        setLldpData({})
      }
    } catch (err: any) {
      console.error('Failed to load topology data:', err)
      setError(err.message || 'Failed to load topology data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadTopologyData()

    // Auto-refresh every 180 seconds (3 minutes)
    const interval = setInterval(loadTopologyData, 180000)
    return () => clearInterval(interval)
  }, [])

  // Parse LLDP data to extract neighbor information
  const parseNeighbors = () => {
    const neighbors: any[] = []

    Object.entries(lldpData).forEach(([node, nodeData]: [string, any]) => {
      if (!nodeData || typeof nodeData !== 'object') return

      // Handle LLDP data structure: nodeData.lldp.interface or nodeData.interface
      const lldpInfo = nodeData.lldp || nodeData
      let interfaces = lldpInfo?.interface

      if (!interfaces) return

      // Handle both array and object formats
      // Array format: [{ "enp195s0": {...} }, { "enp195s0": {...} }]
      // Object format: { "enp195s0": {...}, "eth0": {...} }
      const interfaceList = Array.isArray(interfaces) ? interfaces : [interfaces]

      interfaceList.forEach((interfaceEntry: any) => {
        // Each entry is an object with interface name as key
        Object.entries(interfaceEntry).forEach(([ifname, ifdata]: [string, any]) => {
          if (!ifdata || typeof ifdata !== 'object') return

          // Extract chassis information (neighbor switch/host)
          // The chassis object has the neighbor hostname as key
          const chassis = ifdata.chassis || {}

          // Get chassis name and data
          let chassisName = 'Unknown'
          let chassisData: any = {}

          // Handle different chassis formats
          if (chassis.id) {
            // Format: chassis: { id: { type: "local", value: "..." } }
            chassisName = chassis.id.value || 'Unknown'
          } else {
            // Format: chassis: { "hostname": { id: {...}, descr: "..." } }
            const chassisKeys = Object.keys(chassis)
            if (chassisKeys.length > 0) {
              chassisName = chassisKeys[0]
              chassisData = chassis[chassisName] || {}
            }
          }

          // Extract port information
          const port = ifdata.port || {}

          // Get management IP (can be string or array)
          let mgmtIp = ''
          if (chassisData['mgmt-ip']) {
            mgmtIp = Array.isArray(chassisData['mgmt-ip'])
              ? chassisData['mgmt-ip'][0]
              : chassisData['mgmt-ip']
          }

          neighbors.push({
            node,
            interface: ifname,
            neighbor_name: chassisName,
            neighbor_port: port.id?.value || port.descr?.value || 'Unknown',
            neighbor_descr: chassisData.descr || '',
            neighbor_mgmt_ip: mgmtIp,
            via: ifdata.via || 'LLDP',
            age: ifdata.age || '',
            rid: ifdata.rid || '',
          })
        })
      })
    })

    return neighbors
  }

  const neighbors = parseNeighbors()

  return (
    <div className="max-w-7xl">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Network Topology</h1>
          <p className="text-gray-600">LLDP neighbor discovery and network connections</p>
        </div>
        <div className="flex items-center gap-3">
          {/* View Mode Toggle */}
          <div className="flex items-center gap-2 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setViewMode('graph')}
              className={`flex items-center gap-2 px-3 py-2 rounded-md transition-colors ${
                viewMode === 'graph'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Share2 className="h-4 w-4" />
              Graph
            </button>
            <button
              onClick={() => setViewMode('table')}
              className={`flex items-center gap-2 px-3 py-2 rounded-md transition-colors ${
                viewMode === 'table'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Table className="h-4 w-4" />
              Table
            </button>
          </div>

          <button
            onClick={loadTopologyData}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <Card className="mb-6 border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3 text-red-800">
              <AlertCircle className="h-5 w-5" />
              <span>{error}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-gray-600">Total Nodes</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{Object.keys(lldpData).length}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-gray-600">LLDP Neighbors</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{neighbors.length}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium text-gray-600">Unique Switches</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {new Set(neighbors.map(n => n.neighbor_name)).size}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Graph View */}
      {viewMode === 'graph' && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Share2 className="h-5 w-5" />
              Network Topology Graph
            </CardTitle>
            <CardDescription>
              Interactive visualization of network connections. Click nodes for details. Use controls to zoom and pan.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-12">
                <RefreshCw className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
                <p className="text-gray-600">Loading topology data...</p>
              </div>
            ) : neighbors.length === 0 ? (
              <div className="text-center py-12">
                <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 mb-2">No LLDP neighbors found</p>
                <p className="text-sm text-gray-500">
                  Make sure LLDP daemon (lldpd) is installed and running on cluster nodes.
                </p>
              </div>
            ) : (
              <NetworkTopologyGraph neighbors={neighbors} />
            )}
          </CardContent>
        </Card>
      )}

      {/* LLDP Neighbors Table */}
      {viewMode === 'table' && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Network className="h-5 w-5" />
              LLDP Neighbors
            </CardTitle>
            <CardDescription>
              Network topology discovered via LLDP (Link Layer Discovery Protocol)
            </CardDescription>
          </CardHeader>
          <CardContent>
          {loading ? (
            <div className="text-center py-12">
              <RefreshCw className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
              <p className="text-gray-600">Loading topology data...</p>
            </div>
          ) : neighbors.length === 0 ? (
            <div className="text-center py-12">
              <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600 mb-2">No LLDP neighbors found</p>
              <p className="text-sm text-gray-500">
                Make sure LLDP daemon (lldpd) is installed and running on cluster nodes.
                <br />
                You can install it from the Configuration page under "Package Installs".
              </p>
            </div>
          ) : (
            <CustomDataTable
              columns={[
                { title: 'Node', data: 'node' },
                { title: 'Interface', data: 'interface' },
                { title: 'Neighbor', data: 'neighbor_name' },
                { title: 'Neighbor Port', data: 'neighbor_port' },
                { title: 'Description', data: 'neighbor_descr' },
                { title: 'Mgmt IP', data: 'neighbor_mgmt_ip' },
              ]}
              data={neighbors}
              defaultPageLength={50}
              pageLengthOptions={[50, 100, 500]}
            />
          )}
        </CardContent>
      </Card>
      )}

      {/* Instructions */}
      {neighbors.length === 0 && !loading && (
        <Card className="mt-6 bg-blue-50 border-blue-200">
          <CardHeader>
            <CardTitle className="text-base">How to enable LLDP</CardTitle>
          </CardHeader>
          <CardContent className="text-sm space-y-2">
            <p>1. Go to the Configuration page</p>
            <p>2. Scroll to the "Package Installs" section</p>
            <p>3. Click "Install" on the LLDP Daemon package</p>
            <p>4. Wait for installation to complete (may take a few minutes)</p>
            <p>5. Return to this page and click "Refresh" to see topology data</p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
