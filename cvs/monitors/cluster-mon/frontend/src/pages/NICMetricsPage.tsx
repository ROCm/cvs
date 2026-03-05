import { useEffect, useState } from 'react'
import { RefreshCw, Network } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { useClusterStore } from '@/stores/clusterStore'

export function NICMetricsPage() {
  const latestMetrics = useClusterStore((state) => state.latestMetrics)
  const cachedAdvancedData = useClusterStore((state) => state.nicAdvancedData)
  const setNICAdvancedData = useClusterStore((state) => state.setNICAdvancedData)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<string>('')
  const [loadingAdvanced, setLoadingAdvanced] = useState(false)

  console.log('NIC Metrics Page Render:', {
    hasLatestMetrics: !!latestMetrics,
    hasNic: !!latestMetrics?.nic,
    hasRdmaStats: !!latestMetrics?.nic?.rdma_stats,
    nicKeys: latestMetrics?.nic ? Object.keys(latestMetrics.nic) : []
  })

  useEffect(() => {
    if (latestMetrics?.timestamp) {
      setLastUpdate(new Date(latestMetrics.timestamp).toLocaleString())
    }
  }, [latestMetrics])

  // Fetch advanced NIC info automatically on mount
  const fetchAdvancedInfo = async () => {
    setLoadingAdvanced(true)
    console.log('Fetching advanced NIC info...')
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 180000) // 3 minute timeout

      const response = await fetch('/api/software/nic/advanced', {
        signal: controller.signal
      })
      clearTimeout(timeoutId)

      const data = await response.json()
      console.log('Advanced NIC data received:', {
        nic_pcie_count: Object.values(data.nic_pcie || {}).reduce((sum: number, nics: any) => sum + Object.keys(nics).length, 0),
        congestion_count: Object.values(data.congestion || {}).reduce((sum: number, ifaces: any) => sum + Object.keys(ifaces).length, 0),
      })
      setNICAdvancedData(data)  // Store in global cache
    } catch (error) {
      console.error('Failed to fetch advanced NIC info:', error)
    } finally {
      setLoadingAdvanced(false)
    }
  }

  // Auto-fetch on mount and refresh periodically
  useEffect(() => {
    // Only fetch if no cached data exists
    if (!cachedAdvancedData) {
      fetchAdvancedInfo()
    }

    if (autoRefresh) {
      const interval = setInterval(fetchAdvancedInfo, 180000) // Refresh every 3 minutes
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  // Use cached data for display (shows immediately!)
  const advancedNICData = cachedAdvancedData

  // Table 1: RDMA Links
  // Table 5: NIC PCIe Information
  const nicPCIeData = (() => {
    const data: any[] = []
    if (!advancedNICData?.nic_pcie) return data

    Object.entries(advancedNICData.nic_pcie).forEach(([node, devices]: [string, any]) => {
      if (typeof devices === 'object') {
        Object.entries(devices).forEach(([bdf, info]: [string, any]) => {
          data.push({
            node,
            bdf,
            device: info.device || 'N/A',
            pcie_gen: info.pcie_gen || 'N/A',
            link_speed_cap: info.link_speed_cap || 'N/A',
            link_width_cap: info.link_width_cap || 'N/A',
            link_speed_current: info.link_speed_current || 'N/A',
            link_width_current: info.link_width_current || 'N/A',
          })
        })
      }
    })

    return data
  })()

  // Table 6: Congestion Control Information (PFC and DCQCN)
  const congestionData = (() => {
    const data: any[] = []
    const allKeys = new Set<string>()

    if (!advancedNICData?.congestion) return { rows: data, activeColumns: [] }

    // First pass: collect all keys and check for non-zero values
    const columnMaxValues: Record<string, number> = {}

    Object.entries(advancedNICData.congestion).forEach(([node, interfaces]: [string, any]) => {
      if (typeof interfaces === 'object') {
        Object.entries(interfaces).forEach(([iface, stats]: [string, any]) => {
          if (typeof stats === 'object') {
            Object.entries(stats).forEach(([key, value]) => {
              allKeys.add(key)
              const numValue = typeof value === 'number' ? Math.abs(value) : 0
              columnMaxValues[key] = Math.max(columnMaxValues[key] || 0, numValue)
            })
          }
        })
      }
    })

    // Filter to only keys with non-zero values
    const activeColumns = Array.from(allKeys).filter(key => columnMaxValues[key] > 0)

    // Second pass: create rows
    Object.entries(advancedNICData.congestion).forEach(([node, interfaces]: [string, any]) => {
      if (typeof interfaces === 'object') {
        Object.entries(interfaces).forEach(([iface, stats]: [string, any]) => {
          if (typeof stats === 'object') {
            const row: any = { node, interface: iface }

            activeColumns.forEach(key => {
              row[key] = stats[key] !== undefined ? stats[key] : 0
            })

            data.push(row)
          }
        })
      }
    })

    return { rows: data, activeColumns }
  })()

  // Table 1: RDMA Links
  const rdmaLinksData = (() => {
    const data: any[] = []
    if (!latestMetrics?.nic?.rdma_links) return data

    Object.entries(latestMetrics.nic.rdma_links).forEach(([node, links]: [string, any]) => {
      if (typeof links === 'object' && !links.error) {
        Object.entries(links).forEach(([device, info]: [string, any]) => {
          data.push({
            node,
            device,
            state: info.state || '-',
            physical_state: info.physical_state || '-',
            netdev: info.netdev || '-',
          })
        })
      }
    })

    return data
  })()

  // Table 2: RDMA Statistics - Only show columns with non-zero values
  const rdmaStatsData = (() => {
    const data: any[] = []
    const allKeys = new Set<string>()
    const columnMaxValues: Record<string, number> = {}

    if (!latestMetrics?.nic?.rdma_stats) return { rows: data, activeColumns: [] }

    // First pass: collect ALL unique keys and track max values
    Object.entries(latestMetrics.nic.rdma_stats).forEach(([node, devices]: [string, any]) => {
      if (typeof devices === 'object' && !devices.error) {
        Object.entries(devices).forEach(([device, stats]: [string, any]) => {
          if (typeof stats === 'object') {
            Object.entries(stats).forEach(([key, value]) => {
              allKeys.add(key)
              const numValue = typeof value === 'number' ? Math.abs(value) : (parseInt(value) || 0)
              columnMaxValues[key] = Math.max(columnMaxValues[key] || 0, numValue)
            })
          }
        })
      }
    })

    // Filter to only keys that have non-zero values somewhere
    const activeColumns = Array.from(allKeys).filter(key => columnMaxValues[key] > 0)

    // Second pass: create rows with only active columns
    Object.entries(latestMetrics.nic.rdma_stats).forEach(([node, devices]: [string, any]) => {
      if (typeof devices === 'object' && !devices.error) {
        Object.entries(devices).forEach(([device, stats]: [string, any]) => {
          if (typeof stats === 'object') {
            const row: any = { node, device }

            // Add only active columns
            activeColumns.forEach(key => {
              row[key] = stats[key] !== undefined ? stats[key] : 0
            })

            data.push(row)
          }
        })
      }
    })

    return { rows: data, activeColumns }
  })()

  // Table 3: Ethtool Statistics (ONLY Total TX/RX packets, bytes, errors, drops)
  const ethtoolStatsData = (() => {
    const data: any[] = []

    if (!latestMetrics?.nic?.ethtool_stats) return data

    Object.entries(latestMetrics.nic.ethtool_stats).forEach(([node, interfaces]: [string, any]) => {
      if (typeof interfaces === 'object' && !interfaces.error) {
        Object.entries(interfaces).forEach(([ifname, stats]: [string, any]) => {
          if (typeof stats === 'object') {
            data.push({
              node,
              interface: ifname,
              rx_packets: stats.rx_packets || 0,
              tx_packets: stats.tx_packets || 0,
              rx_bytes: stats.rx_bytes || 0,
              tx_bytes: stats.tx_bytes || 0,
              rx_errors: stats.rx_errors || 0,
              tx_errors: stats.tx_errors || 0,
              rx_dropped: stats.rx_dropped || 0,
              tx_dropped: stats.tx_dropped || 0,
            })
          }
        })
      }
    })

    return data
  })()

  // Table 4: RDMA Resources
  const rdmaResourcesData = (() => {
    const data: any[] = []
    if (!latestMetrics?.nic?.rdma_resources) return data

    Object.entries(latestMetrics.nic.rdma_resources).forEach(([node, devices]: [string, any]) => {
      if (typeof devices === 'object' && !devices.error) {
        Object.entries(devices).forEach(([device, resources]: [string, any]) => {
          data.push({
            node,
            device,
            pd: resources.pd || 0,
            cq: resources.cq || 0,
            qp: resources.qp || 0,
            cm_id: resources.cm_id || 0,
            mr: resources.mr || 0,
            ctx: resources.ctx || 0,
            srq: resources.srq || 0,
          })
        })
      }
    })

    return data
  })()

  // Ethtool columns - simplified to only essential metrics
  const ethtoolColumns = [
    { title: 'Node', data: 'node', className: 'dt-left font-medium' },
    { title: 'Interface', data: 'interface', className: 'dt-left font-mono' },
    { title: 'RX Packets', data: 'rx_packets', className: 'dt-right font-mono text-xs' },
    { title: 'TX Packets', data: 'tx_packets', className: 'dt-right font-mono text-xs' },
    { title: 'RX Bytes', data: 'rx_bytes', className: 'dt-right font-mono text-xs' },
    { title: 'TX Bytes', data: 'tx_bytes', className: 'dt-right font-mono text-xs' },
    {
      title: 'RX Errors',
      data: 'rx_errors',
      className: 'dt-right',
      render: (data: number) => data > 0 ? `<span class="text-red-600 font-medium">${data}</span>` : data,
    },
    {
      title: 'TX Errors',
      data: 'tx_errors',
      className: 'dt-right',
      render: (data: number) => data > 0 ? `<span class="text-red-600 font-medium">${data}</span>` : data,
    },
    {
      title: 'RX Dropped',
      data: 'rx_dropped',
      className: 'dt-right',
      render: (data: number) => data > 0 ? `<span class="text-red-600 font-medium">${data}</span>` : data,
    },
    {
      title: 'TX Dropped',
      data: 'tx_dropped',
      className: 'dt-right',
      render: (data: number) => data > 0 ? `<span class="text-red-600 font-medium">${data}</span>` : data,
    },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">NIC Metrics</h1>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-4 h-4 text-blue-600 rounded"
            />
            Auto-refresh (60s)
          </label>
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <RefreshCw className="h-4 w-4" />
            Last update: {lastUpdate || 'Never'}
          </div>
        </div>
      </div>

      {rdmaLinksData.length > 0 && (
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">{rdmaLinksData.length}</div>
              <p className="text-xs text-gray-600">Total RDMA Links</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">
                {rdmaLinksData.filter(l => l.state === 'ACTIVE').length}
              </div>
              <p className="text-xs text-gray-600">Active Links</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold text-red-600">
                {rdmaLinksData.filter(l => l.state !== 'ACTIVE').length}
              </div>
              <p className="text-xs text-gray-600">Links Down</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-2xl font-bold">
                {rdmaResourcesData.reduce((sum, r) => sum + (r.qp || 0), 0).toLocaleString()}
              </div>
              <p className="text-xs text-gray-600">RDMA Active Queue Pairs</p>
            </CardContent>
          </Card>
        </div>
      )}

      {rdmaLinksData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>RDMA Links</CardTitle>
          </CardHeader>
          <CardContent>
            <CustomDataTable
              columns={[
                { title: 'Node', data: 'node', className: 'dt-left font-medium' },
                { title: 'RDMA Device', data: 'device', className: 'dt-left font-mono' },
                {
                  title: 'State',
                  data: 'state',
                  className: 'dt-center',
                  render: (data: string) => {
                    const color = data === 'ACTIVE' ? 'text-green-600 font-medium' : 'text-red-600'
                    return `<span class="${color}">${data}</span>`
                  },
                },
                {
                  title: 'Physical State',
                  data: 'physical_state',
                  className: 'dt-center',
                  render: (data: string) => {
                    const color = data === 'LINK_UP' ? 'text-green-600' : 'text-gray-600'
                    return `<span class="${color}">${data}</span>`
                  },
                },
                { title: 'Network Device', data: 'netdev', className: 'dt-left font-mono text-xs' },
              ]}
              data={rdmaLinksData}
              defaultPageLength={50}
              pageLengthOptions={[50, 100, 1000]}
            />
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>
            RDMA Statistics {rdmaStatsData.activeColumns.length > 0 ? `(Showing ${rdmaStatsData.activeColumns.length} active counters - Errors/Drops/Retries in red)` : ''}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {rdmaStatsData.rows.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <p>No RDMA statistics available or all counters are zero</p>
              <p className="text-xs mt-2">Waiting for data with non-zero values...</p>
            </div>
          ) : (() => {
            const rdmaStatsColumns = [
              { title: 'Node', data: 'node', className: 'dt-left font-medium' },
              { title: 'Device', data: 'device', className: 'dt-left font-mono' },
            ]

            // Add only active columns (that have non-zero values)
            rdmaStatsData.activeColumns.sort().forEach(key => {
              const lowerKey = key.toLowerCase()
              const isError = lowerKey.includes('err') || lowerKey.includes('drop') ||
                             lowerKey.includes('retry') || lowerKey.includes('timeout') ||
                             lowerKey.includes('discard') || lowerKey.includes('fail') ||
                             lowerKey.includes('nak')

              rdmaStatsColumns.push({
                title: key.replace(/_/g, ' '),
                data: key,
                className: 'dt-right text-xs',
                render: isError
                  ? (data: number) => data > 0 ? `<span class="text-red-600 font-medium">${data}</span>` : data
                  : undefined,
              })
            })

            return (
              <CustomDataTable
                columns={rdmaStatsColumns}
                data={rdmaStatsData.rows}
                defaultPageLength={50}
                pageLengthOptions={[50, 100, 1000]}
              />
            )
          })()}
        </CardContent>
      </Card>

      {ethtoolStatsData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Ethtool Statistics (Errors and Drops highlighted in red)</CardTitle>
          </CardHeader>
          <CardContent>
            <CustomDataTable
              columns={ethtoolColumns}
              data={ethtoolStatsData}
              defaultPageLength={50}
              pageLengthOptions={[50, 100, 1000]}
            />
          </CardContent>
        </Card>
      )}

      {rdmaResourcesData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>RDMA Resources</CardTitle>
          </CardHeader>
          <CardContent>
            <CustomDataTable
              columns={[
                { title: 'Node', data: 'node', className: 'dt-left font-medium' },
                { title: 'RDMA Device', data: 'device', className: 'dt-left font-mono' },
                { title: 'Protection Domains (pd)', data: 'pd', className: 'dt-right' },
                { title: 'Completion Queues (cq)', data: 'cq', className: 'dt-right' },
                { title: 'Queue Pairs (qp)', data: 'qp', className: 'dt-right' },
                { title: 'CM IDs (cm_id)', data: 'cm_id', className: 'dt-right' },
                { title: 'Memory Regions (mr)', data: 'mr', className: 'dt-right' },
                { title: 'Contexts (ctx)', data: 'ctx', className: 'dt-right' },
                { title: 'Shared RQ (srq)', data: 'srq', className: 'dt-right' },
              ]}
              data={rdmaResourcesData}
              defaultPageLength={50}
              pageLengthOptions={[50, 100, 1000]}
            />
          </CardContent>
        </Card>
      )}

      {/* NIC PCIe Information table removed - lspci data collection issue */}

      {/* Table 6: Congestion Control */}
      <Card>
        <CardHeader>
          <CardTitle>Congestion Control Metrics - PFC & ECN</CardTitle>
        </CardHeader>
        <CardContent>
          {loadingAdvanced ? (
            <div className="text-center py-8 text-gray-500">
              <RefreshCw className="h-8 w-8 mx-auto mb-2 animate-spin" />
              <p>Loading congestion metrics...</p>
            </div>
          ) : congestionData.rows.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <p>No congestion data available</p>
            </div>
          ) : (() => {
        const congestionColumns = [
          { title: 'Node', data: 'node', className: 'dt-left font-medium' },
          { title: 'Interface', data: 'interface', className: 'dt-left font-mono' },
        ]

        congestionData.activeColumns.sort().forEach(key => {
          const lowerKey = key.toLowerCase()
          const isError = lowerKey.includes('err') || lowerKey.includes('drop') ||
                         lowerKey.includes('discard')

          congestionColumns.push({
            title: key.replace(/_/g, ' '),
            data: key,
            className: 'dt-right text-xs',
            render: isError
              ? (data: number) => data > 0 ? `<span class="text-red-600 font-medium">${data}</span>` : data
              : (data: number) => data > 0 ? `<span class="font-medium">${data}</span>` : data,
          })
        })

        return (
          <CustomDataTable
            columns={congestionColumns}
            data={congestionData.rows}
            defaultPageLength={50}
            pageLengthOptions={[50, 100, 1000]}
          />
        )
      })()}
        </CardContent>
      </Card>

      {rdmaLinksData.length === 0 && rdmaStatsData.rows.length === 0 && (
        <Card>
          <CardContent className="py-12">
            <div className="text-center text-gray-500">
              <Network className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No NIC data available</p>
              <p className="text-sm mt-2">
                Configure nodes and SSH access to see NIC metrics
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
