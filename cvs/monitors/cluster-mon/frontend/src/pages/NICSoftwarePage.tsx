import { useEffect, useState } from 'react'
import { RefreshCw, Network } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'
import { useClusterStore } from '@/stores/clusterStore'

export function NICSoftwarePage() {
  const cachedSoftwareData = useClusterStore((state) => state.nicSoftwareData)
  const setNICSoftwareData = useClusterStore((state) => state.setNICSoftwareData)

  const [loading, setLoading] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<string>('')
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchSoftwareInfo = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/software/nic/devlink')
      const data = await response.json()
      setNICSoftwareData(data)
      setLastUpdate(new Date().toLocaleString())
    } catch (error) {
      console.error('Failed to fetch NIC devlink info:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!cachedSoftwareData) {
      fetchSoftwareInfo()
    } else {
      if (cachedSoftwareData.timestamp) {
        setLastUpdate(new Date(cachedSoftwareData.timestamp).toLocaleString())
      }
    }

    if (autoRefresh) {
      const interval = setInterval(fetchSoftwareInfo, 180000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const softwareData = cachedSoftwareData

  const getDevlinkTableData = () => {
    if (!softwareData?.devlink) return []

    const rows: any[] = []
    Object.entries(softwareData.devlink).forEach(([node, devices]: [string, any]) => {
      if (devices.error) return

      Object.entries(devices).forEach(([pci_dev, info]: [string, any]) => {
        rows.push({
          node,
          pci_address: info.pci_address || '-',
          driver: info.driver || '-',
          vendor: info.vendor || '-',
          serial_number: info.serial_number || '-',
          board_serial: info.board_serial || '-',
          board_id: info.board_id || '-',
          asic_id: info.asic_id || '-',
          asic_rev: info.asic_rev || '-',
          fw_version: info.fw_version || '-',
          fw_psid: info.fw_psid || '-',
          fw_mgmt: info.fw_mgmt || '-',
          fw_mgmt_api: info.fw_mgmt_api || '-',
          fw_cpld: info.fw_cpld || '-',
        })
      })
    })

    return rows
  }

  const devlinkData = getDevlinkTableData()

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">NIC Software Information</h1>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="w-4 h-4 text-blue-600 rounded"
            />
            Auto-refresh (180s)
          </label>
          <button
            onClick={fetchSoftwareInfo}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <div className="text-sm text-gray-600">
            Last update: {lastUpdate || 'Never'}
          </div>
        </div>
      </div>

      {/* NIC Device Information (devlink) */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            NIC Device Information (AMD AINIC, NVIDIA CX7, Broadcom Thor2)
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-8 text-gray-500">
              <RefreshCw className="h-8 w-8 mx-auto mb-2 animate-spin" />
              <p>Loading NIC device information...</p>
            </div>
          ) : devlinkData.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <p>No NIC device information available</p>
            </div>
          ) : (
            <CustomDataTable
              columns={[
                { title: 'Node', data: 'node', className: 'dt-left font-medium' },
                { title: 'PCI Address', data: 'pci_address', className: 'dt-left font-mono text-xs' },
                { title: 'Driver', data: 'driver', className: 'dt-left' },
                { title: 'Vendor', data: 'vendor', className: 'dt-left' },
                { title: 'Serial Number', data: 'serial_number', className: 'dt-left font-mono text-xs' },
                { title: 'Board Serial', data: 'board_serial', className: 'dt-left font-mono text-xs' },
                { title: 'Board ID', data: 'board_id', className: 'dt-left font-mono text-xs' },
                { title: 'ASIC ID', data: 'asic_id', className: 'dt-left font-mono text-xs' },
                { title: 'ASIC Rev', data: 'asic_rev', className: 'dt-center font-mono text-xs' },
                { title: 'FW Version', data: 'fw_version', className: 'dt-left font-mono text-xs' },
                { title: 'FW PSID', data: 'fw_psid', className: 'dt-left font-mono text-xs' },
                { title: 'FW Mgmt', data: 'fw_mgmt', className: 'dt-left font-mono text-xs' },
                { title: 'FW Mgmt API', data: 'fw_mgmt_api', className: 'dt-left font-mono text-xs' },
                { title: 'FW CPLD', data: 'fw_cpld', className: 'dt-left font-mono text-xs' },
              ]}
              data={devlinkData}
              defaultPageLength={50}
              pageLengthOptions={[50, 100, 500]}
            />
          )}
        </CardContent>
      </Card>
    </div>
  )
}
