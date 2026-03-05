import { useEffect, useState } from 'react'
import { RefreshCw, FileText, AlertCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'

export function LogsPage() {
  const [systemLogs, setSystemLogs] = useState<any[]>([])
  const [userspaceLogs, setUserspaceLogs] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<string>('')
  const [error, setError] = useState<string | null>(null)

  const fetchLogs = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/logs/dmesg')
      const data = await response.json()

      // Transform system errors data into table format
      const systemLogsArray: any[] = []
      if (data.dmesg_errors) {
        Object.entries(data.dmesg_errors).forEach(([node, logOutput]: [string, any]) => {
          if (typeof logOutput === 'string' && logOutput.trim()) {
            systemLogsArray.push({
              node,
              logs: logOutput,
            })
          }
        })
      }

      // Transform userspace errors data into table format
      const userspaceLogsArray: any[] = []
      if (data.userspace_errors) {
        Object.entries(data.userspace_errors).forEach(([node, logOutput]: [string, any]) => {
          if (typeof logOutput === 'string' && logOutput.trim()) {
            userspaceLogsArray.push({
              node,
              logs: logOutput,
            })
          }
        })
      }

      setSystemLogs(systemLogsArray)
      setUserspaceLogs(userspaceLogsArray)
      setLastUpdate(new Date().toLocaleString())
    } catch (err: any) {
      console.error('Failed to fetch logs:', err)
      setError(err.message || 'Failed to fetch logs')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLogs()
  }, [])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">System Logs</h1>
          <p className="text-gray-600">Critical system errors from dmesg (emerg, alert, crit, err)</p>
        </div>
        <button
          onClick={fetchLogs}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Summary */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-red-600">{systemLogs.length}</div>
            <p className="text-xs text-gray-600">Nodes with System Errors</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-orange-600">{userspaceLogs.length}</div>
            <p className="text-xs text-gray-600">Nodes with Userspace Errors</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">
              {systemLogs.reduce((sum, l) => sum + (l.logs.split('\n').length || 0), 0)}
            </div>
            <p className="text-xs text-gray-600">System Error Lines</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-sm text-gray-600">Last Updated</div>
            <div className="text-lg font-medium">{lastUpdate || 'Never'}</div>
          </CardContent>
        </Card>
      </div>

      {/* Error Message */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3 text-red-800">
              <AlertCircle className="h-5 w-5" />
              <span>{error}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* System Error Logs Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-red-600" />
            System Error Logs (emerg, alert, crit, err)
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-12">
              <RefreshCw className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
              <p className="text-gray-600">Collecting system logs from all nodes...</p>
            </div>
          ) : systemLogs.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="h-12 w-12 text-green-400 mx-auto mb-4" />
              <p className="text-gray-600 mb-2">No critical system errors found</p>
              <p className="text-sm text-gray-500">
                All nodes are clean - no emerg, alert, crit, or err level messages in dmesg
              </p>
            </div>
          ) : (
            <CustomDataTable
              columns={[
                { title: 'Node', data: 'node', className: 'dt-left font-medium' },
                {
                  title: 'Logs',
                  data: 'logs',
                  className: 'dt-left font-mono text-xs',
                  render: (data: string) => {
                    return `<pre class="whitespace-pre-wrap break-words max-w-4xl">${data}</pre>`
                  }
                },
              ]}
              data={systemLogs}
              defaultPageLength={50}
              pageLengthOptions={[50, 100, 500]}
            />
          )}
        </CardContent>
      </Card>

      {/* Userspace Error Logs Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-orange-600" />
            Userspace Errors (OOM, Crashes, ML Frameworks)
          </CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-12">
              <RefreshCw className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
              <p className="text-gray-600">Collecting userspace logs from all nodes...</p>
            </div>
          ) : userspaceLogs.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="h-12 w-12 text-green-400 mx-auto mb-4" />
              <p className="text-gray-600 mb-2">No userspace errors found</p>
              <p className="text-sm text-gray-500">
                No OOM kills, segfaults, crashes, warnings, or ML framework errors detected
              </p>
            </div>
          ) : (
            <CustomDataTable
              columns={[
                { title: 'Node', data: 'node', className: 'dt-left font-medium' },
                {
                  title: 'Logs',
                  data: 'logs',
                  className: 'dt-left font-mono text-xs',
                  render: (data: string) => {
                    return `<pre class="whitespace-pre-wrap break-words max-w-4xl">${data}</pre>`
                  }
                },
              ]}
              data={userspaceLogs}
              defaultPageLength={50}
              pageLengthOptions={[50, 100, 500]}
            />
          )}
        </CardContent>
      </Card>

      {/* Instructions */}
      {(systemLogs.length > 0 || userspaceLogs.length > 0) && (
        <Card className="bg-blue-50 border-blue-200">
          <CardHeader>
            <CardTitle className="text-base">About these logs</CardTitle>
          </CardHeader>
          <CardContent className="text-sm space-y-3">
            <div>
              <p className="font-semibold mb-1">System Error Logs (kernel level):</p>
              <ul className="list-disc list-inside ml-2 space-y-1 text-xs">
                <li><span className="font-semibold">:emerg</span> - System is unusable</li>
                <li><span className="font-semibold">:alert</span> - Action must be taken immediately</li>
                <li><span className="font-semibold">:crit</span> - Critical conditions</li>
                <li><span className="font-semibold">:err</span> - Error conditions</li>
              </ul>
            </div>
            <div className="pt-2 border-t">
              <p className="font-semibold mb-1">Userspace Errors (application level):</p>
              <ul className="list-disc list-inside ml-2 space-y-1 text-xs">
                <li><span className="font-semibold">OOM</span> - Out of memory kills</li>
                <li><span className="font-semibold">Segfault</span> - Segmentation faults, general protection faults</li>
                <li><span className="font-semibold">Call Trace</span> - Stack traces and crashes</li>
                <li><span className="font-semibold">Hardware Error / MCE</span> - Machine check exceptions</li>
                <li><span className="font-semibold">ML Frameworks</span> - PyTorch, TensorFlow, Megatron, JAX, VLLM, SGLang, Triton</li>
              </ul>
            </div>
            <p className="mt-2 text-blue-800 pt-2 border-t text-xs">
              Review these logs to identify hardware failures, driver issues, and application crashes.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
