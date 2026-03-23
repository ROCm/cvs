import { useEffect, useState } from 'react'
import { RefreshCw, FileText, AlertCircle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { CustomDataTable } from '@/components/ui/DataTable'

export function LogsPage() {
  const [amdLogs, setAmdLogs] = useState<any[]>([])
  const [systemLogs, setSystemLogs] = useState<any[]>([])
  const [userspaceLogs, setUserspaceLogs] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<string>('')
  const [error, setError] = useState<string | null>(null)

  // Custom search state
  const [grepCommand, setGrepCommand] = useState<string>('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)

  const fetchLogs = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/logs/dmesg')
      const data = await response.json()

      // Transform AMD logs data into table format
      const amdLogsArray: any[] = []
      if (data.amd_logs) {
        Object.entries(data.amd_logs).forEach(([node, logOutput]: [string, any]) => {
          if (typeof logOutput === 'string' && logOutput.trim()) {
            amdLogsArray.push({
              node,
              logs: logOutput,
            })
          }
        })
      }

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

      setAmdLogs(amdLogsArray)
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

  const handleSearch = async () => {
    if (!grepCommand.trim()) {
      setSearchError('Please enter a grep command')
      return
    }

    setIsSearching(true)
    setSearchError(null)

    try {
      const url = `/api/logs/search?grep_command=${encodeURIComponent(grepCommand)}`
      const response = await fetch(url)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Search failed')
      }

      // Transform search results into table format
      const resultsArray: any[] = []
      if (data.results) {
        Object.entries(data.results).forEach(([node, logOutput]: [string, any]) => {
          if (typeof logOutput === 'string' && logOutput.trim()) {
            resultsArray.push({
              node,
              logs: logOutput,
            })
          }
        })
      }

      setSearchResults(resultsArray)
    } catch (err: any) {
      console.error('Failed to search logs:', err)
      setSearchError(err.message || 'Failed to search logs')
    } finally {
      setIsSearching(false)
    }
  }

  const handleClear = () => {
    setGrepCommand('')
    setSearchResults([])
    setSearchError(null)
  }

  const scrollToSection = (sectionId: string) => {
    const element = document.getElementById(sectionId)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  useEffect(() => {
    // Clear search results when page loads
    setGrepCommand('')
    setSearchResults([])
    setSearchError(null)

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
        <Card
          className="cursor-pointer hover:shadow-lg transition-shadow"
          onClick={() => scrollToSection('amd-logs-table')}
        >
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-purple-600">{amdLogs.length}</div>
            <p className="text-xs text-gray-600">Nodes with AMD Errors</p>
          </CardContent>
        </Card>
        <Card
          className="cursor-pointer hover:shadow-lg transition-shadow"
          onClick={() => scrollToSection('system-logs-table')}
        >
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-red-600">{systemLogs.length}</div>
            <p className="text-xs text-gray-600">Nodes with System Errors</p>
          </CardContent>
        </Card>
        <Card
          className="cursor-pointer hover:shadow-lg transition-shadow"
          onClick={() => scrollToSection('userspace-logs-table')}
        >
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-orange-600">{userspaceLogs.length}</div>
            <p className="text-xs text-gray-600">Nodes with Userspace Errors</p>
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

      {/* Custom Log Search */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-blue-600" />
            Custom Log Search
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Info Banner */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm">
              <p className="text-blue-800 mb-2">
                <span className="font-semibold">Powerful grep/egrep search</span> - Use pipes to filter dmesg logs (security validated)
              </p>
              <div className="text-blue-700 space-y-1">
                <div><span className="font-semibold">Examples:</span></div>
                <div className="ml-4 space-y-1 font-mono text-xs">
                  <div>• grep -i 'error'</div>
                  <div>• grep -i 'link down' | grep -v 'SATA'</div>
                  <div>• egrep -E 'GPU|CPU' | grep -v vital</div>
                  <div>• grep -A 3 'timeout' | grep -v correctable</div>
                </div>
              </div>
            </div>

            {/* Grep Command Input */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Grep Command <span className="text-red-500">*</span>
                <span className="text-gray-500 font-normal ml-2 text-xs">(Only grep/egrep with pipes - first 5 lines per node)</span>
              </label>
              <input
                type="text"
                value={grepCommand}
                onChange={(e) => setGrepCommand(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="e.g., grep -i 'error' | grep -v 'vital buffer'"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                disabled={isSearching}
              />
              <p className="text-xs text-gray-500 mt-1">
                Command will be executed as: <code className="bg-gray-100 px-1">sudo dmesg -T | [your grep command] | head -5</code>
              </p>
            </div>

            {/* Buttons */}
            <div className="flex gap-3">
              <button
                onClick={handleSearch}
                disabled={isSearching || !grepCommand.trim()}
                className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RefreshCw className={`h-4 w-4 ${isSearching ? 'animate-spin' : ''}`} />
                {isSearching ? 'Searching...' : 'Search'}
              </button>
              <button
                onClick={handleClear}
                disabled={isSearching}
                className="px-6 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:opacity-50"
              >
                Clear
              </button>
            </div>

            {/* Search Error */}
            {searchError && (
              <div className="flex items-center gap-2 text-red-600 text-sm">
                <AlertCircle className="h-4 w-4" />
                <span>{searchError}</span>
              </div>
            )}

            {/* Search Results Table */}
            {searchResults.length > 0 && (
              <div className="mt-4">
                <p className="text-sm text-gray-600 mb-3">
                  Found results in {searchResults.length} node(s) - showing first 5 lines per node
                </p>
                <CustomDataTable
                  columns={[
                    { title: 'Node', data: 'node', className: 'dt-left font-medium' },
                    {
                      title: 'Matching Logs (first 5 lines)',
                      data: 'logs',
                      className: 'dt-left font-mono text-xs',
                      render: (data: string) => {
                        return `<pre class="whitespace-pre-wrap break-words max-w-4xl">${data}</pre>`
                      }
                    },
                  ]}
                  data={searchResults}
                  defaultPageLength={25}
                  pageLengthOptions={[25, 50, 100]}
                />
              </div>
            )}

            {/* No Results Message */}
            {searchResults.length === 0 && grepCommand && !isSearching && !searchError && (
              <div className="text-center py-8 text-gray-500">
                No matching logs found for command: <span className="font-mono font-semibold text-xs">"{grepCommand}"</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* AMD Logs Table - FIRST TABLE */}
      <div id="amd-logs-table">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-purple-600" />
              AMD Hardware & Driver Logs (PCIe, XGMI, GPUs, CPUs, NICs)
            </CardTitle>
          </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-12">
              <RefreshCw className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-4" />
              <p className="text-gray-600">Collecting AMD logs from all nodes...</p>
            </div>
          ) : amdLogs.length === 0 ? (
            <div className="text-center py-12">
              <FileText className="h-12 w-12 text-green-400 mx-auto mb-4" />
              <p className="text-gray-600 mb-2">No AMD-specific errors found</p>
              <p className="text-sm text-gray-500">
                No errors detected for: PCIe, XGMI, amdgpu, EPYC, CPUs, NICs (ionic/bnxt/mlnx), or Link issues
              </p>
            </div>
          ) : (
            <CustomDataTable
              columns={[
                { title: 'Node', data: 'node', className: 'dt-left font-medium' },
                {
                  title: 'AMD Logs (PCIe, XGMI, GPU, CPU, NIC, Link)',
                  data: 'logs',
                  className: 'dt-left font-mono text-xs',
                  render: (data: string) => {
                    return `<pre class="whitespace-pre-wrap break-words max-w-4xl">${data}</pre>`
                  }
                },
              ]}
              data={amdLogs}
              defaultPageLength={50}
              pageLengthOptions={[50, 100, 500]}
            />
          )}
        </CardContent>
        </Card>
      </div>

      {/* System Error Logs Table */}
      <div id="system-logs-table">
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
      </div>

      {/* Userspace Error Logs Table */}
      <div id="userspace-logs-table">
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
      </div>

      {/* Instructions */}
      {(amdLogs.length > 0 || systemLogs.length > 0 || userspaceLogs.length > 0) && (
        <Card className="bg-blue-50 border-blue-200">
          <CardHeader>
            <CardTitle className="text-base">About these logs</CardTitle>
          </CardHeader>
          <CardContent className="text-sm space-y-3">
            <div>
              <p className="font-semibold mb-1">AMD Hardware & Driver Logs (kernel level):</p>
              <ul className="list-disc list-inside ml-2 space-y-1 text-xs">
                <li><span className="font-semibold">PCIe</span> - PCIe errors and warnings</li>
                <li><span className="font-semibold">XGMI</span> - AMD Infinity Fabric interconnect issues</li>
                <li><span className="font-semibold">amdgpu</span> - AMD GPU driver errors</li>
                <li><span className="font-semibold">EPYC/CPU</span> - CPU and processor errors</li>
                <li><span className="font-semibold">NICs</span> - NIC driver errors (ionic, bnxt, mellanox)</li>
                <li><span className="font-semibold">Link</span> - Network link state changes</li>
              </ul>
            </div>
            <div className="pt-2 border-t">
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
