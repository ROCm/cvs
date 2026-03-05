import { useState, useEffect } from 'react'
import { Upload, Save, Server, CheckCircle, XCircle, Package, Download, RefreshCw, Clock } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { api } from '@/services/api'
import { useClusterStore } from '@/stores/clusterStore'

export function ConfigurationPage() {
  const nodes = useClusterStore((state) => state.nodes)

  const [nodeIps, setNodeIps] = useState('')
  const [username, setUsername] = useState('')
  const [authMethod, setAuthMethod] = useState<'key' | 'password'>('key')
  const [keyFilePath, setKeyFilePath] = useState('~/.ssh/id_rsa')
  const [password, setPassword] = useState('')

  // Jump host configuration
  const [useJumpHost, setUseJumpHost] = useState(false)
  const [jumpHost, setJumpHost] = useState('')
  const [jumpUsername, setJumpUsername] = useState('')
  const [jumpAuthMethod, setJumpAuthMethod] = useState<'key' | 'password'>('key')
  const [jumpKeyFilePath, setJumpKeyFilePath] = useState('~/.ssh/id_rsa')
  const [jumpPassword, setJumpPassword] = useState('')

  // Node access from jump host
  const [nodeUsernameViaJump, setNodeUsernameViaJump] = useState('')
  const [nodeKeyFileOnJumpHost, setNodeKeyFileOnJumpHost] = useState('~/.ssh/id_rsa')

  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [isSaving, setIsSaving] = useState(false)

  // Package management state
  const [packages, setPackages] = useState<any[]>([])
  const [packageStatuses, setPackageStatuses] = useState<{ [key: string]: any }>({})
  const [installingPackage, setInstallingPackage] = useState<string | null>(null)
  const [packageMessage, setPackageMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        const content = event.target?.result as string
        setNodeIps(content)
      }
      reader.readAsText(file)
    }
  }

  // Load available packages
  const loadPackages = async () => {
    try {
      const result = await api.getPackageList()
      setPackages(result.packages || [])
    } catch (error) {
      console.error('Failed to load packages:', error)
    }
  }

  // Check package status
  const checkPackageStatus = async (packageId: string) => {
    try {
      const status = await api.getPackageStatus(packageId)
      setPackageStatuses(prev => ({ ...prev, [packageId]: status }))
    } catch (error) {
      console.error(`Failed to check status for ${packageId}:`, error)
    }
  }

  // Install package
  const handleInstallPackage = async (packageId: string) => {
    try {
      setInstallingPackage(packageId)
      setPackageMessage({ type: 'info', text: `Installing ${packageId} on all nodes... This may take a few minutes.` })

      const result = await api.installPackage(packageId)

      setPackageMessage({
        type: result.success ? 'success' : 'error',
        text: result.message || `Installation completed: ${result.successful} successful, ${result.failed} failed`
      })

      // Refresh package status after installation
      await checkPackageStatus(packageId)

    } catch (error: any) {
      console.error(`Failed to install ${packageId}:`, error)
      setPackageMessage({
        type: 'error',
        text: `Failed to install ${packageId}: ${error.message || 'Unknown error'}`
      })
    } finally {
      setInstallingPackage(null)
    }
  }

  // Refresh all package statuses
  const refreshPackageStatuses = async () => {
    for (const pkg of packages) {
      await checkPackageStatus(pkg.id)
    }
  }


  const handleSaveConfiguration = async () => {
    try {
      setIsSaving(true)

      // Parse node IPs
      const nodes = nodeIps
        .split('\n')
        .map((line) => line.trim())
        .filter((line) => line && !line.startsWith('#'))

      if (nodes.length === 0) {
        setMessage({ type: 'error', text: 'Please enter at least one node IP' })
        setIsSaving(false)
        return
      }

      // Validate username
      const effectiveUsername = useJumpHost ? nodeUsernameViaJump : username
      if (!effectiveUsername || effectiveUsername.trim() === '') {
        setMessage({ type: 'error', text: 'Please enter SSH username' })
        setIsSaving(false)
        return
      }

      // Prepare configuration
      const config = {
        nodes,
        username: effectiveUsername,
        auth_method: authMethod,
        key_file_path: authMethod === 'key' ? (useJumpHost ? nodeKeyFileOnJumpHost : keyFilePath) : undefined,
        password: authMethod === 'password' ? password : undefined,
        use_jump_host: useJumpHost,
        jump_host: useJumpHost ? {
          host: jumpHost,
          username: jumpUsername,
          auth_method: jumpAuthMethod,
          key_file_path: jumpAuthMethod === 'key' ? jumpKeyFilePath : undefined,
          password: jumpAuthMethod === 'password' ? jumpPassword : undefined,
          node_username: nodeUsernameViaJump,
          node_key_file: nodeKeyFileOnJumpHost,
        } : null,  // Send null instead of undefined to ensure backend disables it
      }

      // Validate jump host if enabled
      if (useJumpHost && !jumpHost) {
        setMessage({ type: 'error', text: 'Please enter jump host IP/hostname' })
        setIsSaving(false)
        return
      }

      // Send to backend API
      setMessage({ type: 'success', text: '💾 Saving configuration...' })

      const result = await api.updateConfiguration(config)

      setMessage({
        type: 'success',
        text: `✅ ${result.message}\n\n🔄 Reloading backend configuration...`,
      })

      // Reload backend configuration immediately
      try {
        const reloadResult = await api.reloadConfiguration()

        // Check if key upload is required
        if (reloadResult.requires_key_upload) {
          setMessage({
            type: 'error',
            text: `⚠️ Configuration saved but SSH keys missing!\n\n${reloadResult.error}\n\nPlease upload your SSH private key using the file upload above and try saving again.`,
          })
          setIsSaving(false)
          return
        }

        if (!reloadResult.success) {
          setMessage({
            type: 'error',
            text: `⚠️ Configuration saved but reload failed: ${reloadResult.error}`,
          })
          setIsSaving(false)
          return
        }

        // Reload the current configuration from backend
        const currentConfig = await api.getCurrentConfiguration()

        // Update all form fields with saved values
        if (currentConfig.nodes && currentConfig.nodes.length > 0) {
          setNodeIps(currentConfig.nodes.join('\n'))
        }
        if (currentConfig.username) {
          setUsername(currentConfig.username)
        }
        if (currentConfig.auth_method) {
          setAuthMethod(currentConfig.auth_method as 'key' | 'password')
        }
        if (currentConfig.key_file) {
          setKeyFilePath(currentConfig.key_file)
        }

        // Update jump host settings
        setUseJumpHost(currentConfig.jump_host_enabled || false)

        if (currentConfig.jump_host_enabled) {
          if (currentConfig.jump_host) {
            setJumpHost(currentConfig.jump_host)
          }
          if (currentConfig.jump_host_username) {
            setJumpUsername(currentConfig.jump_host_username)
          }
          if (currentConfig.jump_host_key_file) {
            setJumpKeyFilePath(currentConfig.jump_host_key_file)
          }
          if (currentConfig.node_username_via_jump) {
            setNodeUsernameViaJump(currentConfig.node_username_via_jump)
          }
          if (currentConfig.node_key_file_on_jumphost) {
            setNodeKeyFileOnJumpHost(currentConfig.node_key_file_on_jumphost)
          }
        }

        setMessage({
          type: 'success',
          text: `✅ Configuration saved and applied successfully!\n\nMonitoring ${currentConfig.nodes?.length || 0} nodes.`,
        })
      } catch (err) {
        console.error('Failed to reload configuration:', err)
        setMessage({
          type: 'error',
          text: `❌ Failed to reload configuration: ${err instanceof Error ? err.message : 'Unknown error'}`,
        })
      } finally {
        setIsSaving(false)
      }

    } catch (error: any) {
      console.error('Failed to save configuration:', error)
      setMessage({
        type: 'error',
        text: `❌ Failed to save configuration: ${error.message || 'Unknown error'}`,
      })
      setIsSaving(false)
    }
  }

  // Load current configuration and packages on mount
  useEffect(() => {
    const loadCurrentConfig = async () => {
      try {
        const currentConfig = await api.getCurrentConfiguration()
        if (currentConfig.nodes && currentConfig.nodes.length > 0) {
          setNodeIps(currentConfig.nodes.join('\n'))
        }
        if (currentConfig.username) {
          setUsername(currentConfig.username)
        }
        if (currentConfig.jump_host_enabled) {
          setUseJumpHost(true)
          if (currentConfig.jump_host) {
            setJumpHost(currentConfig.jump_host)
          }
          if (currentConfig.jump_host_username) {
            setJumpUsername(currentConfig.jump_host_username)
          }
        }
      } catch (error) {
        console.error('Failed to load current configuration:', error)
      }
    }

    loadCurrentConfig()
    loadPackages()
  }, [])

  // Load package statuses when packages are loaded
  useEffect(() => {
    if (packages.length > 0) {
      refreshPackageStatuses()
    }
  }, [packages])

  return (
    <div className="max-w-4xl">
      <h1 className="text-3xl font-bold mb-6">Cluster Configuration</h1>

      <div className="space-y-6">
        {/* Message */}
        {message && (
          <div
            className={`p-4 rounded-lg ${
              message.type === 'success'
                ? 'bg-green-50 text-green-800 border border-green-200'
                : 'bg-red-50 text-red-800 border border-red-200'
            }`}
          >
            {message.text}
          </div>
        )}

        {/* Node Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              Cluster Nodes
            </CardTitle>
            <CardDescription>
              Enter node IP addresses or hostnames (one per line)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium mb-2">Upload Nodes File</label>
              <div className="flex items-center gap-4">
                <input
                  type="file"
                  accept=".txt"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 transition-colors"
                >
                  <Upload className="h-4 w-4" />
                  Upload nodes.txt
                </label>
                <span className="text-sm text-gray-500">
                  Or enter IPs manually below
                </span>
              </div>
            </div>

            {/* Manual Input */}
            <div>
              <label className="block text-sm font-medium mb-2">Node IPs / Hostnames</label>
              <textarea
                value={nodeIps}
                onChange={(e) => setNodeIps(e.target.value)}
                placeholder="192.168.1.10&#10;192.168.1.11&#10;192.168.1.12&#10;node1.cluster.local"
                rows={8}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
              />
              <p className="text-xs text-gray-500 mt-1">
                Enter one IP address or hostname per line. Lines starting with # are ignored.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Current Loaded Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              Current Configuration (Loaded from files)
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-semibold">Total Nodes:</span>{' '}
                <span className="font-mono">{nodes.length}</span>
              </div>
              <div>
                <span className="font-semibold">SSH Username:</span>{' '}
                <span className="font-mono">{username}</span>
              </div>
              <div>
                <span className="font-semibold">Auth Method:</span>{' '}
                <span className="font-mono">{authMethod === 'key' ? 'SSH Key' : 'Password'}</span>
              </div>
              {authMethod === 'key' && (
                <div>
                  <span className="font-semibold">Key File:</span>{' '}
                  <span className="font-mono">{keyFilePath}</span>
                </div>
              )}
              <div>
                <span className="font-semibold">Jump Host:</span>{' '}
                <span className="font-mono">{useJumpHost ? `Enabled (${jumpHost})` : 'Disabled (Direct SSH)'}</span>
              </div>
            </div>

            {nodes.length > 0 && (
              <div className="mt-4">
                <div className="font-semibold mb-2">Configured Nodes:</div>
                <div className="bg-gray-50 p-3 rounded max-h-40 overflow-y-auto font-mono text-xs">
                  {nodes.slice(0, 20).map((node, idx) => (
                    <div key={idx} className="py-1">{node.hostname}</div>
                  ))}
                  {nodes.length > 20 && (
                    <div className="text-gray-500 italic mt-2">
                      ... and {nodes.length - 20} more nodes
                    </div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Current Nodes Status */}
        {nodes.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Current Nodes Being Monitored</CardTitle>
              <CardDescription>
                {nodes.length} node(s) configured - SSH reachability status
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {(() => {
                  // Deduplicate nodes by hostname
                  const uniqueNodes = nodes.reduce((acc: any[], curr) => {
                    if (!acc.find(n => n.hostname === curr.hostname)) {
                      acc.push(curr)
                    }
                    return acc
                  }, [])

                  return uniqueNodes.map((node) => (
                    <div
                      key={node.hostname}
                      className="flex items-center justify-between p-3 border border-gray-200 rounded-lg hover:bg-gray-50"
                    >
                      <div className="flex items-center gap-3">
                        {node.status === 'unreachable' ? (
                          <XCircle className="h-5 w-5 text-red-500" />
                        ) : (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        )}
                        <div>
                          <div className="font-medium">{node.hostname}</div>
                          <div className="text-xs text-gray-500">
                            {node.status === 'unreachable' ? (
                              <span className="text-red-600">SSH Connection Failed</span>
                            ) : (
                              <span className="text-green-600">
                                SSH Connected • {node.gpu_count} GPUs
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="text-sm">
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-medium ${
                            node.status === 'unreachable'
                              ? 'bg-red-100 text-red-800'
                              : node.status === 'unhealthy'
                                ? 'bg-yellow-100 text-yellow-800'
                                : 'bg-green-100 text-green-800'
                          }`}
                        >
                          {node.status === 'unreachable'
                            ? 'Unreachable'
                            : node.status === 'unhealthy'
                              ? 'Unhealthy'
                              : 'Healthy'}
                        </span>
                      </div>
                    </div>
                  ))
                })()}
              </div>
            </CardContent>
          </Card>
        )}

        {/* SSH Authentication */}
        <Card>
          <CardHeader>
            <CardTitle>SSH Authentication</CardTitle>
            <CardDescription>Configure SSH access to cluster nodes</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Username */}
            <div>
              <label className="block text-sm font-medium mb-2">SSH Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="root"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            {/* Auth Method */}
            <div>
              <label className="block text-sm font-medium mb-2">Authentication Method</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    value="key"
                    checked={authMethod === 'key'}
                    onChange={() => setAuthMethod('key')}
                    className="w-4 h-4 text-blue-600"
                  />
                  <span>SSH Key</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    value="password"
                    checked={authMethod === 'password'}
                    onChange={() => setAuthMethod('password')}
                    className="w-4 h-4 text-blue-600"
                  />
                  <span>Password</span>
                </label>
              </div>
            </div>

            {/* SSH Key Upload and Path */}
            {authMethod === 'key' && (
              <>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Upload SSH Private Key
                  </label>
                  <input
                    type="file"
                    onChange={async (e) => {
                      const file = e.target.files?.[0]
                      if (file) {
                        try {
                          await api.uploadSshKey(file)
                          setMessage({ type: 'success', text: `SSH key '${file.name}' uploaded successfully` })
                          // Update the key file path to match uploaded file
                          setKeyFilePath(`/root/.ssh/${file.name}`)
                        } catch (error: any) {
                          setMessage({ type: 'error', text: `Failed to upload SSH key: ${error.message}` })
                        }
                      }
                    }}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Upload your SSH private key, or use mounted host paths below (see mounted paths)
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    SSH Private Key Path (in container)
                  </label>
                  <input
                    type="text"
                    value={keyFilePath}
                    onChange={(e) => setKeyFilePath(e.target.value)}
                    placeholder="/root/.ssh/id_rsa"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Upload SSH key to /root/.ssh/* or use path after manual volume mount
                  </p>
                </div>
              </>
            )}

            {/* Password */}
            {authMethod === 'password' && (
              <div>
                <label className="block text-sm font-medium mb-2">SSH Password</label>
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter SSH password"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <p className="text-xs text-yellow-600 mt-1">
                  ⚠️ Storing passwords is less secure. SSH keys are recommended.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Jump Host Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Jump Host / Bastion Configuration</CardTitle>
            <CardDescription>
              Configure jump host for nodes not directly accessible
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Enable Jump Host */}
            <div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useJumpHost}
                  onChange={(e) => setUseJumpHost(e.target.checked)}
                  className="w-4 h-4 text-blue-600 rounded"
                />
                <span className="font-medium">Use Jump Host (Bastion)</span>
              </label>
              <p className="text-xs text-gray-500 mt-1 ml-6">
                Enable this if cluster nodes are only accessible through a jump/bastion host
              </p>
            </div>

            {useJumpHost && (
              <>
                {/* Jump Host Address */}
                <div>
                  <label className="block text-sm font-medium mb-2">Jump Host IP / Hostname</label>
                  <input
                    type="text"
                    value={jumpHost}
                    onChange={(e) => setJumpHost(e.target.value)}
                    placeholder="bastion.example.com or 192.168.1.100"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                {/* Jump Host Username */}
                <div>
                  <label className="block text-sm font-medium mb-2">Jump Host Username</label>
                  <input
                    type="text"
                    value={jumpUsername}
                    onChange={(e) => setJumpUsername(e.target.value)}
                    placeholder="your_username"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                {/* Jump Host Auth Method */}
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Jump Host Authentication
                  </label>
                  <div className="flex gap-4">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="key"
                        checked={jumpAuthMethod === 'key'}
                        onChange={() => setJumpAuthMethod('key')}
                        className="w-4 h-4 text-blue-600"
                      />
                      <span>SSH Key</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value="password"
                        checked={jumpAuthMethod === 'password'}
                        onChange={() => setJumpAuthMethod('password')}
                        className="w-4 h-4 text-blue-600"
                      />
                      <span>Password</span>
                    </label>
                  </div>
                </div>

                {/* Jump Host SSH Key Upload and Path */}
                {jumpAuthMethod === 'key' && (
                  <>
                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Upload Jump Host SSH Private Key
                      </label>
                      <input
                        type="file"
                        onChange={async (e) => {
                          const file = e.target.files?.[0]
                          if (file) {
                            try {
                              await api.uploadSshKey(file)
                              setMessage({ type: 'success', text: `Jump host key '${file.name}' uploaded successfully` })
                              setJumpKeyFilePath(`/root/.ssh/${file.name}`)
                            } catch (error: any) {
                              setMessage({ type: 'error', text: `Failed to upload jump host key: ${error.message}` })
                            }
                          }
                        }}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Upload SSH key to access the jump host from container
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-2">
                        Jump Host SSH Private Key Path (in container)
                      </label>
                      <input
                        type="text"
                        value={jumpKeyFilePath}
                        onChange={(e) => setJumpKeyFilePath(e.target.value)}
                        placeholder="/root/.ssh/id_rsa"
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Path will be set automatically after upload
                      </p>
                    </div>
                  </>
                )}

                {/* Jump Host Password */}
                {jumpAuthMethod === 'password' && (
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Jump Host Password
                    </label>
                    <input
                      type="password"
                      value={jumpPassword}
                      onChange={(e) => setJumpPassword(e.target.value)}
                      placeholder="Enter jump host password"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                )}

                {/* Divider */}
                <div className="border-t border-gray-300 my-4"></div>

                {/* Node Access from Jump Host */}
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-4">
                  <p className="text-sm font-semibold text-yellow-900 mb-2">
                    Node Access Configuration (FROM Jump Host)
                  </p>
                  <p className="text-xs text-yellow-800">
                    Specify how to SSH from the jump host to cluster nodes
                  </p>
                </div>

                {/* Node Username */}
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Node Username (for cluster nodes)
                  </label>
                  <input
                    type="text"
                    value={nodeUsernameViaJump}
                    onChange={(e) => setNodeUsernameViaJump(e.target.value)}
                    placeholder="your_username"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Username to use when SSHing from jump host to cluster nodes
                  </p>
                </div>

                {/* Node Key File Path on Jump Host */}
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Node SSH Private Key Path (ON JUMP HOST)
                  </label>
                  <input
                    type="text"
                    value={nodeKeyFileOnJumpHost}
                    onChange={(e) => setNodeKeyFileOnJumpHost(e.target.value)}
                    placeholder="~/.ssh/id_rsa"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Path to SSH private key ON THE JUMP HOST to access cluster nodes (e.g., ~/.ssh/id_rsa or /home/user/.ssh/cluster_key)
                  </p>
                </div>

                {/* Jump Host Info */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <p className="text-xs text-blue-800">
                    <strong>Jump Host Setup:</strong> The system will SSH to the jump host first,
                    then from the jump host, SSH to cluster nodes using the keyfile specified above.
                  </p>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        {/* Package Installs */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Package className="h-5 w-5" />
              Package Installs
            </CardTitle>
            <CardDescription>
              Install required packages on all cluster nodes
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Package Message */}
            {packageMessage && (
              <div
                className={`p-4 rounded-lg ${
                  packageMessage.type === 'success'
                    ? 'bg-green-50 text-green-800 border border-green-200'
                    : packageMessage.type === 'error'
                    ? 'bg-red-50 text-red-800 border border-red-200'
                    : 'bg-blue-50 text-blue-800 border border-blue-200'
                }`}
              >
                {packageMessage.text}
              </div>
            )}

            {/* Refresh Button */}
            <div className="flex justify-end">
              <button
                onClick={refreshPackageStatuses}
                disabled={packages.length === 0}
                className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RefreshCw className="h-4 w-4" />
                Refresh Status
              </button>
            </div>

            {/* Package List */}
            {packages.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                Loading packages...
              </div>
            ) : (
              <div className="space-y-3">
                {packages.map((pkg) => {
                  const status = packageStatuses[pkg.id]
                  const isInstalling = installingPackage === pkg.id

                  return (
                    <div
                      key={pkg.id}
                      className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50"
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-3">
                          <Package className="h-5 w-5 text-blue-600" />
                          <div>
                            <div className="font-medium">{pkg.name}</div>
                            <div className="text-sm text-gray-600">{pkg.description}</div>
                            <div className="text-xs text-gray-500 mt-1">
                              Package: <span className="font-mono">{pkg.package_name}</span>
                            </div>
                          </div>
                        </div>

                        {/* Installation Status */}
                        {status && (
                          <div className="mt-3 ml-8 text-sm">
                            <div className="flex items-center gap-2">
                              {status.installed_count === status.total_nodes ? (
                                <>
                                  <CheckCircle className="h-4 w-4 text-green-600" />
                                  <span className="text-green-700 font-medium">
                                    Installed on all {status.total_nodes} nodes
                                  </span>
                                </>
                              ) : status.installed_count > 0 ? (
                                <>
                                  <XCircle className="h-4 w-4 text-yellow-600" />
                                  <span className="text-yellow-700 font-medium">
                                    Installed on {status.installed_count}/{status.total_nodes} nodes
                                  </span>
                                </>
                              ) : (
                                <>
                                  <XCircle className="h-4 w-4 text-red-600" />
                                  <span className="text-red-700 font-medium">
                                    Not installed on any nodes
                                  </span>
                                </>
                              )}
                            </div>

                            {/* Show partial installation details */}
                            {status.installed_count > 0 && status.installed_count < status.total_nodes && (
                              <div className="mt-2 text-xs text-gray-600">
                                <div>Missing on: {status.not_installed_nodes.slice(0, 5).join(', ')}
                                  {status.not_installed_nodes.length > 5 && ` and ${status.not_installed_nodes.length - 5} more`}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Install Button */}
                      <div>
                        <button
                          onClick={() => handleInstallPackage(pkg.id)}
                          disabled={isInstalling || nodes.length === 0}
                          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors font-medium ${
                            isInstalling
                              ? 'bg-gray-300 text-gray-600 cursor-not-allowed'
                              : status && status.installed_count === status.total_nodes
                              ? 'bg-green-100 text-green-700 hover:bg-green-200'
                              : 'bg-blue-600 text-white hover:bg-blue-700'
                          }`}
                        >
                          {isInstalling ? (
                            <>
                              <RefreshCw className="h-4 w-4 animate-spin" />
                              Installing...
                            </>
                          ) : status && status.installed_count === status.total_nodes ? (
                            <>
                              <CheckCircle className="h-4 w-4" />
                              Reinstall
                            </>
                          ) : (
                            <>
                              <Download className="h-4 w-4" />
                              Install
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}

            {nodes.length === 0 && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-sm text-yellow-800">
                No nodes configured. Please configure nodes above before installing packages.
              </div>
            )}
          </CardContent>
        </Card>

        {/* Save Button */}
        <div className="flex justify-end">
          <button
            onClick={handleSaveConfiguration}
            disabled={isSaving}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSaving ? (
              <>
                <Clock className="h-4 w-4 animate-pulse" />
                Saving and Starting...
              </>
            ) : (
              <>
                <Save className="h-4 w-4" />
                Save Configuration and Start Monitoring
              </>
            )}
          </button>
        </div>

        {/* Instructions */}
        <Card className="bg-blue-50 border-blue-200">
          <CardHeader>
            <CardTitle className="text-base">After Saving Configuration</CardTitle>
          </CardHeader>
          <CardContent className="text-sm space-y-2">
            <p>1. Configuration will be saved to the backend</p>
            <p>2. Restart the backend server to apply changes:</p>
            <pre className="bg-gray-900 text-gray-100 p-3 rounded mt-2 text-xs">
              cd backend{'\n'}
              kill $(cat backend.pid){'\n'}
              ./run.sh
            </pre>
            <p className="text-blue-700 font-medium mt-4">
              For SSH key authentication to work, ensure your public key is authorized on all nodes.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
