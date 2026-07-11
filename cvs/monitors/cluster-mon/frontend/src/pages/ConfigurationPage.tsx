import { useState, useEffect } from 'react'
import { Upload, Save, CheckCircle, XCircle, Package, Download, RefreshCw, Network, Server } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { api } from '@/services/api'
import { useClusterStore } from '@/stores/clusterStore'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GroupState {
  hosts: string
  username: string
  authMethod: 'key' | 'password'
  keyFile: string
  password: string
}

interface JumpState {
  enabled: boolean
  host: string
  username: string
  authMethod: 'key' | 'password'
  keyFile: string
  password: string
}

const defaultGroup = (): GroupState => ({
  hosts: '', username: 'root', authMethod: 'key',
  keyFile: '~/.ssh/id_rsa', password: '',
})

const defaultJump = (): JumpState => ({
  enabled: false, host: '', username: 'root',
  authMethod: 'key', keyFile: '~/.ssh/id_rsa', password: '',
})

// ---------------------------------------------------------------------------
// SSH credentials form (reused by each tab)
// ---------------------------------------------------------------------------

function SSHForm({ state, onChange, idPrefix }: {
  state: GroupState
  onChange: (s: GroupState) => void
  idPrefix: string
}) {
  const [uploadMsg, setUploadMsg] = useState<{ ok: boolean; text: string } | null>(null)

  const handleKeyUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      await api.uploadSshKey(file)
      const path = `/root/.ssh/${file.name}`
      onChange({ ...state, keyFile: path })
      setUploadMsg({ ok: true, text: `Key '${file.name}' uploaded → ${path}` })
    } catch (err: any) {
      setUploadMsg({ ok: false, text: `Upload failed: ${err.message}` })
    }
    // Reset input so same file can be re-uploaded
    e.target.value = ''
  }

  return (
    <div className="space-y-4 pt-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            SSH Username
          </label>
          <input
            type="text"
            value={state.username}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
            onChange={e => onChange({ ...state, username: e.target.value })}
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Authentication
          </label>
          <div className="flex gap-4 mt-2">
            {(['key', 'password'] as const).map(m => (
              <label key={m} className="flex items-center gap-2 cursor-pointer text-sm dark:text-gray-300">
                <input
                  type="radio"
                  name={`${idPrefix}-auth`}
                  value={m}
                  checked={state.authMethod === m}
                  onChange={() => { onChange({ ...state, authMethod: m }); setUploadMsg(null) }}
                  className="w-4 h-4 text-blue-600"
                />
                {m === 'key' ? 'SSH Key' : 'Password'}
              </label>
            ))}
          </div>
        </div>
      </div>

      {state.authMethod === 'key' ? (
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Private Key <span className="text-xs text-gray-500 font-normal">(stored inside container)</span>
          </label>
          {/* Upload button */}
          <label className="flex items-center gap-2 cursor-pointer w-fit px-3 py-1.5 border border-dashed border-blue-400 dark:border-blue-600 rounded-lg text-xs text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors">
            <Upload className="h-3.5 w-3.5" />
            Upload key file
            <input type="file" className="hidden" onChange={handleKeyUpload} />
          </label>
          {uploadMsg && (
            <p className={`text-xs ${uploadMsg.ok ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              {uploadMsg.text}
            </p>
          )}
          {/* Path input */}
          <input
            type="text"
            value={state.keyFile}
            placeholder="/root/.ssh/id_rsa"
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-mono bg-white dark:bg-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
            onChange={e => onChange({ ...state, keyFile: e.target.value })}
          />
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Upload auto-fills this path. You can also edit it manually if the key is already mounted.
          </p>
        </div>
      ) : (
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Password
          </label>
          <input
            type="password"
            value={state.password}
            placeholder="Stored in memory only — re-enter after restart"
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
            onChange={e => onChange({ ...state, password: e.target.value })}
          />
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Hosts textarea with file upload
// ---------------------------------------------------------------------------

function HostsInput({ value, onChange, placeholder }: {
  value: string
  onChange: (v: string) => void
  placeholder: string
}) {
  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => onChange((ev.target?.result as string) ?? '')
    reader.readAsText(file)
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          IPs / Hostnames <span className="text-xs text-gray-500">(one per line)</span>
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer text-xs text-blue-600 hover:underline">
          <Upload className="h-3.5 w-3.5" />
          Upload .txt
          <input type="file" accept=".txt" className="hidden" onChange={handleFile} />
        </label>
      </div>
      <textarea
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        rows={6}
        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-mono bg-white dark:bg-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-blue-500 resize-none"
      />
      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
        {value.split('\n').map(s => s.trim()).filter(Boolean).length} host(s) configured
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Node Groups + Jump Host card
// ---------------------------------------------------------------------------

type TabKey = 'gpu' | 'scaleup' | 'scaleout'

const TABS: { key: TabKey; label: string; desc: string; icon: React.ElementType }[] = [
  { key: 'gpu',      label: 'GPU Nodes',            desc: 'Regular GPU nodes & compute trays',    icon: Server  },
  { key: 'scaleup',  label: 'Scale-up Switches',    desc: 'IFoE fabric switches (SONiC)',          icon: Network },
  { key: 'scaleout', label: 'Scale-out Switches',   desc: 'Inter-rack switches (SONiC)',           icon: Network },
]

function NodeGroupsCard() {
  const [activeTab, setActiveTab] = useState<TabKey>('gpu')
  const [gpu,      setGpu]      = useState<GroupState>(defaultGroup())
  const [scaleUp,  setScaleUp]  = useState<GroupState>(defaultGroup())
  const [scaleOut, setScaleOut] = useState<GroupState>(defaultGroup())
  const [jump,     setJump]     = useState<JumpState>(defaultJump())
  const [pollInterval, setPollInterval] = useState(120)
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [jumpKeyUploadMsg, setJumpKeyUploadMsg] = useState<{ ok: boolean; text: string } | null>(null)

  const handleJumpKeyUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      await api.uploadSshKey(file)
      const path = `/root/.ssh/${file.name}`
      setJump({ ...jump, keyFile: path })
      setJumpKeyUploadMsg({ ok: true, text: `Key '${file.name}' uploaded → ${path}` })
    } catch (err: any) {
      setJumpKeyUploadMsg({ ok: false, text: `Upload failed: ${err.message}` })
    }
    e.target.value = ''
  }

  // Pre-populate: try node_groups config first, fall back to existing cluster config
  useEffect(() => {
    const loadNodeGroups = async () => {
      try {
        const resp: any = await api.getNodeGroups()
        const fromGroup = (g: any): GroupState => ({
          hosts:      (g?.hosts ?? []).join('\n'),
          username:   g?.ssh?.username    ?? 'root',
          authMethod: (g?.ssh?.auth_method ?? 'key') as 'key' | 'password',
          keyFile:    g?.ssh?.key_file    ?? '~/.ssh/id_rsa',
          password:   '',
        })
        setGpu(fromGroup(resp.gpu_nodes))
        setScaleUp(fromGroup(resp.scale_up_switches))
        setScaleOut(fromGroup(resp.scale_out_switches))
        if (resp.poll_interval) setPollInterval(resp.poll_interval)

        // Jump host — use first group's jump host (shared)
        const jh = resp.gpu_nodes?.jump_host
        if (jh) {
          setJump({
            enabled:    jh.enabled    ?? false,
            host:       jh.host       ?? '',
            username:   jh.username   ?? 'root',
            authMethod: (jh.auth_method ?? 'key') as 'key' | 'password',
            keyFile:    jh.key_file   ?? '~/.ssh/id_rsa',
            password:   '',
          })
        }

        // If GPU nodes empty, pre-populate from existing cluster config
        if (!resp.gpu_nodes?.hosts?.length) {
          const current: any = await api.getCurrentConfiguration()
          if (current?.nodes?.length) {
            setGpu(prev => ({
              ...prev,
              hosts:      current.nodes.join('\n'),
              username:   current.username   ?? prev.username,
              authMethod: (current.auth_method ?? 'key') as 'key' | 'password',
              keyFile:    current.key_file   ?? prev.keyFile,
            }))
          }
        }
      } catch {
        // First run — try existing cluster config
        try {
          const current: any = await api.getCurrentConfiguration()
          if (current?.nodes?.length) {
            setGpu(prev => ({
              ...prev,
              hosts:      current.nodes.join('\n'),
              username:   current.username   ?? prev.username,
              authMethod: (current.auth_method ?? 'key') as 'key' | 'password',
              keyFile:    current.key_file   ?? prev.keyFile,
            }))
          }
          if (current?.jump_host_enabled) {
            setJump({
              enabled:    true,
              host:       current.jump_host         ?? '',
              username:   current.jump_host_username ?? 'root',
              authMethod: 'key',
              keyFile:    current.jump_host_key_file ?? '~/.ssh/id_rsa',
              password:   '',
            })
          }
        } catch { /* ignore */ }
      }
    }
    loadNodeGroups()
  }, [])

  const toGroupPayload = (g: GroupState) => ({
    hosts: g.hosts.split('\n').map((s: string) => s.trim()).filter(Boolean),
    ssh: {
      username:    g.username,
      auth_method: g.authMethod,
      key_file:    g.authMethod === 'key' ? g.keyFile : '~/.ssh/id_rsa',
      timeout:     30,
      password:    g.authMethod === 'password' ? (g.password || null) : null,
    },
    // Shared jump host applied to every group
    jump_host: {
      enabled:     jump.enabled,
      host:        jump.host,
      username:    jump.username,
      auth_method: jump.authMethod,
      key_file:    jump.authMethod === 'key' ? jump.keyFile : '~/.ssh/id_rsa',
      password:    jump.authMethod === 'password' ? (jump.password || null) : null,
    },
  })

  const handleSave = async () => {
    setSaving(true)
    setMsg(null)
    try {
      await api.saveNodeGroups({
        gpu_nodes:          toGroupPayload(gpu),
        scale_up_switches:  toGroupPayload(scaleUp),
        scale_out_switches: toGroupPayload(scaleOut),
        poll_interval:      pollInterval,
      })
      await api.reloadNodeGroups()
      setMsg({ type: 'success', text: 'Node groups saved. GPU nodes synced to cluster and registered with SSH daemon.' })
    } catch (err: any) {
      setMsg({ type: 'error', text: err.message ?? 'Save failed' })
    } finally {
      setSaving(false)
    }
  }

  const stateFor = (tab: TabKey) =>
    tab === 'gpu' ? gpu : tab === 'scaleup' ? scaleUp : scaleOut
  const setterFor = (tab: TabKey) =>
    tab === 'gpu' ? setGpu : tab === 'scaleup' ? setScaleUp : setScaleOut

  return (
    <>
      {/* ── Node Groups ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5 text-blue-600" />
            Node Groups
          </CardTitle>
          <CardDescription>
            Configure SSH access for each host group. All groups use the Go SSH daemon.
            GPU nodes are also synced to the cluster node list for GPU/NIC metrics collection.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-0">
          {msg && (
            <div className={`mb-4 p-3 rounded-lg text-sm ${
              msg.type === 'success'
                ? 'bg-green-50 border border-green-200 text-green-800 dark:bg-green-900/20 dark:border-green-700 dark:text-green-300'
                : 'bg-red-50 border border-red-200 text-red-800 dark:bg-red-900/20 dark:border-red-700 dark:text-red-300'
            }`}>{msg.text}</div>
          )}

          <div className="p-3 mb-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-lg text-xs text-amber-800 dark:text-amber-300">
            <strong>Passwords are in-memory only</strong> — re-enter after each container restart and click Save.
            Key file paths are persisted to <code>config/node_groups.yaml</code>.
          </div>

          {/* Tabs */}
          <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
            <div className="flex gap-0">
              {TABS.map(({ key, label, desc, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => setActiveTab(key)}
                  className={`flex items-start gap-2 px-5 py-3 text-sm border-b-2 -mb-px transition-colors text-left ${
                    activeTab === key
                      ? 'border-blue-600 text-blue-600 dark:text-blue-400'
                      : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-4 w-4 mt-0.5 shrink-0" />
                  <div>
                    <div className="font-medium">{label}</div>
                    <div className="text-xs text-gray-400 dark:text-gray-500 font-normal">{desc}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Tab content */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <HostsInput
              value={stateFor(activeTab).hosts}
              onChange={v => setterFor(activeTab)({ ...stateFor(activeTab), hosts: v })}
              placeholder={
                activeTab === 'gpu'
                  ? '192.168.1.101\n192.168.1.102\ncompute-tray-01.rack.local'
                  : activeTab === 'scaleup'
                  ? '192.168.2.1\n192.168.2.2'
                  : '192.168.3.1'
              }
            />
            <SSHForm
              state={stateFor(activeTab)}
              onChange={setterFor(activeTab)}
              idPrefix={activeTab}
            />
          </div>
        </CardContent>
      </Card>

      {/* ── Jump Host ── */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Jump Host / Bastion</CardTitle>
          <CardDescription>
            Optional. When enabled, all three node groups connect through this jump host.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={jump.enabled}
              onChange={e => setJump({ ...jump, enabled: e.target.checked })}
              className="w-4 h-4 text-blue-600 rounded"
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Use Jump Host for all groups
            </span>
          </label>

          {jump.enabled && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Jump Host IP / Hostname
                </label>
                <input
                  type="text"
                  value={jump.host}
                  placeholder="bastion.example.com"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
                  onChange={e => setJump({ ...jump, host: e.target.value })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Username
                </label>
                <input
                  type="text"
                  value={jump.username}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
                  onChange={e => setJump({ ...jump, username: e.target.value })}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Auth Method
                </label>
                <div className="flex gap-4 mt-2">
                  {(['key', 'password'] as const).map(m => (
                    <label key={m} className="flex items-center gap-2 cursor-pointer text-sm dark:text-gray-300">
                      <input
                        type="radio"
                        name="jump-auth"
                        value={m}
                        checked={jump.authMethod === m}
                        onChange={() => setJump({ ...jump, authMethod: m })}
                        className="w-4 h-4 text-blue-600"
                      />
                      {m === 'key' ? 'SSH Key' : 'Password'}
                    </label>
                  ))}
                </div>
              </div>
              <div>
                {jump.authMethod === 'key' ? (
                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                      Private Key <span className="text-xs text-gray-500 font-normal">(for authenticating to jump host)</span>
                    </label>
                    {/* Upload button */}
                    <label className="flex items-center gap-2 cursor-pointer w-fit px-3 py-1.5 border border-dashed border-blue-400 dark:border-blue-600 rounded-lg text-xs text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors">
                      <Upload className="h-3.5 w-3.5" />
                      Upload key file
                      <input type="file" className="hidden" onChange={handleJumpKeyUpload} />
                    </label>
                    {jumpKeyUploadMsg && (
                      <p className={`text-xs ${jumpKeyUploadMsg.ok ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {jumpKeyUploadMsg.text}
                      </p>
                    )}
                    {/* Path input */}
                    <input
                      type="text"
                      value={jump.keyFile}
                      placeholder="/root/.ssh/jumphost.pem"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm font-mono bg-white dark:bg-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
                      onChange={e => setJump({ ...jump, keyFile: e.target.value })}
                    />
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Upload auto-fills this path. The key is stored inside the container.
                    </p>
                  </div>
                ) : (
                  <>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Password
                    </label>
                    <input
                      type="password"
                      value={jump.password}
                      placeholder="In-memory only — re-enter after restart"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-sm bg-white dark:bg-gray-800 dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
                      onChange={e => setJump({ ...jump, password: e.target.value })}
                    />
                  </>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Single Save & Apply — applies everything above */}
      <div className="flex items-center justify-between py-2">
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Poll Interval (seconds)
          </label>
          <input
            type="number"
            min={10}
            max={3600}
            value={pollInterval}
            className="w-24 px-3 py-1.5 border border-gray-300 dark:border-gray-600 rounded text-sm bg-white dark:bg-gray-800 dark:text-gray-200 focus:ring-1 focus:ring-blue-500"
            onChange={e => setPollInterval(Number(e.target.value))}
          />
        </div>
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium"
        >
          <Save className="h-4 w-4" />
          {saving ? 'Saving...' : 'Save & Apply'}
        </button>
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// Package Installs card (unchanged logic, kept as a self-contained component)
// ---------------------------------------------------------------------------

function PackageInstallsCard() {
  const nodes = useClusterStore((state) => state.nodes)
  const [packages, setPackages] = useState<any[]>([])
  const [packageStatuses, setPackageStatuses] = useState<{ [key: string]: any }>({})
  const [installingPackage, setInstallingPackage] = useState<string | null>(null)
  const [packageMessage, setPackageMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null)

  const loadPackages = async () => {
    try {
      const result = await api.getPackageList()
      setPackages(result.packages || [])
    } catch { /* ignore */ }
  }

  const checkPackageStatus = async (packageId: string) => {
    try {
      const status = await api.getPackageStatus(packageId)
      setPackageStatuses(prev => ({ ...prev, [packageId]: status }))
    } catch { /* ignore */ }
  }

  const refreshAll = async () => {
    for (const pkg of packages) await checkPackageStatus(pkg.id)
  }

  const handleInstall = async (packageId: string) => {
    setInstallingPackage(packageId)
    setPackageMessage({ type: 'info', text: `Installing ${packageId} on all nodes...` })
    try {
      const result = await api.installPackage(packageId)
      setPackageMessage({
        type: result.success ? 'success' : 'error',
        text: result.message || `Done: ${result.successful} ok, ${result.failed} failed`,
      })
      await checkPackageStatus(packageId)
    } catch (err: any) {
      setPackageMessage({ type: 'error', text: `Failed: ${err.message}` })
    } finally {
      setInstallingPackage(null)
    }
  }

  useEffect(() => { loadPackages() }, [])
  useEffect(() => { if (packages.length) refreshAll() }, [packages])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Package className="h-5 w-5" />
          Package Installs
        </CardTitle>
        <CardDescription>Install required packages on all GPU nodes</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {packageMessage && (
          <div className={`p-3 rounded-lg text-sm ${
            packageMessage.type === 'success' ? 'bg-green-50 text-green-800 border border-green-200' :
            packageMessage.type === 'error'   ? 'bg-red-50 text-red-800 border border-red-200' :
                                                'bg-blue-50 text-blue-800 border border-blue-200'
          }`}>{packageMessage.text}</div>
        )}

        <div className="flex justify-end">
          <button onClick={refreshAll} disabled={packages.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-sm font-medium disabled:opacity-50">
            <RefreshCw className="h-4 w-4" />
            Refresh Status
          </button>
        </div>

        {packages.length === 0 ? (
          <div className="text-center py-8 text-gray-500 dark:text-gray-400">Loading packages...</div>
        ) : (
          <div className="space-y-3">
            {packages.map(pkg => {
              const status = packageStatuses[pkg.id]
              const isInstalling = installingPackage === pkg.id
              return (
                <div key={pkg.id} className="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800">
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <Package className="h-5 w-5 text-blue-600" />
                      <div>
                        <div className="font-medium dark:text-gray-200">{pkg.name}</div>
                        <div className="text-sm text-gray-600 dark:text-gray-400">{pkg.description}</div>
                        <div className="text-xs text-gray-500 mt-1 font-mono">{pkg.package_name}</div>
                      </div>
                    </div>
                    {status && (
                      <div className="mt-3 ml-8 text-sm flex items-center gap-2">
                        {status.installed_count === status.total_nodes ? (
                          <><CheckCircle className="h-4 w-4 text-green-600" /><span className="text-green-700 dark:text-green-400">Installed on all {status.total_nodes} nodes</span></>
                        ) : status.installed_count > 0 ? (
                          <><XCircle className="h-4 w-4 text-yellow-600" /><span className="text-yellow-700 dark:text-yellow-400">Installed on {status.installed_count}/{status.total_nodes} nodes</span></>
                        ) : (
                          <><XCircle className="h-4 w-4 text-red-600" /><span className="text-red-700 dark:text-red-400">Not installed</span></>
                        )}
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => handleInstall(pkg.id)}
                    disabled={isInstalling || nodes.length === 0}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm ${
                      isInstalling ? 'bg-gray-300 dark:bg-gray-700 text-gray-600 dark:text-gray-400 cursor-not-allowed' :
                      status?.installed_count === status?.total_nodes ? 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 hover:bg-green-200' :
                      'bg-blue-600 text-white hover:bg-blue-700'
                    }`}
                  >
                    {isInstalling ? (
                      <><RefreshCw className="h-4 w-4 animate-spin" />Installing...</>
                    ) : status?.installed_count === status?.total_nodes ? (
                      <><CheckCircle className="h-4 w-4" />Reinstall</>
                    ) : (
                      <><Download className="h-4 w-4" />Install</>
                    )}
                  </button>
                </div>
              )
            })}
          </div>
        )}

        {nodes.length === 0 && (
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded-lg p-3 text-sm text-yellow-800 dark:text-yellow-300">
            No GPU nodes configured. Save your node groups above to enable package installs.
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function ConfigurationPage() {
  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Configuration</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Configure SSH access for GPU nodes and switches. Compute trays are auto-detected from GPU nodes via afmctl.
        </p>
      </div>

      <NodeGroupsCard />
      <PackageInstallsCard />
    </div>
  )
}
