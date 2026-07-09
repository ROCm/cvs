import { useState, useEffect } from 'react'
import { Save, Upload, Server, Network } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import { api } from '@/services/api'

interface SSHFormState {
  username: string
  auth_method: 'key' | 'password'
  key_file: string
  password: string
}

const defaultSSH = (): SSHFormState => ({
  username: 'admin',
  auth_method: 'key',
  key_file: '~/.ssh/id_rsa',
  password: '',
})

export function RackConfigPage() {
  const [computeTrays, setComputeTrays] = useState('')
  const [switchTrays, setSwitchTrays] = useState('')
  const [computeSSH, setComputeSSH] = useState<SSHFormState>(defaultSSH())
  const [switchSSH, setSwitchSSH] = useState<SSHFormState>(defaultSSH())
  const [pollInterval, setPollInterval] = useState(120)
  const [isSaving, setIsSaving] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  // Load existing config on mount
  useEffect(() => {
    api.getRackConfig().then((resp: any) => {
      const s = resp.settings
      if (!s) return
      setComputeTrays((s.compute_trays || []).join('\n'))
      setSwitchTrays((s.switch_trays || []).join('\n'))
      setPollInterval(s.poll_interval ?? 120)
      if (s.compute_ssh) {
        setComputeSSH({
          username: s.compute_ssh.username ?? 'admin',
          auth_method: s.compute_ssh.auth_method ?? 'key',
          key_file: s.compute_ssh.key_file ?? '~/.ssh/id_rsa',
          password: '',
        })
      }
      if (s.switch_ssh) {
        setSwitchSSH({
          username: s.switch_ssh.username ?? 'admin',
          auth_method: s.switch_ssh.auth_method ?? 'key',
          key_file: s.switch_ssh.key_file ?? '~/.ssh/id_rsa',
          password: '',
        })
      }
    }).catch(() => {/* first run — no config yet */})
  }, [])

  const handleFileUpload = (
    e: React.ChangeEvent<HTMLInputElement>,
    setter: (v: string) => void
  ) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => setter((ev.target?.result as string) ?? '')
    reader.readAsText(file)
  }

  const handleSave = async () => {
    setIsSaving(true)
    setMessage(null)
    try {
      const body = {
        compute_trays: computeTrays.split('\n').map(s => s.trim()).filter(Boolean),
        switch_trays: switchTrays.split('\n').map(s => s.trim()).filter(Boolean),
        compute_ssh: {
          username: computeSSH.username,
          auth_method: computeSSH.auth_method,
          key_file: computeSSH.auth_method === 'key' ? computeSSH.key_file : null,
          timeout: 30,
        },
        switch_ssh: {
          username: switchSSH.username,
          auth_method: switchSSH.auth_method,
          key_file: switchSSH.auth_method === 'key' ? switchSSH.key_file : null,
          timeout: 30,
        },
        poll_interval: pollInterval,
        compute_password: computeSSH.auth_method === 'password' ? computeSSH.password : null,
        switch_password: switchSSH.auth_method === 'password' ? switchSSH.password : null,
      }
      await api.updateRackConfig(body)
      await api.reloadRackConfig()
      setMessage({ type: 'success', text: 'Rack configuration saved and collector restarted.' })
    } catch (err: any) {
      setMessage({ type: 'error', text: err.message ?? 'Save failed' })
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Rack Configuration</h1>
        <p className="text-gray-500 mt-1">
          Configure compute tray and switch tray IPs with SSH credentials for IFoE monitoring.
        </p>
      </div>

      {message && (
        <div
          className={`p-4 rounded-lg ${
            message.type === 'success'
              ? 'bg-green-50 border border-green-200 text-green-800'
              : 'bg-red-50 border border-red-200 text-red-800'
          }`}
        >
          {message.text}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Compute Trays */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5 text-blue-600" />
              Compute Trays
            </CardTitle>
            <CardDescription>
              Up to 18 compute trays with IFoE stations. One IP/hostname per line.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Tray IPs / Hostnames
              </label>
              <textarea
                className="w-full h-36 px-3 py-2 border border-gray-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="192.168.1.101&#10;192.168.1.102&#10;..."
                value={computeTrays}
                onChange={(e) => setComputeTrays(e.target.value)}
              />
              <label className="mt-2 flex items-center gap-2 cursor-pointer text-sm text-blue-600 hover:underline">
                <Upload className="h-4 w-4" />
                Upload .txt list
                <input
                  type="file"
                  accept=".txt"
                  className="hidden"
                  onChange={(e) => handleFileUpload(e, setComputeTrays)}
                />
              </label>
            </div>
            <SSHCredentialForm
              label="Compute SSH Credentials"
              state={computeSSH}
              onChange={setComputeSSH}
            />
          </CardContent>
        </Card>

        {/* Switch Trays */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Network className="h-5 w-5 text-purple-600" />
              Switch Trays (SONiC)
            </CardTitle>
            <CardDescription>
              Up to 6 switch trays with dual ASICs. One IP/hostname per line.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Tray IPs / Hostnames
              </label>
              <textarea
                className="w-full h-36 px-3 py-2 border border-gray-300 rounded-lg text-sm font-mono focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                placeholder="192.168.2.1&#10;192.168.2.2&#10;..."
                value={switchTrays}
                onChange={(e) => setSwitchTrays(e.target.value)}
              />
              <label className="mt-2 flex items-center gap-2 cursor-pointer text-sm text-blue-600 hover:underline">
                <Upload className="h-4 w-4" />
                Upload .txt list
                <input
                  type="file"
                  accept=".txt"
                  className="hidden"
                  onChange={(e) => handleFileUpload(e, setSwitchTrays)}
                />
              </label>
            </div>
            <SSHCredentialForm
              label="Switch SSH Credentials"
              state={switchSSH}
              onChange={setSwitchSSH}
            />
          </CardContent>
        </Card>
      </div>

      {/* Poll Interval */}
      <Card>
        <CardHeader>
          <CardTitle>Collection Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium text-gray-700 w-40">
              Poll Interval (seconds)
            </label>
            <input
              type="number"
              min={10}
              max={3600}
              className="w-32 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
              value={pollInterval}
              onChange={(e) => setPollInterval(Number(e.target.value))}
            />
            <span className="text-sm text-gray-500">
              Data is collected every {pollInterval}s. Default: 120s.
            </span>
          </div>
        </CardContent>
      </Card>

      {/* Save */}
      <div className="flex justify-end">
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="flex items-center gap-2 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 font-medium"
        >
          <Save className="h-4 w-4" />
          {isSaving ? 'Saving...' : 'Save & Reload Collector'}
        </button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-component: SSH credential form
// ---------------------------------------------------------------------------

function SSHCredentialForm({
  label,
  state,
  onChange,
}: {
  label: string
  state: SSHFormState
  onChange: (s: SSHFormState) => void
}) {
  return (
    <div className="border-t border-gray-100 pt-4 space-y-3">
      <p className="text-sm font-semibold text-gray-700">{label}</p>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-gray-600 mb-1">Username</label>
          <input
            type="text"
            className="w-full px-3 py-1.5 border border-gray-300 rounded text-sm focus:ring-1 focus:ring-blue-500"
            value={state.username}
            onChange={(e) => onChange({ ...state, username: e.target.value })}
          />
        </div>
        <div>
          <label className="block text-xs text-gray-600 mb-1">Auth Method</label>
          <div className="flex gap-3 mt-1">
            {(['key', 'password'] as const).map((m) => (
              <label key={m} className="flex items-center gap-1.5 text-sm cursor-pointer">
                <input
                  type="radio"
                  name={`auth-${label}`}
                  value={m}
                  checked={state.auth_method === m}
                  onChange={() => onChange({ ...state, auth_method: m })}
                />
                {m === 'key' ? 'SSH Key' : 'Password'}
              </label>
            ))}
          </div>
        </div>
      </div>

      {state.auth_method === 'key' ? (
        <div>
          <label className="block text-xs text-gray-600 mb-1">Key File Path</label>
          <input
            type="text"
            className="w-full px-3 py-1.5 border border-gray-300 rounded text-sm font-mono focus:ring-1 focus:ring-blue-500"
            value={state.key_file}
            onChange={(e) => onChange({ ...state, key_file: e.target.value })}
          />
        </div>
      ) : (
        <div>
          <label className="block text-xs text-gray-600 mb-1">Password</label>
          <input
            type="password"
            className="w-full px-3 py-1.5 border border-gray-300 rounded text-sm focus:ring-1 focus:ring-blue-500"
            value={state.password}
            onChange={(e) => onChange({ ...state, password: e.target.value })}
            placeholder="Stored in memory only — not saved to disk"
          />
        </div>
      )}
    </div>
  )
}
