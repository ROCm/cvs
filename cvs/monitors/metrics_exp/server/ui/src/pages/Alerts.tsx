import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  Bell,
  Plus,
  Trash2,
  Mail,
  MessageSquare,
  Webhook,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Send,
  Settings,
  Zap,
  ChevronDown,
  ChevronUp,
  Server,
  Users,
  Edit2,
} from 'lucide-react'
import api from '../api'

// Types
interface ContactPoint {
  id: number
  name: string
  description?: string
  contact_type: string
  settings: Record<string, unknown>
  has_credentials: boolean
  grafana_uid?: string
  last_synced_at?: string
  sync_error?: string
  alert_rule_count: number
  created_at: string
  updated_at: string
}

interface AlertTemplate {
  id: number
  name: string
  display_name: string
  description?: string
  category: string
  is_default: boolean
  datasource_type: string
  query_expression: string
  default_threshold: Record<string, unknown>
  default_severity: string
  summary_template?: string
}

interface AlertRule {
  id: number
  name: string
  description?: string
  enabled: boolean
  severity: string
  monitoring_server_id: number
  template_id?: number
  contact_point_id?: number
  node_group_id?: number
  datasource_type: string
  query_expression?: string
  threshold_config?: Record<string, unknown>
  grafana_uid?: string
  last_synced_at?: string
  sync_error?: string
  template_name?: string
  contact_point_name?: string
  node_group_name?: string
  monitoring_server_name?: string
}

interface MonitoringServer {
  id: number
  name: string
  grafana_url?: string
}

interface NodeGroup {
  id: number
  name: string
}

// API Functions
const fetchContactPoints = () =>
  api.get<ContactPoint[]>('/alerts/contact-points').then(r => r.data)

const fetchTemplates = () =>
  api.get<AlertTemplate[]>('/alerts/templates').then(r => r.data)

const fetchAlertRules = () =>
  api.get<AlertRule[]>('/alerts/rules').then(r => r.data)

const fetchMonitoringServers = () =>
  api.get<MonitoringServer[]>('/monitoring-servers').then(r => r.data)

const fetchNodeGroups = () =>
  api.get<NodeGroup[]>('/nodegroups').then(r => r.data)

const createContactPoint = (data: {
  name: string
  description?: string
  contact_type: string
  settings: Record<string, unknown>
}) => api.post('/alerts/contact-points', data).then(r => r.data)

const updateContactPoint = (id: number, data: {
  name?: string
  description?: string
  contact_type?: string
  settings?: Record<string, unknown>
}) => api.put(`/alerts/contact-points/${id}`, data).then(r => r.data)

const deleteContactPoint = (id: number) =>
  api.delete(`/alerts/contact-points/${id}`)

const testContactPoint = (id: number) =>
  api.post(`/alerts/contact-points/${id}/test`).then(r => r.data)

const syncContactPoints = () =>
  api.post('/alerts/contact-points/sync').then(r => r.data)

const createAlertRules = (data: {
  monitoring_server_id: number
  contact_point_id?: number
  node_group_id?: number
  template_ids: number[]
}) => api.post('/alerts/rules/bulk', data).then(r => r.data)

const deleteAlertRule = (id: number) =>
  api.delete(`/alerts/rules/${id}`)

const enableAlertRule = (id: number) =>
  api.post(`/alerts/rules/${id}/enable`).then(r => r.data)

const disableAlertRule = (id: number) =>
  api.post(`/alerts/rules/${id}/disable`).then(r => r.data)

const syncAlertRules = () =>
  api.post('/alerts/rules/sync').then(r => r.data)

// Icon mapping for contact types
const contactTypeIcons: Record<string, React.ElementType> = {
  email: Mail,
  slack: MessageSquare,
  msteams: MessageSquare,
  teams: MessageSquare,
  webhook: Webhook,
  pagerduty: Bell,
  opsgenie: Bell,
  discord: MessageSquare,
}

const contactTypeLabels: Record<string, string> = {
  email: 'Email',
  slack: 'Slack',
  msteams: 'Microsoft Teams',
  teams: 'Microsoft Teams',
  webhook: 'Webhook',
  pagerduty: 'PagerDuty',
  opsgenie: 'OpsGenie',
  discord: 'Discord',
}

const severityColors: Record<string, string> = {
  info: 'bg-blue-100 text-blue-800',
  warning: 'bg-yellow-100 text-yellow-800',
  critical: 'bg-red-100 text-red-800',
}

const categoryLabels: Record<string, string> = {
  node_health: 'Node Health',
  gpu_hardware: 'GPU Hardware',
  thermal: 'Thermal',
  memory: 'Memory',
  network: 'Network & RDMA',
  storage: 'Storage & Filesystem',
  logs: 'Logs',
}

export default function Alerts() {
  const [activeTab, setActiveTab] = useState<'rules' | 'contacts'>('rules')
  const [showAddContact, setShowAddContact] = useState(false)
  const [showAddRules, setShowAddRules] = useState(false)
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['node_health', 'gpu_hardware']))

  const queryClient = useQueryClient()

  // Queries
  const { data: contactPoints = [], isLoading: loadingContacts } = useQuery({
    queryKey: ['contactPoints'],
    queryFn: fetchContactPoints,
  })

  const { data: templates = [], isLoading: loadingTemplates } = useQuery({
    queryKey: ['alertTemplates'],
    queryFn: fetchTemplates,
  })

  const { data: alertRules = [], isLoading: loadingRules } = useQuery({
    queryKey: ['alertRules'],
    queryFn: fetchAlertRules,
  })

  const { data: monitoringServers = [] } = useQuery({
    queryKey: ['monitoringServers'],
    queryFn: fetchMonitoringServers,
  })

  const { data: nodeGroups = [] } = useQuery({
    queryKey: ['nodeGroups'],
    queryFn: fetchNodeGroups,
  })

  // Mutations
  const createContactMutation = useMutation({
    mutationFn: createContactPoint,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['contactPoints'] })
      setShowAddContact(false)
    },
  })

  const updateContactMutation = useMutation({
    mutationFn: ({ id, data }: { id: number; data: Parameters<typeof updateContactPoint>[1] }) =>
      updateContactPoint(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['contactPoints'] })
    },
  })

  const deleteContactMutation = useMutation({
    mutationFn: deleteContactPoint,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['contactPoints'] })
    },
  })

  const testContactMutation = useMutation({
    mutationFn: testContactPoint,
  })

  const syncContactsMutation = useMutation({
    mutationFn: syncContactPoints,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['contactPoints'] })
    },
  })

  const createRulesMutation = useMutation({
    mutationFn: createAlertRules,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alertRules'] })
      setShowAddRules(false)
    },
  })

  const deleteRuleMutation = useMutation({
    mutationFn: deleteAlertRule,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alertRules'] })
    },
  })

  const enableRuleMutation = useMutation({
    mutationFn: enableAlertRule,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alertRules'] })
    },
  })

  const disableRuleMutation = useMutation({
    mutationFn: disableAlertRule,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alertRules'] })
    },
  })

  const syncRulesMutation = useMutation({
    mutationFn: syncAlertRules,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alertRules'] })
    },
  })

  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev)
      if (next.has(category)) {
        next.delete(category)
      } else {
        next.add(category)
      }
      return next
    })
  }

  // Group rules by category
  const rulesByCategory = alertRules.reduce((acc, rule) => {
    const template = templates.find(t => t.id === rule.template_id)
    const category = template?.category || 'other'
    if (!acc[category]) acc[category] = []
    acc[category].push(rule)
    return acc
  }, {} as Record<string, AlertRule[]>)

  const isLoading = loadingContacts || loadingTemplates || loadingRules

  return (
    <div className="p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <Bell className="h-6 w-6 text-red-600" />
            Alert Configuration
          </h1>
          <p className="text-gray-600 mt-1">
            Configure how you receive alerts for your GPU fleet
          </p>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <CheckCircle2 className="h-5 w-5 text-green-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {alertRules.filter(r => r.enabled).length}
              </div>
              <div className="text-sm text-gray-500">Active Alerts</div>
            </div>
          </div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gray-100 rounded-lg">
              <XCircle className="h-5 w-5 text-gray-400" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {alertRules.filter(r => !r.enabled).length}
              </div>
              <div className="text-sm text-gray-500">Disabled</div>
            </div>
          </div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Send className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {contactPoints.length}
              </div>
              <div className="text-sm text-gray-500">Contact Points</div>
            </div>
          </div>
        </div>
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-red-100 rounded-lg">
              <AlertTriangle className="h-5 w-5 text-red-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {alertRules.filter(r => r.sync_error).length}
              </div>
              <div className="text-sm text-gray-500">Sync Errors</div>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('rules')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'rules'
                ? 'border-red-500 text-red-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <Zap className="inline h-4 w-4 mr-2" />
            Alert Rules ({alertRules.length})
          </button>
          <button
            onClick={() => setActiveTab('contacts')}
            className={`py-4 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'contacts'
                ? 'border-red-500 text-red-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <Send className="inline h-4 w-4 mr-2" />
            Contact Points ({contactPoints.length})
          </button>
        </nav>
      </div>

      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
        </div>
      ) : activeTab === 'contacts' ? (
        <ContactPointsTab
          contactPoints={contactPoints}
          showAddContact={showAddContact}
          setShowAddContact={setShowAddContact}
          createContactMutation={createContactMutation}
          updateContactMutation={updateContactMutation}
          deleteContactMutation={deleteContactMutation}
          testContactMutation={testContactMutation}
          syncContactsMutation={syncContactsMutation}
        />
      ) : (
        <AlertRulesTab
          alertRules={alertRules}
          templates={templates}
          contactPoints={contactPoints}
          monitoringServers={monitoringServers}
          nodeGroups={nodeGroups}
          rulesByCategory={rulesByCategory}
          expandedCategories={expandedCategories}
          toggleCategory={toggleCategory}
          showAddRules={showAddRules}
          setShowAddRules={setShowAddRules}
          createRulesMutation={createRulesMutation}
          deleteRuleMutation={deleteRuleMutation}
          enableRuleMutation={enableRuleMutation}
          disableRuleMutation={disableRuleMutation}
          syncRulesMutation={syncRulesMutation}
        />
      )}
    </div>
  )
}

// Contact Points Tab Component
function ContactPointsTab({
  contactPoints,
  showAddContact,
  setShowAddContact,
  createContactMutation,
  updateContactMutation,
  deleteContactMutation,
  testContactMutation,
  syncContactsMutation,
}: {
  contactPoints: ContactPoint[]
  showAddContact: boolean
  setShowAddContact: (show: boolean) => void
  createContactMutation: ReturnType<typeof useMutation<unknown, Error, Parameters<typeof createContactPoint>[0]>>
  updateContactMutation: ReturnType<typeof useMutation<unknown, Error, { id: number; data: Parameters<typeof updateContactPoint>[1] }>>
  deleteContactMutation: ReturnType<typeof useMutation<unknown, Error, number>>
  testContactMutation: ReturnType<typeof useMutation<unknown, Error, number>>
  syncContactsMutation: ReturnType<typeof useMutation<unknown, Error, void>>
}) {
  const [contactType, setContactType] = useState('email')
  const [contactName, setContactName] = useState('')
  const [contactDescription, setContactDescription] = useState('')
  const [emailAddresses, setEmailAddresses] = useState('')
  const [webhookUrl, setWebhookUrl] = useState('')
  const [slackWebhook, setSlackWebhook] = useState('')
  const [editingContactId, setEditingContactId] = useState<number | null>(null)

  const resetForm = () => {
    setContactName('')
    setContactDescription('')
    setEmailAddresses('')
    setWebhookUrl('')
    setSlackWebhook('')
    setContactType('email')
    setEditingContactId(null)
  }

  const startEditing = (cp: ContactPoint) => {
    setEditingContactId(cp.id)
    setContactName(cp.name)
    setContactDescription(cp.description || '')
    setContactType(cp.contact_type)

    const settings = cp.settings || {}
    if (cp.contact_type === 'email') {
      // Handle both array format (old) and string format (new/Grafana)
      const addresses = settings.addresses
      let emailList = ''
      if (Array.isArray(addresses)) {
        emailList = addresses.join(', ')
      } else if (typeof addresses === 'string') {
        // Convert semicolon-separated (Grafana format) to comma-separated for display
        emailList = addresses.split(';').map((e: string) => e.trim()).filter(Boolean).join(', ')
      }
      setEmailAddresses(emailList)
      setWebhookUrl('')
      setSlackWebhook('')
    } else if (cp.contact_type === 'slack') {
      setSlackWebhook((settings.url as string) || '')
      setEmailAddresses('')
      setWebhookUrl('')
    } else {
      setWebhookUrl((settings.url as string) || '')
      setEmailAddresses('')
      setSlackWebhook('')
    }

    setShowAddContact(true)
  }

  const buildSettings = (): Record<string, unknown> => {
    switch (contactType) {
      case 'email':
        // Grafana expects addresses as semicolon-separated string
        const addrs = emailAddresses.split(',').map(e => e.trim()).filter(Boolean).join(';')
        return { addresses: addrs, singleEmail: false }
      case 'msteams':
        return { url: webhookUrl }
      case 'slack':
        return { url: slackWebhook }
      case 'webhook':
        return { url: webhookUrl }
      case 'discord':
        return { url: webhookUrl }
      default:
        return {}
    }
  }

  const handleCreateContact = () => {
    const settings = buildSettings()

    createContactMutation.mutate({
      name: contactName,
      description: contactDescription || undefined,
      contact_type: contactType,
      settings,
    }, {
      onSuccess: () => resetForm(),
    })
  }

  const handleUpdateContact = () => {
    if (!editingContactId) return

    const settings = buildSettings()

    updateContactMutation.mutate({
      id: editingContactId,
      data: {
        name: contactName,
        description: contactDescription || undefined,
        contact_type: contactType,
        settings,
      },
    }, {
      onSuccess: () => {
        resetForm()
        setShowAddContact(false)
      },
    })
  }

  return (
    <div>
      {/* Actions */}
      <div className="flex justify-between items-center mb-4">
        <p className="text-gray-600">
          Contact points define where alerts are sent (email, Teams, Slack, etc.)
        </p>
        <div className="flex gap-2">
          <button
            onClick={() => syncContactsMutation.mutate(undefined)}
            disabled={syncContactsMutation.isPending}
            className="px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${syncContactsMutation.isPending ? 'animate-spin' : ''}`} />
            Sync to Grafana
          </button>
          <button
            onClick={() => {
              if (showAddContact) {
                setShowAddContact(false)
                resetForm()
              } else {
                resetForm()
                setShowAddContact(true)
              }
            }}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2"
          >
            <Plus className="h-4 w-4" />
            Add Contact Point
          </button>
        </div>
      </div>

      {/* Add/Edit Contact Form */}
      {showAddContact && (
        <div className="bg-white p-6 rounded-lg shadow border border-gray-200 mb-6">
          <h3 className="text-lg font-semibold mb-4">
            {editingContactId ? 'Edit Contact Point' : 'Add Contact Point'}
          </h3>

          {/* Contact Type Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Notification Type
            </label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {[
                { type: 'email', label: 'Email', icon: Mail },
                { type: 'msteams', label: 'Microsoft Teams', icon: MessageSquare },
                { type: 'slack', label: 'Slack', icon: MessageSquare },
                { type: 'webhook', label: 'Webhook', icon: Webhook },
              ].map(({ type, label, icon: Icon }) => (
                <button
                  key={type}
                  onClick={() => setContactType(type)}
                  className={`p-3 rounded-lg border-2 flex flex-col items-center gap-2 transition-colors ${
                    contactType === type
                      ? 'border-red-500 bg-red-50 text-red-700'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span className="text-sm font-medium">{label}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Name *
              </label>
              <input
                type="text"
                value={contactName}
                onChange={e => setContactName(e.target.value)}
                placeholder="e.g., GPU Alerts Team"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-red-500 focus:border-red-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <input
                type="text"
                value={contactDescription}
                onChange={e => setContactDescription(e.target.value)}
                placeholder="Optional description"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-red-500 focus:border-red-500"
              />
            </div>
          </div>

          {/* Type-specific settings */}
          <div className="mt-4">
            {contactType === 'email' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Email Addresses *
                </label>
                <input
                  type="text"
                  value={emailAddresses}
                  onChange={e => setEmailAddresses(e.target.value)}
                  placeholder="user1@example.com, user2@example.com"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-red-500 focus:border-red-500"
                />
                <p className="text-sm text-gray-500 mt-1">Separate multiple addresses with commas</p>
              </div>
            )}
            {contactType === 'msteams' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Teams Webhook URL *
                </label>
                <input
                  type="url"
                  value={webhookUrl}
                  onChange={e => setWebhookUrl(e.target.value)}
                  placeholder="https://outlook.office.com/webhook/..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-red-500 focus:border-red-500"
                />
                <p className="text-sm text-gray-500 mt-1">
                  Get this from Teams channel → Connectors → Incoming Webhook
                </p>
              </div>
            )}
            {contactType === 'slack' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Slack Webhook URL *
                </label>
                <input
                  type="url"
                  value={slackWebhook}
                  onChange={e => setSlackWebhook(e.target.value)}
                  placeholder="https://hooks.slack.com/services/..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-red-500 focus:border-red-500"
                />
                <p className="text-sm text-gray-500 mt-1">
                  Get this from Slack → Apps → Incoming Webhooks
                </p>
              </div>
            )}
            {contactType === 'webhook' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Webhook URL *
                </label>
                <input
                  type="url"
                  value={webhookUrl}
                  onChange={e => setWebhookUrl(e.target.value)}
                  placeholder="https://your-service.com/webhook"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-red-500 focus:border-red-500"
                />
              </div>
            )}
          </div>

          <div className="mt-6 flex justify-end gap-2">
            <button
              onClick={() => {
                setShowAddContact(false)
                resetForm()
              }}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            {editingContactId ? (
              <button
                onClick={handleUpdateContact}
                disabled={!contactName || updateContactMutation.isPending}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
              >
                {updateContactMutation.isPending ? 'Saving...' : 'Save Changes'}
              </button>
            ) : (
              <button
                onClick={handleCreateContact}
                disabled={!contactName || createContactMutation.isPending}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
              >
                {createContactMutation.isPending ? 'Creating...' : 'Create Contact Point'}
              </button>
            )}
          </div>
        </div>
      )}

      {/* Contact Points List */}
      {contactPoints.length === 0 ? (
        <div className="bg-white p-12 rounded-lg shadow border border-gray-200 text-center">
          <Send className="h-12 w-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Contact Points</h3>
          <p className="text-gray-500 mb-4">
            Add a contact point to receive alert notifications
          </p>
          <button
            onClick={() => setShowAddContact(true)}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Add Your First Contact Point
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {contactPoints.map(cp => {
            const Icon = contactTypeIcons[cp.contact_type] || Bell
            return (
              <div key={cp.id} className="bg-white p-4 rounded-lg shadow border border-gray-200">
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-gray-100 rounded-lg">
                      <Icon className="h-5 w-5 text-gray-600" />
                    </div>
                    <div>
                      <h4 className="font-medium text-gray-900">{cp.name}</h4>
                      <span className="text-sm text-gray-500">
                        {contactTypeLabels[cp.contact_type] || cp.contact_type}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => startEditing(cp)}
                      className="p-2 text-gray-400 hover:text-green-600 hover:bg-green-50 rounded"
                      title="Edit contact point"
                    >
                      <Edit2 className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => testContactMutation.mutate(cp.id)}
                      disabled={testContactMutation.isPending}
                      className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded"
                      title="Send test notification"
                    >
                      <Send className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => {
                        if (confirm('Delete this contact point?')) {
                          deleteContactMutation.mutate(cp.id)
                        }
                      }}
                      className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded"
                      title="Delete contact point"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {cp.description && (
                  <p className="text-sm text-gray-600 mb-3">{cp.description}</p>
                )}

                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-500">
                    {cp.alert_rule_count} alert{cp.alert_rule_count !== 1 ? 's' : ''} using this
                  </span>
                  {cp.grafana_uid ? (
                    <span className="flex items-center gap-1 text-green-600">
                      <CheckCircle2 className="h-4 w-4" />
                      Synced
                    </span>
                  ) : cp.sync_error ? (
                    <span className="flex items-center gap-1 text-red-600" title={cp.sync_error}>
                      <XCircle className="h-4 w-4" />
                      Error
                    </span>
                  ) : (
                    <span className="text-gray-400">Not synced</span>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// Alert Rules Tab Component
function AlertRulesTab({
  alertRules,
  templates,
  contactPoints,
  monitoringServers,
  nodeGroups,
  rulesByCategory,
  expandedCategories,
  toggleCategory,
  showAddRules,
  setShowAddRules,
  createRulesMutation,
  deleteRuleMutation,
  enableRuleMutation,
  disableRuleMutation,
  syncRulesMutation,
}: {
  alertRules: AlertRule[]
  templates: AlertTemplate[]
  contactPoints: ContactPoint[]
  monitoringServers: MonitoringServer[]
  nodeGroups: NodeGroup[]
  rulesByCategory: Record<string, AlertRule[]>
  expandedCategories: Set<string>
  toggleCategory: (category: string) => void
  showAddRules: boolean
  setShowAddRules: (show: boolean) => void
  createRulesMutation: ReturnType<typeof useMutation<unknown, Error, Parameters<typeof createAlertRules>[0]>>
  deleteRuleMutation: ReturnType<typeof useMutation<unknown, Error, number>>
  enableRuleMutation: ReturnType<typeof useMutation<unknown, Error, number>>
  disableRuleMutation: ReturnType<typeof useMutation<unknown, Error, number>>
  syncRulesMutation: ReturnType<typeof useMutation<unknown, Error, void>>
}) {
  const [selectedServer, setSelectedServer] = useState<number | ''>('')
  const [selectedContact, setSelectedContact] = useState<number | ''>('')
  const [selectedNodeGroup, setSelectedNodeGroup] = useState<number | ''>('')
  const [selectedTemplates, setSelectedTemplates] = useState<Set<number>>(new Set())

  // Pre-select default templates when opening the form
  useEffect(() => {
    if (showAddRules && templates.length > 0) {
      const defaults = templates.filter(t => t.is_default).map(t => t.id)
      setSelectedTemplates(new Set(defaults))
    }
  }, [showAddRules, templates])

  // Auto-select first monitoring server
  useEffect(() => {
    if (showAddRules && monitoringServers.length > 0 && selectedServer === '') {
      setSelectedServer(monitoringServers[0].id)
    }
  }, [showAddRules, monitoringServers, selectedServer])

  const toggleTemplate = (id: number) => {
    setSelectedTemplates(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const handleCreateRules = () => {
    if (selectedServer === '' || selectedTemplates.size === 0) return

    createRulesMutation.mutate({
      monitoring_server_id: selectedServer as number,
      contact_point_id: selectedContact === '' ? undefined : (selectedContact as number),
      node_group_id: selectedNodeGroup === '' ? undefined : (selectedNodeGroup as number),
      template_ids: Array.from(selectedTemplates),
    })
  }

  // Group templates by category
  const templatesByCategory = templates.reduce((acc, t) => {
    if (!acc[t.category]) acc[t.category] = []
    acc[t.category].push(t)
    return acc
  }, {} as Record<string, AlertTemplate[]>)

  return (
    <div>
      {/* Actions */}
      <div className="flex justify-between items-center mb-4">
        <p className="text-gray-600">
          Alert rules define what conditions trigger notifications
        </p>
        <div className="flex gap-2">
          <button
            onClick={() => syncRulesMutation.mutate(undefined)}
            disabled={syncRulesMutation.isPending}
            className="px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${syncRulesMutation.isPending ? 'animate-spin' : ''}`} />
            Sync to Grafana
          </button>
          <button
            onClick={() => setShowAddRules(!showAddRules)}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2"
          >
            <Plus className="h-4 w-4" />
            Setup Alerts
          </button>
        </div>
      </div>

      {/* Setup Alerts Form */}
      {showAddRules && (
        <div className="bg-white p-6 rounded-lg shadow border border-gray-200 mb-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Setup Alert Rules
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                <Server className="inline h-4 w-4 mr-1" />
                Monitoring Server *
              </label>
              <select
                value={selectedServer}
                onChange={e => setSelectedServer(e.target.value ? Number(e.target.value) : '')}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-red-500 focus:border-red-500"
              >
                <option value="">Select server...</option>
                {monitoringServers.map(s => (
                  <option key={s.id} value={s.id}>{s.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                <Send className="inline h-4 w-4 mr-1" />
                Send Alerts To
              </label>
              <select
                value={selectedContact}
                onChange={e => setSelectedContact(e.target.value ? Number(e.target.value) : '')}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-red-500 focus:border-red-500"
              >
                <option value="">All contact points</option>
                {contactPoints.map(cp => (
                  <option key={cp.id} value={cp.id}>
                    {cp.name} ({contactTypeLabels[cp.contact_type] || cp.contact_type})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                <Users className="inline h-4 w-4 mr-1" />
                Apply To Node Group
              </label>
              <select
                value={selectedNodeGroup}
                onChange={e => setSelectedNodeGroup(e.target.value ? Number(e.target.value) : '')}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-red-500 focus:border-red-500"
              >
                <option value="">All node groups</option>
                {nodeGroups.map(ng => (
                  <option key={ng.id} value={ng.id}>{ng.name}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Template Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Select Alert Types
            </label>

            {Object.entries(templatesByCategory).map(([category, catTemplates]) => (
              <div key={category} className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm font-medium text-gray-600">
                    {categoryLabels[category] || category}
                  </span>
                  <div className="flex-1 border-t border-gray-200" />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                  {catTemplates.map(template => (
                    <label
                      key={template.id}
                      className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                        selectedTemplates.has(template.id)
                          ? 'border-red-500 bg-red-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={selectedTemplates.has(template.id)}
                        onChange={() => toggleTemplate(template.id)}
                        className="mt-1 h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300 rounded"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-gray-900 text-sm">
                            {template.display_name}
                          </span>
                          {template.is_default && (
                            <span className="px-1.5 py-0.5 text-xs bg-green-100 text-green-700 rounded">
                              Recommended
                            </span>
                          )}
                          <span className={`px-1.5 py-0.5 text-xs rounded ${severityColors[template.default_severity]}`}>
                            {template.default_severity}
                          </span>
                        </div>
                        {template.description && (
                          <p className="text-xs text-gray-500 mt-1 line-clamp-2">
                            {template.description}
                          </p>
                        )}
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-500">
              {selectedTemplates.size} alert{selectedTemplates.size !== 1 ? 's' : ''} selected
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  setShowAddRules(false)
                  setSelectedTemplates(new Set())
                }}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateRules}
                disabled={selectedServer === '' || selectedTemplates.size === 0 || createRulesMutation.isPending}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
              >
                {createRulesMutation.isPending ? 'Creating...' : `Create ${selectedTemplates.size} Alert${selectedTemplates.size !== 1 ? 's' : ''}`}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Alert Rules List */}
      {alertRules.length === 0 ? (
        <div className="bg-white p-12 rounded-lg shadow border border-gray-200 text-center">
          <Zap className="h-12 w-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Alert Rules</h3>
          <p className="text-gray-500 mb-4">
            Set up alert rules to get notified about issues with your GPU fleet
          </p>
          <button
            onClick={() => setShowAddRules(true)}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Setup Your First Alerts
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          {Object.entries(rulesByCategory).map(([category, rules]) => (
            <div key={category} className="bg-white rounded-lg shadow border border-gray-200 overflow-hidden">
              <button
                onClick={() => toggleCategory(category)}
                className="w-full px-4 py-3 flex items-center justify-between bg-gray-50 hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <span className="font-medium text-gray-900">
                    {categoryLabels[category] || category}
                  </span>
                  <span className="text-sm text-gray-500">
                    {rules.length} rule{rules.length !== 1 ? 's' : ''}
                  </span>
                  <span className="text-sm text-green-600">
                    {rules.filter(r => r.enabled).length} active
                  </span>
                </div>
                {expandedCategories.has(category) ? (
                  <ChevronUp className="h-5 w-5 text-gray-400" />
                ) : (
                  <ChevronDown className="h-5 w-5 text-gray-400" />
                )}
              </button>

              {expandedCategories.has(category) && (
                <div className="divide-y divide-gray-100">
                  {rules.map(rule => (
                    <div key={rule.id} className="px-4 py-3 flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <button
                          onClick={() => {
                            if (rule.enabled) {
                              disableRuleMutation.mutate(rule.id)
                            } else {
                              enableRuleMutation.mutate(rule.id)
                            }
                          }}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                            rule.enabled ? 'bg-green-500' : 'bg-gray-200'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              rule.enabled ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className={`font-medium ${rule.enabled ? 'text-gray-900' : 'text-gray-400'}`}>
                              {rule.name}
                            </span>
                            <span className={`px-1.5 py-0.5 text-xs rounded ${severityColors[rule.severity]}`}>
                              {rule.severity}
                            </span>
                            {rule.grafana_uid && (
                              <span title="Synced to Grafana">
                                <CheckCircle2 className="h-4 w-4 text-green-500" />
                              </span>
                            )}
                            {rule.sync_error && (
                              <span title={rule.sync_error}>
                                <XCircle className="h-4 w-4 text-red-500" />
                              </span>
                            )}
                          </div>
                          <div className="text-sm text-gray-500 flex items-center gap-2">
                            {rule.contact_point_name && (
                              <span>→ {rule.contact_point_name}</span>
                            )}
                            {rule.node_group_name && (
                              <span className="text-gray-400">| {rule.node_group_name}</span>
                            )}
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          if (confirm('Delete this alert rule?')) {
                            deleteRuleMutation.mutate(rule.id)
                          }
                        }}
                        className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
