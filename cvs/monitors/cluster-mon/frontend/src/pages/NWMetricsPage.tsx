import { useState } from 'react'
import { NICMetricsPage } from './NICMetricsPage'
import { IFoEDetailsPage } from './IFoEDetailsPage'

type Tab = 'scale-out' | 'scale-up'

export function NWMetricsPage() {
  const [activeTab, setActiveTab] = useState<Tab>('scale-out')

  return (
    <div className="space-y-4">
      {/* Page header + tab selector */}
      <div>
        <h1 className="text-3xl font-bold mb-4">NW Metrics</h1>

        <div className="flex gap-1 border-b border-gray-200">
          {([
            { key: 'scale-out', label: 'Scale Out Metrics', desc: 'RDMA / NIC / Ethernet' },
            { key: 'scale-up',  label: 'Scale Up Metrics',  desc: 'IFoE / Rack Trays'    },
          ] as { key: Tab; label: string; desc: string }[]).map(({ key, label, desc }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`px-6 py-3 text-sm font-medium border-b-2 -mb-px transition-colors text-left ${
                activeTab === key
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <div>{label}</div>
              <div className="text-xs font-normal text-gray-400">{desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Tab content — render both but hide the inactive one to preserve state */}
      <div className={activeTab === 'scale-out' ? '' : 'hidden'}>
        <NICMetricsPage />
      </div>
      <div className={activeTab === 'scale-up' ? '' : 'hidden'}>
        <IFoEDetailsPage />
      </div>
    </div>
  )
}
