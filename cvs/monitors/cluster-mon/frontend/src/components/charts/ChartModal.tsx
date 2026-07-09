/**
 * ChartModal — full-screen overlay for expanded chart view.
 * Dismiss by clicking backdrop, pressing Escape, or clicking ×.
 */

import { useEffect, useState } from 'react'
import { X, Maximize2 } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'

// ── Modal overlay ─────────────────────────────────────────────────────────────

interface ChartModalProps {
  title: string
  isOpen: boolean
  onClose: () => void
  children: React.ReactNode
}

export function ChartModal({ title, isOpen, onClose, children }: ChartModalProps) {
  useEffect(() => {
    if (!isOpen) return
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', handleKey)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', handleKey)
      document.body.style.overflow = ''
    }
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-8"
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/65 backdrop-blur-sm" onClick={onClose} />

      {/* Modal card */}
      <div className="relative z-10 bg-white dark:bg-gray-900 rounded-2xl shadow-2xl w-full max-w-6xl max-h-[92vh] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700 shrink-0">
          <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100">{title}</h3>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title="Close"
          >
            <X className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          </button>
        </div>

        {/* Chart content — fills remaining height */}
        <div className="flex-1 overflow-auto p-6 min-h-0">
          {children}
        </div>
      </div>
    </div>
  )
}

// ── Expandable chart card ─────────────────────────────────────────────────────

/**
 * ExpandableChartCard
 *
 * Wraps any chart in a Card that shows a maximize button on hover.
 * Clicking it opens a full-screen modal with the large version.
 *
 * Props:
 *   title        — Card header text
 *   small        — JSX rendered inside the card (small size)
 *   large        — JSX rendered inside the modal (large size). Falls back to small.
 *   className    — extra Card className
 */
export function ExpandableChartCard({
  title,
  small,
  large,
  className = '',
}: {
  title: string
  small: React.ReactNode
  large?: React.ReactNode
  className?: string
}) {
  const [modalOpen, setModalOpen] = useState(false)

  return (
    <>
      <Card className={`relative group/chart transition-shadow duration-200 hover:shadow-md ${className}`}>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">{title}</CardTitle>
            <button
              onClick={() => setModalOpen(true)}
              title="Expand chart"
              className="opacity-0 group-hover/chart:opacity-100 transition-opacity duration-150
                         p-1.5 rounded-lg border border-gray-200 dark:border-gray-700
                         bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700
                         shadow-sm hover:shadow-md"
            >
              <Maximize2 className="h-3.5 w-3.5 text-gray-500 dark:text-gray-400" />
            </button>
          </div>
        </CardHeader>
        <CardContent>{small}</CardContent>
      </Card>

      <ChartModal
        title={title}
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
      >
        {large ?? small}
      </ChartModal>
    </>
  )
}
