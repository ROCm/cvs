import { useState } from 'react'

interface HeatmapCell {
  node: string
  id: string
  value: number
  label?: string
  tooltipData?: Record<string, any>
}

interface GenericHeatmapProps {
  data: HeatmapCell[]
  title?: string
  colorStops?: { threshold: number; color: string }[]
  valueFormatter?: (value: number) => string
  unit?: string
}

export function GenericHeatmap({
  data,
  title,
  colorStops,
  valueFormatter,
  unit = '',
}: GenericHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<HeatmapCell | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

  // Default color stops for utilization (0-100%)
  const defaultColorStops = [
    { threshold: 90, color: 'bg-red-500' },
    { threshold: 80, color: 'bg-red-400' },
    { threshold: 70, color: 'bg-orange-400' },
    { threshold: 60, color: 'bg-orange-300' },
    { threshold: 50, color: 'bg-yellow-400' },
    { threshold: 40, color: 'bg-yellow-300' },
    { threshold: 30, color: 'bg-green-300' },
    { threshold: 20, color: 'bg-green-200' },
    { threshold: 10, color: 'bg-green-100' },
    { threshold: 0, color: 'bg-gray-100' },
  ]

  const stops = colorStops || defaultColorStops

  // Get color based on value
  const getColor = (value: number): string => {
    for (const stop of stops) {
      if (value >= stop.threshold) {
        return stop.color
      }
    }
    return stops[stops.length - 1].color
  }

  // Group data by node
  const nodeGroups = data.reduce((acc, cell) => {
    if (!acc[cell.node]) {
      acc[cell.node] = []
    }
    acc[cell.node].push(cell)
    return acc
  }, {} as Record<string, HeatmapCell[]>)

  // Get all unique IDs for column headers with numeric sorting (card0, card1, card2, not card1, card10, card2)
  const allIDs = Array.from(new Set(data.map(c => c.id))).sort((a, b) => {
    // Extract numeric part if ID contains 'card'
    if (a.includes('card') && b.includes('card')) {
      const numA = parseInt(a.replace('card', ''))
      const numB = parseInt(b.replace('card', ''))
      return numA - numB
    }
    return a.localeCompare(b)
  })

  const handleMouseMove = (e: React.MouseEvent, cell: HeatmapCell) => {
    setHoveredCell(cell)
    setTooltipPos({ x: e.clientX, y: e.clientY })
  }

  const formatValue = valueFormatter || ((v: number) => v.toFixed(1))

  if (data.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No data available for heatmap
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Legend */}
      <div className="flex items-center gap-4 text-sm flex-wrap">
        <span className="font-medium">{title || 'Value'}:</span>
        <div className="flex items-center gap-2 flex-wrap">
          {stops.map((stop, idx) => {
            // Calculate range for this color
            const nextStop = stops[idx + 1]
            let rangeText = ''

            if (idx === stops.length - 1) {
              // Last stop
              rangeText = `0-${stop.threshold}${unit}`
            } else {
              // Middle stops
              rangeText = `${nextStop.threshold}-${stop.threshold}${unit}`
            }

            return (
              <div key={idx} className="flex items-center gap-1">
                <div className={`w-6 h-4 ${stop.color} border border-gray-300 rounded`}></div>
                <span className="text-xs">{rangeText}</span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Heatmap Grid */}
      <div className="overflow-x-auto">
        <table className="border-collapse">
          <thead>
            <tr>
              <th className="border border-gray-300 bg-gray-50 px-4 py-2 text-left font-semibold text-sm sticky left-0 z-10">
                Node
              </th>
              {allIDs.map(id => (
                <th
                  key={id}
                  className="border border-gray-300 bg-gray-50 px-3 py-2 text-center font-semibold text-sm"
                >
                  {id}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(nodeGroups).map(([node, cells]) => (
              <tr key={node}>
                <td className="border border-gray-300 px-4 py-2 font-medium text-sm bg-white sticky left-0 z-10">
                  {node}
                </td>
                {allIDs.map(id => {
                  const cell = cells.find(c => c.id === id)
                  if (!cell) {
                    return (
                      <td key={id} className="border border-gray-300 bg-gray-50">
                        <div className="w-16 h-12"></div>
                      </td>
                    )
                  }

                  return (
                    <td
                      key={id}
                      className="border border-gray-300 p-0 cursor-pointer transition-all hover:ring-2 hover:ring-blue-400"
                      onMouseMove={(e) => handleMouseMove(e, cell)}
                      onMouseLeave={() => setHoveredCell(null)}
                    >
                      <div
                        className={`w-16 h-12 flex items-center justify-center ${getColor(
                          cell.value
                        )} transition-colors`}
                      >
                        <span className="font-bold text-xs text-gray-800">
                          {formatValue(cell.value)}{unit}
                        </span>
                      </div>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Tooltip */}
      {hoveredCell && (
        <div
          className="fixed z-50 bg-gray-900 text-white text-xs rounded-lg shadow-lg p-3 pointer-events-none"
          style={{
            left: `${tooltipPos.x + 15}px`,
            top: `${tooltipPos.y + 15}px`,
          }}
        >
          <div className="space-y-1">
            <div className="font-semibold border-b border-gray-700 pb-1 mb-1">
              {hoveredCell.node} / {hoveredCell.id}
            </div>
            <div>
              <span className="text-gray-400">{title || 'Value'}:</span>{' '}
              <span className="font-medium">{formatValue(hoveredCell.value)}{unit}</span>
            </div>
            {hoveredCell.tooltipData && Object.entries(hoveredCell.tooltipData).map(([key, value]) => (
              <div key={key}>
                <span className="text-gray-400">{key}:</span>{' '}
                <span className="font-medium">{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
