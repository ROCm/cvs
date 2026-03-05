import { useState } from 'react'

interface GPUData {
  node: string
  gpuId: string
  utilization: number
  temperature: number
  memoryUsedMB: number
  memoryTotalMB: number
}

interface GPUHeatmapProps {
  data: GPUData[]
}

export function GPUHeatmap({ data }: GPUHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<GPUData | null>(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })

  // Group data by node
  const nodeGroups = data.reduce((acc, gpu) => {
    if (!acc[gpu.node]) {
      acc[gpu.node] = []
    }
    acc[gpu.node].push(gpu)
    return acc
  }, {} as Record<string, GPUData[]>)

  // Get all unique GPU IDs (card0, card1, etc.) for column headers
  // Sort numerically: card0, card1, card2, ... card10, card11 (not card1, card10, card11, card2)
  const allGPUIDs = Array.from(new Set(data.map(g => g.gpuId))).sort((a, b) => {
    const numA = parseInt(a.replace('card', ''))
    const numB = parseInt(b.replace('card', ''))
    return numA - numB
  })

  // Get color based on utilization percentage
  const getUtilizationColor = (util: number): string => {
    if (util >= 90) return 'bg-red-500'
    if (util >= 80) return 'bg-red-400'
    if (util >= 70) return 'bg-orange-400'
    if (util >= 60) return 'bg-orange-300'
    if (util >= 50) return 'bg-yellow-400'
    if (util >= 40) return 'bg-yellow-300'
    if (util >= 30) return 'bg-green-300'
    if (util >= 20) return 'bg-green-200'
    if (util >= 10) return 'bg-green-100'
    return 'bg-gray-100'
  }

  const handleMouseMove = (e: React.MouseEvent, gpu: GPUData) => {
    setHoveredCell(gpu)
    setTooltipPos({ x: e.clientX, y: e.clientY })
  }

  if (data.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No GPU data available for heatmap
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Legend */}
      <div className="flex items-center gap-4 text-sm">
        <span className="font-medium">Utilization:</span>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <div className="w-6 h-4 bg-gray-100 border border-gray-300 rounded"></div>
            <span className="text-xs">0-10%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-6 h-4 bg-green-200 border border-gray-300 rounded"></div>
            <span className="text-xs">10-30%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-6 h-4 bg-yellow-300 border border-gray-300 rounded"></div>
            <span className="text-xs">30-60%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-6 h-4 bg-orange-400 border border-gray-300 rounded"></div>
            <span className="text-xs">60-80%</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-6 h-4 bg-red-500 border border-gray-300 rounded"></div>
            <span className="text-xs">80-100%</span>
          </div>
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
              {allGPUIDs.map(gpuId => (
                <th
                  key={gpuId}
                  className="border border-gray-300 bg-gray-50 px-3 py-2 text-center font-semibold text-sm"
                >
                  {gpuId}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(nodeGroups).map(([node, gpus]) => (
              <tr key={node}>
                <td className="border border-gray-300 px-4 py-2 font-medium text-sm bg-white sticky left-0 z-10">
                  {node}
                </td>
                {allGPUIDs.map(gpuId => {
                  const gpu = gpus.find(g => g.gpuId === gpuId)
                  if (!gpu) {
                    return (
                      <td key={gpuId} className="border border-gray-300 bg-gray-50">
                        <div className="w-16 h-12"></div>
                      </td>
                    )
                  }

                  return (
                    <td
                      key={gpuId}
                      className="border border-gray-300 p-0 cursor-pointer transition-all hover:ring-2 hover:ring-blue-400"
                      onMouseMove={(e) => handleMouseMove(e, gpu)}
                      onMouseLeave={() => setHoveredCell(null)}
                    >
                      <div
                        className={`w-16 h-12 flex items-center justify-center ${getUtilizationColor(
                          gpu.utilization
                        )} transition-colors`}
                      >
                        <span className="font-bold text-xs text-gray-800">
                          {gpu.utilization.toFixed(0)}%
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
              {hoveredCell.node} / {hoveredCell.gpuId}
            </div>
            <div>
              <span className="text-gray-400">Utilization:</span>{' '}
              <span className="font-medium">{hoveredCell.utilization.toFixed(1)}%</span>
            </div>
            <div>
              <span className="text-gray-400">Temperature:</span>{' '}
              <span className="font-medium">{hoveredCell.temperature.toFixed(0)}°C</span>
            </div>
            <div>
              <span className="text-gray-400">Memory:</span>{' '}
              <span className="font-medium">
                {(hoveredCell.memoryUsedMB / 1024).toFixed(1)} GB / {(hoveredCell.memoryTotalMB / 1024).toFixed(0)} GB
              </span>
            </div>
            <div>
              <span className="text-gray-400">Memory %:</span>{' '}
              <span className="font-medium">
                {hoveredCell.memoryTotalMB > 0
                  ? ((hoveredCell.memoryUsedMB / hoveredCell.memoryTotalMB) * 100).toFixed(1)
                  : 0}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
