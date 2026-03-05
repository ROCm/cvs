import { useEffect, useRef, useState } from 'react'
import CytoscapeComponent from 'react-cytoscapejs'
import Cytoscape from 'cytoscape'
import dagre from 'cytoscape-dagre'
import { ZoomIn, ZoomOut, Maximize2, Maximize, Minimize } from 'lucide-react'

// Register dagre layout
Cytoscape.use(dagre)

interface NetworkTopologyGraphProps {
  neighbors: Array<{
    node: string
    interface: string
    neighbor_name: string
    neighbor_port: string
    neighbor_descr: string
    neighbor_mgmt_ip: string
  }>
}

export function NetworkTopologyGraph({ neighbors }: NetworkTopologyGraphProps) {
  const cyRef = useRef<Cytoscape.Core | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [elements, setElements] = useState<any[]>([])
  const [selectedNode, setSelectedNode] = useState<any>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)

  console.log('NetworkTopologyGraph render: neighbors count =', neighbors.length, 'elements count =', elements.length)

  useEffect(() => {
    // Transform LLDP data into Cytoscape graph format
    const nodes: any[] = []
    const edges: any[] = []
    const nodeSet = new Set<string>()

    console.log('NetworkTopologyGraph: Processing neighbors:', neighbors.length)

    neighbors.forEach((neighbor, idx) => {
      const serverId = neighbor.node
      const switchId = neighbor.neighbor_name

      // Add server node
      if (!nodeSet.has(serverId)) {
        nodeSet.add(serverId)
        nodes.push({
          data: {
            id: serverId,
            label: serverId,
            type: 'server',
            nodeType: 'server',
          },
        })
      }

      // Add switch/neighbor node
      if (!nodeSet.has(switchId)) {
        nodeSet.add(switchId)

        // Determine if it's a switch or another server based on description
        const isSwitch = neighbor.neighbor_descr?.toLowerCase().includes('arista') ||
                        neighbor.neighbor_descr?.toLowerCase().includes('cisco') ||
                        neighbor.neighbor_descr?.toLowerCase().includes('switch') ||
                        neighbor.neighbor_descr?.toLowerCase().includes('juniper') ||
                        neighbor.neighbor_descr?.toLowerCase().includes('aruba') ||
                        neighbor.neighbor_port?.match(/ethernet|eth\d+\/|gi\d+\/|\d+\/\d+\/\d+/) !== null

        nodes.push({
          data: {
            id: switchId,
            label: switchId,
            type: isSwitch ? 'switch' : 'server',
            nodeType: isSwitch ? 'switch' : 'server',
            description: neighbor.neighbor_descr,
            mgmt_ip: neighbor.neighbor_mgmt_ip,
          },
        })
      }

      // Add edge (connection)
      // For hierarchical layout with TB direction, connect FROM switch TO server
      // so switches appear at the top level
      edges.push({
        data: {
          id: `edge-${idx}`,
          source: switchId,
          target: serverId,
          label: `${neighbor.neighbor_port} ↔ ${neighbor.interface}`,
          sourcePort: neighbor.neighbor_port,
          targetPort: neighbor.interface,
        },
      })
    })

    console.log('NetworkTopologyGraph: Created nodes:', nodes.length, 'edges:', edges.length)
    const allElements = [...nodes, ...edges]
    console.log('NetworkTopologyGraph: Total elements:', allElements.length)
    setElements(allElements)
  }, [neighbors])

  // Cytoscape stylesheet
  const stylesheet = [
    {
      selector: 'node',
      style: {
        'background-color': '#3b82f6',
        'label': 'data(label)',
        'color': '#000000',
        'text-valign': 'center',
        'text-halign': 'center',
        'font-size': '8px',
        'font-weight': '600',
        'width': '80px',
        'height': '80px',
        'border-width': '2px',
        'border-color': '#1e40af',
        'text-wrap': 'wrap',
        'text-max-width': '70px',
      },
    },
    {
      selector: 'node[type="switch"]',
      style: {
        'background-color': '#10b981',
        'shape': 'round-rectangle',
        'width': '120px',
        'height': '60px',
        'border-color': '#059669',
        'font-size': '9px',
        'font-weight': '700',
        'color': '#000000',
        'text-max-width': '110px',
      },
    },
    {
      selector: 'node[type="server"]',
      style: {
        'background-color': '#3b82f6',
        'shape': 'ellipse',
        'border-color': '#1e40af',
        'width': '70px',
        'height': '70px',
        'font-size': '7px',
        'color': '#000000',
        'text-max-width': '60px',
      },
    },
    {
      selector: 'node:selected',
      style: {
        'background-color': '#f59e0b',
        'border-color': '#d97706',
        'border-width': '2px',
      },
    },
    {
      selector: 'edge',
      style: {
        'width': 1,
        'line-color': '#cbd5e1',
        'target-arrow-color': '#cbd5e1',
        'curve-style': 'bezier',
        'font-size': '7px',
        'color': '#64748b',
        'text-background-color': '#ffffff',
        'text-background-opacity': 0.9,
        'text-background-padding': '2px',
        'text-rotation': 'autorotate',
        'label': '',  // Hide labels by default
      },
    },
    {
      selector: 'edge:selected',
      style: {
        'line-color': '#f59e0b',
        'width': 2,
        'label': 'data(label)',
      },
    },
    {
      selector: 'edge:hover',
      style: {
        'label': 'data(label)',
      },
    },
  ]

  // Layout configuration - dagre for hierarchical layout
  const layout = {
    name: 'dagre',
    rankDir: 'TB', // Top to Bottom - with edges from switch to server, switches appear at top
    nodeSep: 150, // Horizontal spacing between nodes (increased for larger nodes)
    rankSep: 200, // Vertical spacing between ranks (increased for larger nodes)
    animate: true,
    animationDuration: 500,
    fit: true,
    padding: 50,
  }

  const handleZoomIn = () => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 1.2)
      cyRef.current.center()
    }
  }

  const handleZoomOut = () => {
    if (cyRef.current) {
      cyRef.current.zoom(cyRef.current.zoom() * 0.8)
      cyRef.current.center()
    }
  }

  const handleFitToScreen = () => {
    if (cyRef.current) {
      cyRef.current.fit(undefined, 50)
    }
  }

  const handleFullscreen = async () => {
    if (!containerRef.current) return

    if (!isFullscreen) {
      // Enter fullscreen
      try {
        await containerRef.current.requestFullscreen()
        setIsFullscreen(true)
      } catch (err) {
        console.error('Error entering fullscreen:', err)
      }
    } else {
      // Exit fullscreen
      try {
        await document.exitFullscreen()
        setIsFullscreen(false)
      } catch (err) {
        console.error('Error exiting fullscreen:', err)
      }
    }
  }

  // Listen for fullscreen changes (e.g., user pressing ESC)
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  const handleCyReady = (cy: Cytoscape.Core) => {
    cyRef.current = cy

    // Fit to screen on initial load
    cy.fit(undefined, 50)

    // Add click event handler
    cy.on('tap', 'node', (event) => {
      const node = event.target
      setSelectedNode({
        id: node.data('id'),
        label: node.data('label'),
        type: node.data('type'),
        description: node.data('description'),
        mgmt_ip: node.data('mgmt_ip'),
      })
    })

    // Add edge click handler
    cy.on('tap', 'edge', (event) => {
      const edge = event.target
      setSelectedNode({
        id: edge.data('id'),
        type: 'connection',
        sourcePort: edge.data('sourcePort'),
        targetPort: edge.data('targetPort'),
        source: edge.data('source'),
        target: edge.data('target'),
      })
    })

    // Show edge labels on hover
    cy.on('mouseover', 'edge', (event) => {
      event.target.style('label', event.target.data('label'))
    })

    cy.on('mouseout', 'edge', (event) => {
      if (!event.target.selected()) {
        event.target.style('label', '')
      }
    })
  }

  // Re-run layout when elements change
  useEffect(() => {
    if (cyRef.current && elements.length > 0) {
      const layoutConfig = {
        name: 'dagre',
        rankDir: 'TB',
        nodeSep: 150,
        rankSep: 200,
        animate: true,
        animationDuration: 500,
        fit: true,
        padding: 50,
      }

      const layoutInstance = cyRef.current.layout(layoutConfig)
      layoutInstance.run()
    }
  }, [elements])

  return (
    <div className="relative" ref={containerRef}>
      {/* Graph Container with scrollbars */}
      <div
        className="border border-gray-200 rounded-lg overflow-auto bg-gray-50"
        style={{ height: isFullscreen ? '100vh' : '800px' }}
      >
        <div style={{ width: '2000px', height: '1200px' }}>
          <CytoscapeComponent
            elements={elements}
            style={{ width: '100%', height: '100%' }}
            stylesheet={stylesheet as any}
            layout={layout}
            cy={handleCyReady}
          />
        </div>
      </div>

      {/* Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <button
          onClick={handleZoomIn}
          className="p-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 shadow-sm"
          title="Zoom In"
        >
          <ZoomIn className="h-5 w-5 text-gray-700" />
        </button>
        <button
          onClick={handleZoomOut}
          className="p-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 shadow-sm"
          title="Zoom Out"
        >
          <ZoomOut className="h-5 w-5 text-gray-700" />
        </button>
        <button
          onClick={handleFitToScreen}
          className="p-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 shadow-sm"
          title="Fit to Viewport"
        >
          <Maximize2 className="h-5 w-5 text-gray-700" />
        </button>
        <button
          onClick={handleFullscreen}
          className="p-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 shadow-sm"
          title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
        >
          {isFullscreen ? (
            <Minimize className="h-5 w-5 text-gray-700" />
          ) : (
            <Maximize className="h-5 w-5 text-gray-700" />
          )}
        </button>
      </div>

      {/* Legend */}
      <div className="absolute top-4 left-4 bg-white border border-gray-300 rounded-lg p-2 shadow-sm">
        <div className="text-xs font-semibold mb-1.5">Legend</div>
        <div className="flex items-center gap-2 mb-1.5">
          <div className="w-6 h-6 rounded-full bg-blue-500 border border-blue-700"></div>
          <span className="text-xs">Server</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-8 h-5 rounded bg-green-500 border border-green-700"></div>
          <span className="text-xs">Switch</span>
        </div>
        <div className="text-xs text-gray-500 mt-2 pt-2 border-t">
          Click to select • Hover edge for ports
        </div>
      </div>

      {/* Selected Node Info */}
      {selectedNode && (
        <div className="absolute bottom-4 left-4 bg-white border border-gray-300 rounded-lg p-4 shadow-lg max-w-md">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold text-sm">
              {selectedNode.type === 'connection' ? 'Connection Details' : 'Node Details'}
            </h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-gray-500 hover:text-gray-700"
            >
              ✕
            </button>
          </div>

          {selectedNode.type === 'connection' ? (
            <div className="text-sm space-y-1">
              <div><span className="font-medium">From:</span> {selectedNode.source}</div>
              <div><span className="font-medium">Port:</span> {selectedNode.sourcePort}</div>
              <div><span className="font-medium">To:</span> {selectedNode.target}</div>
              <div><span className="font-medium">Port:</span> {selectedNode.targetPort}</div>
            </div>
          ) : (
            <div className="text-sm space-y-1">
              <div><span className="font-medium">Name:</span> {selectedNode.label}</div>
              <div><span className="font-medium">Type:</span> {selectedNode.type}</div>
              {selectedNode.description && (
                <div><span className="font-medium">Description:</span> {selectedNode.description}</div>
              )}
              {selectedNode.mgmt_ip && (
                <div><span className="font-medium">Mgmt IP:</span> {selectedNode.mgmt_ip}</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
