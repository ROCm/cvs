import { useEffect, useRef } from 'react'
import { useClusterStore } from '@/stores/clusterStore'

// Use relative path to go through Vite proxy
const WS_URL = import.meta.env.VITE_WS_URL || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/metrics`

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const setConnected = useClusterStore((state) => state.setConnected)
  const updateFromWebSocket = useClusterStore((state) => state.updateFromWebSocket)
  const setError = useClusterStore((state) => state.setError)

  const connect = () => {
    try {
      const ws = new WebSocket(WS_URL)

      ws.onopen = () => {
        console.log('WebSocket connected')
        setConnected(true)
        setError(null)

        // Send ping every 30 seconds to keep connection alive
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send('ping')
          }
        }, 30000)

        ws.addEventListener('close', () => {
          clearInterval(pingInterval)
        })
      }

      ws.onmessage = (event) => {
        try {
          // Ignore ping/pong messages
          if (event.data === 'pong' || event.data === 'ping') {
            return
          }

          const data = JSON.parse(event.data)
          updateFromWebSocket(data)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setError('WebSocket connection error')
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected, reconnecting...')
        // Don't immediately set disconnected - keep showing last data
        // setConnected(false)

        // Attempt to reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect...')
          setConnected(false) // Only set disconnected right before reconnecting
          connect()
        }, 3000)
      }

      wsRef.current = ws
    } catch (error) {
      console.error('Failed to create WebSocket:', error)
      setError('Failed to connect to server')
    }
  }

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  return wsRef.current
}
