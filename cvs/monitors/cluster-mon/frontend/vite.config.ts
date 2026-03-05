import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Get backend URL from environment or use localhost default
const backendUrl = process.env.VITE_BACKEND_URL || 'http://localhost:8005'
const backendWsUrl = backendUrl.replace('http://', 'ws://').replace('https://', 'wss://')

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',  // Listen on all network interfaces
    port: 3000,
    proxy: {
      '/api': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/ws': {
        target: backendWsUrl,
        ws: true,
        timeout: 0, // Disable timeout for WebSocket connections
      },
    },
  },
})
