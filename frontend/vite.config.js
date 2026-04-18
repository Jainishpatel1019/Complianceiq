import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy /api calls to FastAPI during local dev (vite port 5173 → api port 8081)
      '/api': { target: 'http://localhost:8081', changeOrigin: true },
      '/ws':  { target: 'ws://localhost:8081',  ws: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
