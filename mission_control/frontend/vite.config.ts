import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const apiTarget = process.env.MC_API_TARGET || 'http://127.0.0.1:8001'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': apiTarget,
      '/ws': { target: apiTarget.replace(/^http/, 'ws'), ws: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/recharts')) return 'charts-vendor'
          if (id.includes('node_modules/@tanstack')) return 'query-vendor'
          if (id.includes('node_modules/lucide-react')) return 'icons-vendor'
          if (
            id.includes('node_modules/react/') ||
            id.includes('node_modules/react-dom/') ||
            id.includes('node_modules/react-router')
          ) return 'react-vendor'
        },
      },
    },
  },
})
