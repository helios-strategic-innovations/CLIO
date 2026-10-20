import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import '@fontsource-variable/eb-garamond/index.css'
import '@fontsource-variable/eb-garamond/wght-italic.css'
import './index.css'
import App from './App.tsx'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      gcTime: 30 * 60 * 1000,
      refetchOnWindowFocus: true,
      retry: 3,
      retryDelay: attempt => Math.min(1000 * 2 ** attempt, 10000),
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
