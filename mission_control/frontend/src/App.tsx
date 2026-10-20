import { lazy, Suspense, type ReactNode } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { getToken } from './lib/api';
import Layout from './components/Layout';
import PageShell from './components/PageShell';
import { PanelSkeleton } from './components/DataState';

const Login = lazy(() => import('./pages/Login'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Portfolio = lazy(() => import('./pages/Portfolio'));
const Orders = lazy(() => import('./pages/Orders'));
const Hydrinia = lazy(() => import('./pages/Hydrinia'));
const Knowledge = lazy(() => import('./pages/Knowledge'));
const News = lazy(() => import('./pages/News'));
const Logs = lazy(() => import('./pages/Logs'));
const Control = lazy(() => import('./pages/Control'));
const Analytics = lazy(() => import('./pages/Analytics'));

function ProtectedRoute({ children }: { children: ReactNode }) {
  return getToken() ? <>{children}</> : <Navigate to="/login" replace />;
}

function RouteFallback() {
  return <PageShell><PanelSkeleton /></PageShell>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<RouteFallback />}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
            <Route index element={<Dashboard />} />
            <Route path="portfolio" element={<Portfolio />} />
            <Route path="orders" element={<Orders />} />
            <Route path="hydrinia" element={<Hydrinia />} />
            <Route path="knowledge" element={<Knowledge />} />
            <Route path="news" element={<News />} />
            <Route path="logs" element={<Logs />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="control" element={<Control />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
