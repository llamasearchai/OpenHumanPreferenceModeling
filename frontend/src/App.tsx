import React, { Suspense, lazy } from 'react';
import { QueryClientProvider } from '@tanstack/react-query';
import { Navigate, RouterProvider, createBrowserRouter } from 'react-router-dom';
import { queryClient } from './lib/query/query-client';
import { AppLayout } from './components/layout/AppLayout';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Toaster } from './components/ui/toaster';
import { Skeleton } from './components/ui/skeleton';

// Eager imports for critical path
import DashboardPage from './pages/Dashboard';
import NotFoundPage from './pages/NotFound';

// Lazy imports for non-critical pages (code splitting)
const AnnotationsPage = lazy(() => import('./pages/Annotations'));
const MetricsPage = lazy(() => import('./pages/Metrics'));
const AlertsPage = lazy(() => import('./pages/Alerts'));
const CalibrationPage = lazy(() => import('./pages/Calibration'));
const SettingsPage = lazy(() => import('./pages/Settings'));
const ActiveLearningPage = lazy(() => import('./pages/ActiveLearning'));
const FederatedLearningPage = lazy(() => import('./pages/FederatedLearning'));
const QualityControlPage = lazy(() => import('./pages/QualityControl'));
const TrainingPage = lazy(() => import('./pages/Training'));
const PlaygroundPage = lazy(() => import('./pages/Playground'));

// Loading fallback component
function PageSkeleton() {
  return (
    <div className="space-y-6 p-6">
      <Skeleton className="h-8 w-48" />
      <div className="grid gap-4 md:grid-cols-4">
        <Skeleton className="h-32" />
        <Skeleton className="h-32" />
        <Skeleton className="h-32" />
        <Skeleton className="h-32" />
      </div>
      <Skeleton className="h-64" />
    </div>
  );
}

function withPageSuspense(element: React.ReactNode) {
  return <Suspense fallback={<PageSkeleton />}>{element}</Suspense>;
}

const router = createBrowserRouter(
  [
    // App shell
    {
      path: '/',
      element: <AppLayout />,
      children: [
        { index: true, element: <DashboardPage /> },
        { path: 'dashboard', element: <Navigate to="/" replace /> },
        {
          path: 'annotations',
          element: withPageSuspense(<AnnotationsPage />),
        },
        {
          path: 'metrics',
          element: withPageSuspense(<MetricsPage />),
        },
        {
          path: 'alerts',
          element: withPageSuspense(<AlertsPage />),
        },
        {
          path: 'calibration',
          element: withPageSuspense(<CalibrationPage />),
        },
        {
          path: 'active-learning',
          element: withPageSuspense(<ActiveLearningPage />),
        },
        {
          path: 'federated-learning',
          element: withPageSuspense(<FederatedLearningPage />),
        },
        {
          path: 'quality-control',
          element: withPageSuspense(<QualityControlPage />),
        },
        {
          path: 'training',
          element: withPageSuspense(<TrainingPage />),
        },
        {
          path: 'settings',
          element: withPageSuspense(<SettingsPage />),
        },
        {
          path: 'playground',
          element: withPageSuspense(<PlaygroundPage />),
        },
      ],
    },

    // Catch-all
    { path: '*', element: <NotFoundPage /> },
  ],
  { future: { v7_relativeSplatPath: true } }
);

function App() {
  React.useEffect(() => {
    // #region agent log
    fetch('http://127.0.0.1:7259/ingest/44e72182-20fc-4ac5-ace5-6d05735c6915',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'App.tsx:mount',message:'App mounted',data:{},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
    // #endregion
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <RouterProvider
          router={router}
          future={{ v7_startTransition: true }}
        />
        <Toaster />
      </ErrorBoundary>
    </QueryClientProvider>
  );
}

export default App;
