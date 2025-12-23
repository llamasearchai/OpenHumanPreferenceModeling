import { Suspense, lazy } from 'react';
import { QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { queryClient } from './lib/query/query-client';
import { AuthProvider } from './contexts/AuthContext';
import { ProtectedRoute } from './components/ProtectedRoute';
import { AppLayout } from './components/layout/AppLayout';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Toaster } from './components/ui/toaster';
import { Skeleton } from './components/ui/skeleton';

// Eager imports for critical path (login, initial load)
import LoginPage from './pages/Login';
import RegisterPage from './pages/Register';
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

function App() {
  return (
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <ErrorBoundary>
            <Routes>
              {/* Public routes */}
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />

              {/* Protected app shell */}
              <Route
                element={
                  <ProtectedRoute>
                    <AppLayout />
                  </ProtectedRoute>
                }
              >
                <Route index element={<DashboardPage />} />
                <Route path="/dashboard" element={<Navigate to="/" replace />} />
                <Route
                  path="/annotations"
                  element={
                    <Suspense fallback={<PageSkeleton />}>
                      <AnnotationsPage />
                    </Suspense>
                  }
                />
                <Route
                  path="/metrics"
                  element={
                    <Suspense fallback={<PageSkeleton />}>
                      <MetricsPage />
                    </Suspense>
                  }
                />
                <Route
                  path="/alerts"
                  element={
                    <Suspense fallback={<PageSkeleton />}>
                      <AlertsPage />
                    </Suspense>
                  }
                />
                <Route
                  path="/calibration"
                  element={
                    <Suspense fallback={<PageSkeleton />}>
                      <CalibrationPage />
                    </Suspense>
                  }
                />
                <Route
                  path="/active-learning"
                  element={
                    <Suspense fallback={<PageSkeleton />}>
                      <ActiveLearningPage />
                    </Suspense>
                  }
                />
                <Route
                  path="/federated-learning"
                  element={
                    <Suspense fallback={<PageSkeleton />}>
                      <FederatedLearningPage />
                    </Suspense>
                  }
                />
                <Route
                  path="/quality-control"
                  element={
                    <Suspense fallback={<PageSkeleton />}>
                      <QualityControlPage />
                    </Suspense>
                  }
                />
                <Route
                  path="/training"
                  element={
                    <Suspense fallback={<PageSkeleton />}>
                      <TrainingPage />
                    </Suspense>
                  }
                />
                <Route
                  path="/settings"
                  element={
                    <Suspense fallback={<PageSkeleton />}>
                      <SettingsPage />
                    </Suspense>
                  }
                />
                <Route
                  path="/playground"
                  element={
                    <Suspense fallback={<PageSkeleton />}>
                      <PlaygroundPage />
                    </Suspense>
                  }
                />
              </Route>

              {/* Catch-all */}
              <Route path="*" element={<NotFoundPage />} />
            </Routes>

            <Toaster />
          </ErrorBoundary>
        </AuthProvider>
      </QueryClientProvider>
    </BrowserRouter>
  );
}

export default App;
