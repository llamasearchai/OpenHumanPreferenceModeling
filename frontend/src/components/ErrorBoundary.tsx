/**
 * Error Boundary Component
 *
 * Purpose: Catch and handle React rendering errors with proper logging,
 * recovery options, and specialized handling for different error types.
 */

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { RefreshCw, Home, AlertTriangle, Bug } from 'lucide-react';
import { logError, extractErrorMessage } from '@/lib/errors';

// Error severity levels
type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
  retryCount: number;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  /** Custom fallback render function with error details */
  fallbackRender?: (props: {
    error: Error;
    resetError: () => void;
    retryCount: number;
  }) => React.ReactNode;
  /** Called when an error is caught */
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
  /** Called when error boundary is reset */
  onReset?: () => void;
  /** Maximum retry attempts before showing permanent error */
  maxRetries?: number;
  /** Context identifier for error logging */
  context?: string;
  /** Minimum height for the error display */
  minHeight?: string;
  /** Show detailed error in development mode */
  showDetails?: boolean;
}

/**
 * Determines error severity based on error type and message
 */
function getErrorSeverity(error: Error): ErrorSeverity {
  const message = error.message.toLowerCase();

  if (message.includes('network') || message.includes('fetch')) {
    return 'medium';
  }
  if (message.includes('webgl') || message.includes('three') || message.includes('canvas')) {
    return 'low'; // 3D rendering issues are often recoverable
  }
  if (message.includes('chunk') || message.includes('loading')) {
    return 'medium'; // Lazy loading issues
  }
  if (message.includes('memory') || message.includes('stack')) {
    return 'critical';
  }

  return 'high';
}

/**
 * Gets user-friendly error message
 */
function getUserFriendlyMessage(error: Error): string {
  const message = error.message.toLowerCase();

  if (message.includes('network') || message.includes('fetch')) {
    return 'Unable to connect to the server. Please check your internet connection and try again.';
  }
  if (message.includes('webgl') || message.includes('three') || message.includes('canvas')) {
    return 'There was a problem rendering the 3D visualization. Your browser may not support WebGL.';
  }
  if (message.includes('chunk') || message.includes('loading')) {
    return 'Failed to load a required component. Please refresh the page.';
  }
  if (message.includes('timeout')) {
    return 'The request took too long to complete. Please try again.';
  }

  return extractErrorMessage(error, 'An unexpected error occurred. Please try again.');
}

export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null, retryCount: 0 };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    const { onError, context } = this.props;
    // #region agent log
    fetch('http://127.0.0.1:7259/ingest/44e72182-20fc-4ac5-ace5-6d05735c6915',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ErrorBoundary.tsx:componentDidCatch',message:'React error boundary caught error',data:{message:error.message,componentStack:errorInfo.componentStack?.slice(0,500)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
    // #endregion

    // Store error info for display
    this.setState({ errorInfo });

    // Log error with context
    logError(error, context || 'ErrorBoundary');

    // Call custom error handler if provided
    onError?.(error, errorInfo);
  }

  handleRetry = () => {
    const { maxRetries = 3, onReset } = this.props;
    const { retryCount } = this.state;

    if (retryCount < maxRetries) {
      this.setState((prev) => ({
        hasError: false,
        error: null,
        errorInfo: null,
        retryCount: prev.retryCount + 1,
      }));
      onReset?.();
    }
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  handleRefresh = () => {
    window.location.reload();
  };

  render() {
    const { hasError, error, errorInfo, retryCount } = this.state;
    const {
      children,
      fallback,
      fallbackRender,
      maxRetries = 3,
      minHeight = '400px',
      showDetails = import.meta.env.DEV,
    } = this.props;

    if (hasError && error) {
      // Use custom fallback if provided
      if (fallback) {
        return fallback;
      }

      // Use custom fallback render if provided
      if (fallbackRender) {
        return fallbackRender({ error, resetError: this.handleRetry, retryCount });
      }

      const severity = getErrorSeverity(error);
      const userMessage = getUserFriendlyMessage(error);
      const canRetry = retryCount < maxRetries;

      return (
        <div
          className="flex items-center justify-center p-8"
          style={{ minHeight }}
          role="alert"
          aria-live="assertive"
        >
          <div className="max-w-md w-full space-y-4">
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle className="flex items-center gap-2">
                Something went wrong
                {severity === 'critical' && (
                  <span className="text-xs bg-destructive/20 px-2 py-0.5 rounded">
                    Critical
                  </span>
                )}
              </AlertTitle>
              <AlertDescription className="mt-2">
                {userMessage}
              </AlertDescription>
            </Alert>

            {/* Development error details */}
            {showDetails && (
              <details className="text-xs text-muted-foreground border rounded-lg p-3">
                <summary className="cursor-pointer flex items-center gap-2 font-medium">
                  <Bug className="h-3 w-3" />
                  Technical Details
                </summary>
                <div className="mt-2 space-y-2">
                  <div>
                    <span className="font-medium">Error:</span>{' '}
                    <code className="bg-muted px-1 rounded">{error.message}</code>
                  </div>
                  {error.name && error.name !== 'Error' && (
                    <div>
                      <span className="font-medium">Type:</span> {error.name}
                    </div>
                  )}
                  {errorInfo?.componentStack && (
                    <div>
                      <span className="font-medium">Component Stack:</span>
                      <pre className="mt-1 text-[10px] bg-muted p-2 rounded overflow-auto max-h-32">
                        {errorInfo.componentStack}
                      </pre>
                    </div>
                  )}
                </div>
              </details>
            )}

            {/* Retry count indicator */}
            {retryCount > 0 && (
              <p className="text-xs text-muted-foreground text-center">
                Retry attempt {retryCount} of {maxRetries}
              </p>
            )}

            {/* Action buttons */}
            <div className="flex gap-2">
              {canRetry ? (
                <Button onClick={this.handleRetry} className="flex-1">
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Try Again
                </Button>
              ) : (
                <Button onClick={this.handleRefresh} className="flex-1">
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Refresh Page
                </Button>
              )}
              <Button variant="outline" onClick={this.handleGoHome}>
                <Home className="mr-2 h-4 w-4" />
                Home
              </Button>
            </div>
          </div>
        </div>
      );
    }

    return children;
  }
}

/**
 * Specialized error boundary for 3D/WebGL components
 * Provides better fallback for GPU-related errors
 */
interface Canvas3DErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  height?: string;
}

interface Canvas3DErrorBoundaryState {
  hasError: boolean;
  errorType: 'webgl' | 'memory' | 'unknown' | null;
}

export class Canvas3DErrorBoundary extends React.Component<
  Canvas3DErrorBoundaryProps,
  Canvas3DErrorBoundaryState
> {
  constructor(props: Canvas3DErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, errorType: null };
  }

  static getDerivedStateFromError(error: Error): Canvas3DErrorBoundaryState {
    const message = error.message.toLowerCase();

    let errorType: 'webgl' | 'memory' | 'unknown' = 'unknown';
    if (message.includes('webgl') || message.includes('context')) {
      errorType = 'webgl';
    } else if (message.includes('memory') || message.includes('allocation')) {
      errorType = 'memory';
    }

    return { hasError: true, errorType };
  }

  componentDidCatch(error: Error, _errorInfo: React.ErrorInfo) {
    logError(error, 'Canvas3DErrorBoundary');
  }

  render() {
    const { hasError, errorType } = this.state;
    const { children, fallback, height = '400px' } = this.props;

    if (hasError) {
      if (fallback) {
        return fallback;
      }

      return (
        <div
          className="flex items-center justify-center bg-muted/20 rounded-lg border border-dashed"
          style={{ height }}
          role="alert"
        >
          <div className="text-center p-6 max-w-sm">
            <AlertTriangle className="h-10 w-10 mx-auto mb-4 text-muted-foreground" />
            <h3 className="font-medium mb-2">
              {errorType === 'webgl'
                ? '3D Visualization Unavailable'
                : errorType === 'memory'
                  ? 'Memory Limit Exceeded'
                  : 'Visualization Error'}
            </h3>
            <p className="text-sm text-muted-foreground mb-4">
              {errorType === 'webgl'
                ? 'Your browser or device does not support WebGL. Try using a different browser or enabling hardware acceleration.'
                : errorType === 'memory'
                  ? 'The visualization requires more memory than available. Try reducing the number of data points.'
                  : 'An error occurred while rendering the 3D visualization.'}
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => window.location.reload()}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Reload
            </Button>
          </div>
        </div>
      );
    }

    return children;
  }
}
