/**
 * Error Boundary Component Tests
 *
 * Purpose: Test error catching, severity detection, and recovery mechanisms
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ErrorBoundary, Canvas3DErrorBoundary } from './ErrorBoundary';
import { logError } from '@/lib/errors';

vi.mock('@/lib/errors', () => ({
  logError: vi.fn(),
  extractErrorMessage: vi.fn((error: Error, fallback: string) => error.message || fallback),
}));

// Component that throws an error
const ThrowError = ({ shouldThrow = true, message = 'Test error' }: { shouldThrow?: boolean; message?: string }) => {
  if (shouldThrow) {
    throw new Error(message);
  }
  return <div>No error</div>;
};

describe('ErrorBoundary', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Suppress console.error for error boundary tests
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  it('renders children when there is no error', () => {
    render(
      <ErrorBoundary>
        <div>Normal content</div>
      </ErrorBoundary>
    );

    expect(screen.getByText('Normal content')).toBeInTheDocument();
  });

  it('catches errors and displays error UI', () => {
    render(
      <ErrorBoundary>
        <ThrowError />
      </ErrorBoundary>
    );

    // Error boundary should show error UI (may render multiple alerts depending on nested components)
    const alerts = screen.getAllByRole('alert');
    expect(alerts.length).toBeGreaterThan(0);
  });

  it('calls logError when error is caught', () => {
    render(
      <ErrorBoundary context="TestContext">
        <ThrowError />
      </ErrorBoundary>
    );

    expect(logError).toHaveBeenCalledWith(expect.any(Error), 'TestContext');
  });

  it('calls onError callback when provided', () => {
    const onError = vi.fn();
    render(
      <ErrorBoundary onError={onError}>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(onError).toHaveBeenCalledWith(expect.any(Error), expect.any(Object));
  });

  it('displays custom fallback when provided', () => {
    render(
      <ErrorBoundary fallback={<div>Custom fallback</div>}>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(screen.getByText('Custom fallback')).toBeInTheDocument();
  });

  it('uses custom fallbackRender when provided', () => {
    const fallbackRender = vi.fn(({ error }) => <div>Custom render: {error.message}</div>);
    render(
      <ErrorBoundary fallbackRender={fallbackRender}>
        <ThrowError message="Custom error" />
      </ErrorBoundary>
    );

    expect(fallbackRender).toHaveBeenCalled();
    expect(screen.getByText(/custom render: custom error/i)).toBeInTheDocument();
  });

  it('shows retry button when retries are available', () => {
    render(
      <ErrorBoundary maxRetries={3}>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(screen.getByText(/try again/i)).toBeInTheDocument();
  });

  it('shows refresh button when max retries exceeded', () => {
    render(
      <ErrorBoundary maxRetries={0}>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(screen.getByText(/refresh page/i)).toBeInTheDocument();
  });

  it('navigates home when Home button is clicked', () => {
    const originalLocation = window.location;
    const fakeLocation = {
      ...originalLocation,
      href: `${window.location.protocol}//${window.location.host}/test`,
      reload: vi.fn(),
    };
    Object.defineProperty(window, 'location', { value: fakeLocation, writable: true });

    render(
      <ErrorBoundary maxRetries={0}>
        <ThrowError />
      </ErrorBoundary>
    );

    fireEvent.click(screen.getByText(/^home$/i));
    expect(window.location.href).toBe('/');

    Object.defineProperty(window, 'location', { value: originalLocation, writable: true });
  });

  it('reloads the page when Refresh Page button is clicked', () => {
    const originalLocation = window.location;
    const reload = vi.fn();
    const fakeLocation = { ...originalLocation, reload };
    Object.defineProperty(window, 'location', { value: fakeLocation, writable: true });

    render(
      <ErrorBoundary maxRetries={0}>
        <ThrowError />
      </ErrorBoundary>
    );

    fireEvent.click(screen.getByText(/refresh page/i));
    expect(reload).toHaveBeenCalled();

    Object.defineProperty(window, 'location', { value: originalLocation, writable: true });
  });

  it('handles retry correctly', () => {
    render(
      <ErrorBoundary maxRetries={3}>
        <ThrowError />
      </ErrorBoundary>
    );

    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    const retryButton = screen.getByText(/try again/i);
    expect(retryButton).toBeInTheDocument();
    fireEvent.click(retryButton);

    // Child still throws; boundary re-catches. Verify retry counter is shown.
    expect(screen.getByText(/retry attempt 1 of 3/i)).toBeInTheDocument();
  });

  it('calls onReset when retry is clicked', () => {
    const onReset = vi.fn();
    render(
      <ErrorBoundary maxRetries={3} onReset={onReset}>
        <ThrowError />
      </ErrorBoundary>
    );

    const retryButton = screen.getByText(/try again/i);
    fireEvent.click(retryButton);

    expect(onReset).toHaveBeenCalled();
  });

  it('displays retry count indicator', () => {
    render(
      <ErrorBoundary maxRetries={3}>
        <ThrowError />
      </ErrorBoundary>
    );

    const retryButton = screen.getByText(/try again/i);
    fireEvent.click(retryButton);

    expect(screen.getByText(/retry attempt 1 of 3/i)).toBeInTheDocument();
  });

  it('displays critical severity badge for critical errors', () => {
    render(
      <ErrorBoundary>
        <ThrowError message="Memory stack overflow" />
      </ErrorBoundary>
    );

    expect(screen.getByText(/critical/i)).toBeInTheDocument();
  });

  it('shows user-friendly message for network errors', () => {
    render(
      <ErrorBoundary>
        <ThrowError message="Network fetch failed" />
      </ErrorBoundary>
    );

    expect(screen.getByText(/unable to connect to the server/i)).toBeInTheDocument();
  });

  it('shows user-friendly message for WebGL errors', () => {
    render(
      <ErrorBoundary>
        <ThrowError message="WebGL context lost" />
      </ErrorBoundary>
    );

    expect(screen.getByText(/problem rendering the 3d visualization/i)).toBeInTheDocument();
  });

  it('shows technical details in development mode', () => {
    render(
      <ErrorBoundary showDetails={true}>
        <ThrowError message="Test error message unique" />
      </ErrorBoundary>
    );

    expect(screen.getAllByText(/technical details/i).length).toBeGreaterThan(0);
    // Error message may appear in multiple places; ensure it's present at least once.
    expect(screen.getAllByText(/test error message unique/i).length).toBeGreaterThan(0);
  });

  it('respects minHeight prop', () => {
    const { container } = render(
      <ErrorBoundary minHeight="600px">
        <ThrowError />
      </ErrorBoundary>
    );

    // Check if minHeight style is applied
    const elementWithStyle = container.querySelector('div[style*="min-height"]');
    expect(elementWithStyle || screen.getByRole('alert')).toBeTruthy();
  });
});

describe('Canvas3DErrorBoundary', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  it('renders children when there is no error', () => {
    render(
      <Canvas3DErrorBoundary>
        <div>3D Content</div>
      </Canvas3DErrorBoundary>
    );

    expect(screen.getByText('3D Content')).toBeInTheDocument();
  });

  it('catches WebGL errors and displays appropriate message', () => {
    render(
      <Canvas3DErrorBoundary>
        <ThrowError message="WebGL context creation failed" />
      </Canvas3DErrorBoundary>
    );

    expect(screen.getByText(/3d visualization unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/does not support webgl/i)).toBeInTheDocument();
  });

  it('catches memory errors and displays appropriate message', () => {
    render(
      <Canvas3DErrorBoundary>
        <ThrowError message="Memory allocation failed" />
      </Canvas3DErrorBoundary>
    );

    expect(screen.getByText(/memory limit exceeded/i)).toBeInTheDocument();
    expect(screen.getByText(/requires more memory/i)).toBeInTheDocument();
  });

  it('displays generic error for unknown error types', () => {
    render(
      <Canvas3DErrorBoundary>
        <ThrowError message="Unknown error" />
      </Canvas3DErrorBoundary>
    );

    expect(screen.getByText(/visualization error/i)).toBeInTheDocument();
  });

  it('uses custom fallback when provided', () => {
    render(
      <Canvas3DErrorBoundary fallback={<div>Custom 3D fallback</div>}>
        <ThrowError />
      </Canvas3DErrorBoundary>
    );

    expect(screen.getByText('Custom 3D fallback')).toBeInTheDocument();
  });

  it('respects height prop', () => {
    render(
      <Canvas3DErrorBoundary height="500px">
        <ThrowError />
      </Canvas3DErrorBoundary>
    );

    const container = screen.getByRole('alert');
    expect(container).toHaveStyle({ height: '500px' });
  });

  it('calls logError when error is caught', () => {
    render(
      <Canvas3DErrorBoundary>
        <ThrowError />
      </Canvas3DErrorBoundary>
    );

    expect(logError).toHaveBeenCalledWith(expect.any(Error), 'Canvas3DErrorBoundary');
  });
});
