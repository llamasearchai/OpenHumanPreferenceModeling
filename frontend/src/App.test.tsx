/**
 * App Component Tests
 *
 * Purpose: Test the main application entry point, routing, and provider setup
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from './App';

// Mock eager-loaded critical pages to avoid running real queries in App tests
vi.mock('./pages/Dashboard', () => ({
  default: () => <div>Dashboard Page</div>,
}));
vi.mock('./pages/NotFound', () => ({
  default: () => <div>Not Found Page</div>,
}));

// Mock lazy-loaded pages
vi.mock('./pages/Annotations', () => ({
  default: () => <div>Annotations Page</div>,
}));
vi.mock('./pages/Metrics', () => ({
  default: () => <div>Metrics Page</div>,
}));
vi.mock('./pages/Alerts', () => ({
  default: () => <div>Alerts Page</div>,
}));
vi.mock('./pages/Calibration', () => ({
  default: () => <div>Calibration Page</div>,
}));
vi.mock('./pages/Settings', () => ({
  default: () => <div>Settings Page</div>,
}));
vi.mock('./pages/ActiveLearning', () => ({
  default: () => <div>Active Learning Page</div>,
}));
vi.mock('./pages/FederatedLearning', () => ({
  default: () => <div>Federated Learning Page</div>,
}));
vi.mock('./pages/QualityControl', () => ({
  default: () => <div>Quality Control Page</div>,
}));
vi.mock('./pages/Training', () => ({
  default: () => <div>Training Page</div>,
}));
vi.mock('./pages/Playground', () => ({
  default: () => <div>Playground Page</div>,
}));

// Mock components
vi.mock('./components/ErrorBoundary', () => ({
  ErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
vi.mock('./components/ui/toaster', () => ({
  Toaster: () => <div data-testid="toaster">Toaster</div>,
}));

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the app with providers', () => {
    render(<App />);
    expect(screen.getByTestId('toaster')).toBeInTheDocument();
  });
});
