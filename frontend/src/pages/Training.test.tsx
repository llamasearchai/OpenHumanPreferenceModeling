/**
 * Training Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import TrainingPage from './Training';

// Mock recharts
vi.mock('recharts', () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Legend: () => null,
}));

// Mock realtime hook
vi.mock('@/hooks/use-realtime', () => ({
  useRealtimeTrainingProgress: () => ({
    metrics: [],
    step: 0,
    loss: 0,
    isConnected: false,
    error: null,
  }),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const renderTraining = () => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <TrainingPage />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('TrainingPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders page headings', () => {
      renderTraining();
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('renders run selector', () => {
      renderTraining();
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    it('renders cards', () => {
      renderTraining();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('tabs', () => {
    it('renders tab navigation', () => {
      renderTraining();
      expect(screen.getByRole('tablist')).toBeInTheDocument();
    });

    it('has loss curve tab', () => {
      renderTraining();
      expect(screen.getByRole('tab', { name: /Loss Curve/i })).toBeInTheDocument();
    });

    it('has gradient norms tab', () => {
      renderTraining();
      expect(screen.getByRole('tab', { name: /Gradient Norms/i })).toBeInTheDocument();
    });

    it('has configuration tab', () => {
      renderTraining();
      expect(screen.getByRole('tab', { name: /Configuration/i })).toBeInTheDocument();
    });
  });

  describe('cards', () => {
    it('renders card sections', () => {
      renderTraining();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('accessibility', () => {
    it('has accessible headings', () => {
      renderTraining();
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('tabs are keyboard accessible', () => {
      renderTraining();
      const tabs = screen.getAllByRole('tab');
      tabs.forEach((tab) => {
        expect(tab).toHaveAttribute('tabindex');
      });
    });

    it('run selector is focusable', () => {
      renderTraining();
      const selector = screen.getByRole('combobox');
      expect(selector).not.toHaveAttribute('tabindex', '-1');
    });
  });
});
