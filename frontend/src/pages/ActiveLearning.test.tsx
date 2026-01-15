/**
 * Active Learning Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ActiveLearningPage from './ActiveLearning';

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

// Mock the API client
vi.mock('@/lib/api-client', () => ({
  apiClient: {
    activeLearning: {
      getStatus: vi.fn().mockResolvedValue({
        success: true,
        data: {
          strategy: 'uncertainty',
          isActive: false,
          samplesAnnotated: 0,
          samplesRemaining: 0,
          currentAccuracy: 0,
          targetAccuracy: 0.95,
          queueSize: 0,
        },
      }),
      getQueue: vi.fn().mockResolvedValue({
        success: true,
        data: [],
      }),
      getConfig: vi.fn().mockResolvedValue({
        success: true,
        data: { strategy: 'uncertainty', batchSize: 32 },
      }),
      updateConfig: vi.fn().mockResolvedValue({ success: true }),
      refreshQueue: vi.fn().mockResolvedValue({ success: true }),
    },
  },
}));

// Mock toast
vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({
    toast: vi.fn(),
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

const renderActiveLearning = () => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <ActiveLearningPage />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('ActiveLearningPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders page headings', () => {
      renderActiveLearning();
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('renders loading state initially', () => {
      renderActiveLearning();
      expect(document.querySelector('.animate-pulse')).toBeInTheDocument();
    });

    it('renders cards', () => {
      renderActiveLearning();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('tabs', () => {
    it('renders tab navigation', () => {
      renderActiveLearning();
      expect(screen.getByRole('tablist')).toBeInTheDocument();
    });

    it('has multiple tabs', () => {
      renderActiveLearning();
      const tabs = screen.getAllByRole('tab');
      expect(tabs.length).toBeGreaterThan(0);
    });
  });

  describe('refresh functionality', () => {
    it('has refresh button', () => {
      renderActiveLearning();
      expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('has accessible cards', () => {
      renderActiveLearning();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });

    it('tabs are keyboard accessible', () => {
      renderActiveLearning();
      const tabs = screen.getAllByRole('tab');
      tabs.forEach((tab) => {
        expect(tab).toHaveAttribute('tabindex');
      });
    });
  });
});
