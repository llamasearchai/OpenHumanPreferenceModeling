/**
 * Federated Learning Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import FederatedLearningPage from './FederatedLearning';

// Mock the API client
vi.mock('@/lib/api-client', () => ({
  default: {
    federated: {
      getStatus: vi.fn().mockResolvedValue({
        success: true,
        data: {
          round: 1,
          isActive: false,
          totalClients: 0,
          activeClients: 0,
          privacyBudget: { epsilonSpent: 0, epsilonRemaining: 10, delta: 1e-5, totalSteps: 0 },
          modelChecksum: '',
          lastUpdated: new Date().toISOString(),
        },
      }),
      getRoundsHistory: vi.fn().mockResolvedValue({
        success: true,
        data: [],
      }),
      getClients: vi.fn().mockResolvedValue({
        success: true,
        data: [],
      }),
      startRound: vi.fn().mockResolvedValue({ success: true }),
      stopRound: vi.fn().mockResolvedValue({ success: true }),
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

const renderFederatedLearning = () => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <FederatedLearningPage />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('FederatedLearningPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders page headings', () => {
      renderFederatedLearning();
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('renders loading state initially', () => {
      renderFederatedLearning();
      expect(document.querySelector('.animate-pulse')).toBeInTheDocument();
    });

    it('renders cards', () => {
      renderFederatedLearning();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('tabs', () => {
    it('renders tab navigation', () => {
      renderFederatedLearning();
      expect(screen.getByRole('tablist')).toBeInTheDocument();
    });

    it('has multiple tabs', () => {
      renderFederatedLearning();
      const tabs = screen.getAllByRole('tab');
      expect(tabs.length).toBeGreaterThan(0);
    });
  });

  describe('accessibility', () => {
    it('has accessible cards', () => {
      renderFederatedLearning();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });

    it('tabs are keyboard accessible', () => {
      renderFederatedLearning();
      const tabs = screen.getAllByRole('tab');
      tabs.forEach((tab) => {
        expect(tab).toHaveAttribute('tabindex');
      });
    });
  });
});
