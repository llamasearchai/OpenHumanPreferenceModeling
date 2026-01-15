/**
 * Alerts Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AlertsPage from './Alerts';

// Mock the API client
vi.mock('@/lib/api-client', () => ({
  monitoringApi: {
    getAlerts: vi.fn().mockResolvedValue({
      success: true,
      data: [],
    }),
    acknowledgeAlert: vi.fn().mockResolvedValue({
      success: true,
    }),
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

const renderAlerts = () => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AlertsPage />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('AlertsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders page heading', () => {
      renderAlerts();
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('renders loading state initially', () => {
      renderAlerts();
      expect(document.querySelector('.animate-pulse')).toBeInTheDocument();
    });

    it('renders refresh button', () => {
      renderAlerts();
      expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
    });

    it('renders filter controls', () => {
      renderAlerts();
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });
  });

  describe('export functionality', () => {
    it('renders export button', () => {
      renderAlerts();
      expect(screen.getByRole('button', { name: /Export/i })).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('has accessible cards', () => {
      renderAlerts();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });

    it('refresh button has accessible name', () => {
      renderAlerts();
      expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
    });
  });
});
