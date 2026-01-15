/**
 * Quality Control Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import QualityControlPage from './QualityControl';

// Mock chart components
vi.mock('@/components/widgets/PieChart', () => ({
  PieChart: () => <div data-testid="pie-chart">PieChart</div>,
}));

vi.mock('@/components/widgets/BarChart', () => ({
  BarChart: () => <div data-testid="bar-chart">BarChart</div>,
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

const renderQualityControl = () => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <QualityControlPage />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('QualityControlPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders quality control heading', () => {
      renderQualityControl();
      expect(screen.getByRole('heading', { name: /Quality Control/i })).toBeInTheDocument();
    });

    it('renders page description', () => {
      renderQualityControl();
      expect(screen.getByText(/Monitor annotator/i)).toBeInTheDocument();
    });

    it('renders loading state initially', () => {
      renderQualityControl();
      expect(document.querySelector('.animate-pulse')).toBeInTheDocument();
    });
  });

  describe('search and filter', () => {
    it('renders search input', () => {
      renderQualityControl();
      expect(screen.getByLabelText(/Search annotators/i)).toBeInTheDocument();
    });

    it('can type in search input', async () => {
      const user = userEvent.setup();
      renderQualityControl();

      const searchInput = screen.getByLabelText(/Search annotators/i);
      await user.type(searchInput, 'test');

      expect(searchInput).toHaveValue('test');
    });

    it('has status filter', () => {
      renderQualityControl();
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });
  });

  describe('tabs', () => {
    it('renders tab navigation', () => {
      renderQualityControl();
      expect(screen.getByRole('tablist')).toBeInTheDocument();
    });

    it('has multiple tabs', () => {
      renderQualityControl();
      const tabs = screen.getAllByRole('tab');
      expect(tabs.length).toBeGreaterThan(0);
    });
  });

  describe('refresh functionality', () => {
    it('has refresh button', () => {
      renderQualityControl();
      expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
    });
  });

  describe('cards', () => {
    it('renders card sections', () => {
      renderQualityControl();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('accessibility', () => {
    it('has accessible headings', () => {
      renderQualityControl();
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('search input is focusable', () => {
      renderQualityControl();
      const search = screen.getByLabelText(/Search annotators/i);
      expect(search).not.toHaveAttribute('tabindex', '-1');
    });

    it('tabs are keyboard accessible', () => {
      renderQualityControl();
      const tabs = screen.getAllByRole('tab');
      tabs.forEach((tab) => {
        expect(tab).toHaveAttribute('tabindex');
      });
    });
  });
});
