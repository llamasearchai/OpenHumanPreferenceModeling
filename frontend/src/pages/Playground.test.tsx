/**
 * Playground Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import PlaygroundPage from './Playground';

// Mock recharts
vi.mock('recharts', () => ({
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Cell: () => null,
}));

// Mock the API client
const mockPredict = vi.fn().mockResolvedValue({
  success: true,
  data: {
    probabilities: [0.1, 0.2, 0.4, 0.2, 0.1],
    action_index: 2,
    confidence: 0.4,
  },
});

vi.mock('@/lib/api-client', () => ({
  api: {
    post: (...args: unknown[]) => mockPredict(...args),
  },
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

const renderPlayground = () => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <PlaygroundPage />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('PlaygroundPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders playground heading', () => {
      renderPlayground();
      expect(screen.getByRole('heading', { name: /Model Playground/i })).toBeInTheDocument();
    });

    it('renders state vector input section', () => {
      renderPlayground();
      expect(screen.getByText(/State Vector Input/i)).toBeInTheDocument();
    });

    it('renders all 5 dimension sliders', () => {
      renderPlayground();
      const sliders = document.querySelectorAll('[role="slider"]');
      expect(sliders.length).toBe(5);
    });

    it('renders predict button', () => {
      renderPlayground();
      expect(screen.getByRole('button', { name: /Predict/i })).toBeInTheDocument();
    });

    it('renders results section', () => {
      renderPlayground();
      expect(screen.getByText(/Prediction Results/i)).toBeInTheDocument();
    });
  });

  describe('dimension sliders', () => {
    it('displays all dimension labels', () => {
      renderPlayground();
      expect(screen.getByText(/Dimension 1/i)).toBeInTheDocument();
      expect(screen.getByText(/Dimension 2/i)).toBeInTheDocument();
      expect(screen.getByText(/Dimension 3/i)).toBeInTheDocument();
      expect(screen.getByText(/Dimension 4/i)).toBeInTheDocument();
      expect(screen.getByText(/Dimension 5/i)).toBeInTheDocument();
    });

    it('shows initial slider values', () => {
      renderPlayground();
      // Default value is 0.5 for all sliders
      const values = screen.getAllByText('0.50');
      expect(values.length).toBe(5);
    });
  });

  describe('prediction', () => {
    it('calls predict API when button clicked', async () => {
      const user = userEvent.setup();
      renderPlayground();

      const predictButton = screen.getByRole('button', { name: /Predict/i });
      await user.click(predictButton);

      await waitFor(() => {
        expect(mockPredict).toHaveBeenCalled();
      });
    });

    it('displays results after prediction', async () => {
      const user = userEvent.setup();
      renderPlayground();

      const predictButton = screen.getByRole('button', { name: /Predict/i });
      await user.click(predictButton);

      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });
    });
  });

  describe('chart display', () => {
    it('renders chart container', async () => {
      const user = userEvent.setup();
      renderPlayground();

      const predictButton = screen.getByRole('button', { name: /Predict/i });
      await user.click(predictButton);

      await waitFor(() => {
        expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      });
    });
  });

  describe('accessibility', () => {
    it('sliders are keyboard accessible', () => {
      renderPlayground();
      const sliders = document.querySelectorAll('[role="slider"]');
      sliders.forEach((slider) => {
        expect(slider).toHaveAttribute('tabindex', '0');
      });
    });

    it('predict button is focusable', () => {
      renderPlayground();
      const button = screen.getByRole('button', { name: /Predict/i });
      expect(button).not.toHaveAttribute('tabindex', '-1');
    });

    it('has proper heading hierarchy', () => {
      renderPlayground();
      const h1 = screen.getByRole('heading', { level: 1 });
      expect(h1).toHaveTextContent(/Model Playground/i);
    });
  });
});
