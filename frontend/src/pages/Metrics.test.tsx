import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import MetricsPage from './Metrics';
import { monitoringApi } from '@/lib/api-client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock TimeSeriesChart widget
vi.mock('@/components/widgets/TimeSeriesChart', () => ({
  TimeSeriesChart: ({ title, isLoading, data }: { title?: string; isLoading?: boolean; data?: unknown[] }) => (
    <div data-testid="time-series-chart">
      {isLoading ? 'Loading...' : data && data.length > 0 ? title || 'Chart' : 'No data available'}
    </div>
  ),
}));

// Mock MetricCard widget
vi.mock('@/components/widgets/MetricCard', () => ({
  MetricCard: ({ title, value, isLoading }: { title: string; value: string | number; isLoading?: boolean }) => (
    <div data-testid={`metric-card-${title.toLowerCase().replace(/\s+/g, '-')}`}>
      {isLoading ? 'Loading...' : `${title}: ${value}`}
    </div>
  ),
}));

const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      gcTime: 0,
    },
  },
});

function renderWithClient(ui: React.ReactElement) {
  const testClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={testClient}>{ui}</QueryClientProvider>
  );
}

describe('MetricsPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders loading state initially', async () => {
    const getMetricsSpy = vi.spyOn(monitoringApi, 'getMetrics');
    // Mock to never resolve
    getMetricsSpy.mockImplementation(() => new Promise(() => {}));
    
    renderWithClient(<MetricsPage />);
    
    expect(screen.getByText('Metrics')).toBeInTheDocument();
    expect(screen.getByText('Monitor system performance and health')).toBeInTheDocument();
    // Multiple MetricCards show "Loading..." when isLoading is true
    const loadingElements = screen.getAllByText(/Loading/i);
    expect(loadingElements.length).toBeGreaterThan(0);
    // Verify at least one metric card is in loading state
    expect(screen.getByTestId('metric-card-current')).toHaveTextContent('Loading...');
  });

  it('renders data and chart with accessibility attributes', async () => {
    const mockData = [
      {
        timestamp: '2024-01-01T10:00:00Z',
        value: 0.5,
        name: 'test_metric',
        tags: { env: 'prod' },
      },
      {
        timestamp: '2024-01-01T10:01:00Z',
        value: 0.6,
        name: 'test_metric',
        tags: { env: 'prod' },
      },
    ];

    const getMetricsSpy = vi.spyOn(monitoringApi, 'getMetrics');
    getMetricsSpy.mockResolvedValue({
      success: true,
      data: mockData,
    });

    renderWithClient(<MetricsPage />);

    // Verify data appears
    await waitFor(() => {
      expect(screen.getByTestId('metric-card-current')).toBeInTheDocument();
    });

    expect(getMetricsSpy).toHaveBeenCalled();

    // Check for chart widget
    await waitFor(() => {
      const chart = screen.getByTestId('time-series-chart');
      expect(chart).toBeInTheDocument();
    });

    // Check for table caption
    await waitFor(() => {
      expect(screen.getByText(/Table showing the last/i)).toBeInTheDocument();
    });

    // Check table headers have scope
    const headers = screen.getAllByRole('columnheader');
    headers.forEach((header) => {
      expect(header).toHaveAttribute('scope', 'col');
    });
  });

  it('handles empty data state', async () => {
    const getMetricsSpy = vi.spyOn(monitoringApi, 'getMetrics');
    getMetricsSpy.mockResolvedValue({
      success: true,
      data: [],
    });

    renderWithClient(<MetricsPage />);

    await waitFor(() => {
      // Both TimeSeriesChart and table show "No data available" - check for chart specifically
      const chart = screen.getByTestId('time-series-chart');
      expect(chart).toHaveTextContent('No data available');
    });
  });

  it('handles error state', async () => {
    const getMetricsSpy = vi.spyOn(monitoringApi, 'getMetrics');
    getMetricsSpy.mockResolvedValue({
      success: false,
      error: { 
        type: 'error',
        title: 'Error',
        status: 500,
        code: 'ERROR',
        detail: 'API Error' 
      },
    });

    renderWithClient(<MetricsPage />);

    // Error state - metrics will be empty/undefined, so TimeSeriesChart shows "No data available"
    await waitFor(() => {
      // The page should still render, but with no data
      expect(screen.getByText('Metrics')).toBeInTheDocument();
    }, { timeout: 2000 });
  });
});
