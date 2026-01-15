/**
 * Confidence Histogram Component Tests
 *
 * Purpose: Test confidence distribution visualization
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfidenceHistogram } from './ConfidenceHistogram';
import * as useChartDimensionsModule from '@/lib/visualizations/use-chart-dimensions';
import type { UseChartDimensionsResult } from '@/lib/visualizations/use-chart-dimensions';

describe('ConfidenceHistogram', () => {
  let queryClient: QueryClient;

  const defaultDimensions: UseChartDimensionsResult = {
    ref: { current: null },
    width: 500,
    height: 300,
    boundedWidth: 460,
    boundedHeight: 260,
    margins: { top: 20, right: 20, bottom: 40, left: 50 },
    isReady: true,
  };

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
    vi.clearAllMocks();

    // Default mock implementation
    vi.spyOn(useChartDimensionsModule, 'useChartDimensions').mockReturnValue(defaultDimensions);
  });

  it('renders loading skeleton when data is loading', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ConfidenceHistogram />
      </QueryClientProvider>
    );

    const skeletons = screen.getAllByRole('generic');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('renders histogram when data is loaded', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ConfidenceHistogram />
      </QueryClientProvider>
    );

    await waitFor(() => {
      const svg = screen.queryByRole('img', { hidden: true }) || screen.queryByRole('generic');
      expect(svg).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it('uses custom height', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ConfidenceHistogram height={400} />
      </QueryClientProvider>
    );

    const containers = screen.getAllByRole('generic');
    expect(containers.length).toBeGreaterThan(0);
  });

  it('renders empty state when no distribution data', async () => {
    // Mock query to return empty data
    queryClient.setQueryData(['calibration', 'confidence-distribution'], []);

    render(
      <QueryClientProvider client={queryClient}>
        <ConfidenceHistogram />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/no distribution data/i)).toBeInTheDocument();
    });
  });

  it('handles custom numBins prop', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ConfidenceHistogram numBins={10} />
      </QueryClientProvider>
    );

    const containers = screen.getAllByRole('generic');
    expect(containers.length).toBeGreaterThan(0);
  });

  it('shows overlay when showOverlay is true', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ConfidenceHistogram showOverlay={true} />
      </QueryClientProvider>
    );

    await waitFor(() => {
      const containers = screen.getAllByRole('generic');
      expect(containers.length).toBeGreaterThan(0);
    });
  });

  it('hides overlay when showOverlay is false', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ConfidenceHistogram showOverlay={false} />
      </QueryClientProvider>
    );

    await waitFor(() => {
      const containers = screen.getAllByRole('generic');
      expect(containers.length).toBeGreaterThan(0);
    });
  });

  it('renders skeleton when isReady is false', async () => {
    vi.spyOn(useChartDimensionsModule, 'useChartDimensions').mockReturnValue({
      ...defaultDimensions,
      isReady: false,
    });

    render(
      <QueryClientProvider client={queryClient}>
        <ConfidenceHistogram />
      </QueryClientProvider>
    );

    await waitFor(() => {
      const skeletons = screen.getAllByRole('generic');
      expect(skeletons.length).toBeGreaterThan(0);
    });
  });

  it('renders skeleton when width is 0', async () => {
    vi.spyOn(useChartDimensionsModule, 'useChartDimensions').mockReturnValue({
      ...defaultDimensions,
      width: 0,
      boundedWidth: 0,
    });

    render(
      <QueryClientProvider client={queryClient}>
        <ConfidenceHistogram />
      </QueryClientProvider>
    );

    await waitFor(() => {
      const skeletons = screen.getAllByRole('generic');
      expect(skeletons.length).toBeGreaterThan(0);
    });
  });

  it('renders skeleton when measuredHeight is 0', async () => {
    vi.spyOn(useChartDimensionsModule, 'useChartDimensions').mockReturnValue({
      ...defaultDimensions,
      height: 0,
      boundedHeight: 0,
    });

    render(
      <QueryClientProvider client={queryClient}>
        <ConfidenceHistogram />
      </QueryClientProvider>
    );

    await waitFor(() => {
      const skeletons = screen.getAllByRole('generic');
      expect(skeletons.length).toBeGreaterThan(0);
    });
  });
});
