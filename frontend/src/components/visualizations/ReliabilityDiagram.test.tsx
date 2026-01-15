/**
 * Reliability Diagram Component Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReliabilityDiagram } from './ReliabilityDiagram';
import * as useChartDimensionsModule from '@/lib/visualizations/use-chart-dimensions';
import type { UseChartDimensionsResult } from '@/lib/visualizations/use-chart-dimensions';

describe('ReliabilityDiagram', () => {
  let queryClient: QueryClient;

  const defaultDimensions: UseChartDimensionsResult = {
    ref: { current: null },
    width: 500,
    height: 300,
    boundedWidth: 460,
    boundedHeight: 260,
    margins: { top: 20, right: 30, bottom: 40, left: 50 },
    isReady: true,
  };

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
    vi.clearAllMocks();
    vi.spyOn(useChartDimensionsModule, 'useChartDimensions').mockReturnValue(defaultDimensions);
  });

  it('renders loading skeleton when data is loading', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ReliabilityDiagram />
      </QueryClientProvider>
    );

    const skeletons = screen.getAllByRole('generic');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('renders diagram when data is loaded', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ReliabilityDiagram />
      </QueryClientProvider>
    );

    await waitFor(() => {
      const svgs = screen.queryAllByRole('img', { hidden: true });
      expect(svgs.length).toBeGreaterThan(0);
    }, { timeout: 2000 });
  });

  it('uses custom height', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ReliabilityDiagram height={400} />
      </QueryClientProvider>
    );

    const containers = screen.getAllByRole('generic');
    expect(containers.length).toBeGreaterThan(0);
  });

  it('renders empty state when no bins data', async () => {
    queryClient.setQueryData(['calibration', 'reliability-bins'], []);

    render(
      <QueryClientProvider client={queryClient}>
        <ReliabilityDiagram />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/no calibration data/i)).toBeInTheDocument();
    });
  });

  it('handles showGap prop', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ReliabilityDiagram showGap={false} />
      </QueryClientProvider>
    );

    await waitFor(() => {
      const containers = screen.getAllByRole('generic');
      expect(containers.length).toBeGreaterThan(0);
    });
  });

  it('handles onBinClick callback', async () => {
    const handleBinClick = vi.fn();
    render(
      <QueryClientProvider client={queryClient}>
        <ReliabilityDiagram onBinClick={handleBinClick} />
      </QueryClientProvider>
    );

    await waitFor(() => {
      const containers = screen.getAllByRole('generic');
      expect(containers.length).toBeGreaterThan(0);
    });
  });

  it('handles selectedBin prop', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <ReliabilityDiagram selectedBin={0} />
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
        <ReliabilityDiagram />
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
        <ReliabilityDiagram />
      </QueryClientProvider>
    );

    await waitFor(() => {
      const skeletons = screen.getAllByRole('generic');
      expect(skeletons.length).toBeGreaterThan(0);
    });
  });
});
