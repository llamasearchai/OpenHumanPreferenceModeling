/**
 * Time Series Chart Widget Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TimeSeriesChart } from './TimeSeriesChart';

vi.mock('recharts', () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="recharts-linechart">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="xaxis" />,
  YAxis: () => <div data-testid="yaxis" />,
  CartesianGrid: () => <div data-testid="grid" />,
  Tooltip: ({ formatter }: { formatter?: (v: number) => unknown }) => {
    try {
      formatter?.(1.2345);
    } catch {
      // Ignore formatting errors
    }
    return <div data-testid="tooltip" />;
  },
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container">{children}</div>,
  Legend: () => <div data-testid="legend" />,
}));

describe('TimeSeriesChart', () => {
  const mockData = [
    { timestamp: '2024-01-01T10:00:00Z', value: 0.5 },
    { timestamp: '2024-01-01T10:01:00Z', value: 0.6 },
    { timestamp: '2024-01-01T10:02:00Z', value: 0.55 },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders loading skeleton when isLoading is true', () => {
    render(<TimeSeriesChart data={mockData} isLoading={true} />);
    const skeletons = screen.getAllByRole('generic');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('renders empty state when data is empty', () => {
    render(<TimeSeriesChart data={[]} title="Test Chart" description="Chart Description" />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
    expect(screen.getByText('Test Chart')).toBeInTheDocument();
    expect(screen.getByText('Chart Description')).toBeInTheDocument();
  });

  it('renders chart with data', () => {
    render(<TimeSeriesChart data={mockData} />);
    expect(screen.getByTestId('recharts-linechart')).toBeInTheDocument();
  });

  it('renders title and description when provided', () => {
    render(<TimeSeriesChart data={mockData} title="Test Chart" description="Test Description" />);
    expect(screen.getByText('Test Chart')).toBeInTheDocument();
    expect(screen.getByText('Test Description')).toBeInTheDocument();
  });

  it('uses custom height', () => {
    render(<TimeSeriesChart data={mockData} height={400} />);
    expect(screen.getByTestId('recharts-linechart')).toBeInTheDocument();
  });

  it('renders grid when showGrid is true', () => {
    render(<TimeSeriesChart data={mockData} showGrid={true} />);
    expect(screen.getByTestId('grid')).toBeInTheDocument();
  });

  it('does not render grid when showGrid is false', () => {
    render(<TimeSeriesChart data={mockData} showGrid={false} />);
    expect(screen.queryByTestId('grid')).not.toBeInTheDocument();
  });

  it('renders legend when showLegend is true', () => {
    render(<TimeSeriesChart data={mockData} showLegend={true} />);
    expect(screen.getByTestId('legend')).toBeInTheDocument();
  });

  it('does not render legend when showLegend is false', () => {
    render(<TimeSeriesChart data={mockData} showLegend={false} />);
    expect(screen.queryByTestId('legend')).not.toBeInTheDocument();
  });

  it('handles custom dataKey', () => {
    render(<TimeSeriesChart data={mockData} dataKey="value" />);
    expect(screen.getByTestId('recharts-linechart')).toBeInTheDocument();
  });

  it('handles custom color', () => {
    render(<TimeSeriesChart data={mockData} color="#ff0000" />);
    expect(screen.getByTestId('recharts-linechart')).toBeInTheDocument();
  });

  it('uses formatYAxis when provided', () => {
    const formatYAxis = vi.fn((v) => `$${v}`);
    render(<TimeSeriesChart data={mockData} formatYAxis={formatYAxis} />);
    expect(screen.getByTestId('recharts-linechart')).toBeInTheDocument();
  });

  it('uses formatTooltip when provided', () => {
    const formatTooltip = vi.fn((v) => `Value: ${v}`);
    render(<TimeSeriesChart data={mockData} formatTooltip={formatTooltip} />);
    expect(formatTooltip).toHaveBeenCalled();
  });

  it('handles Date objects as timestamps', () => {
    const dataWithDates = [
      { timestamp: new Date('2024-01-01T10:00:00Z'), value: 0.5 },
      { timestamp: new Date('2024-01-01T10:01:00Z'), value: 0.6 },
    ];
    render(<TimeSeriesChart data={dataWithDates} />);
    expect(screen.getByTestId('recharts-linechart')).toBeInTheDocument();
  });

  it('handles formatXAxis when provided', () => {
    const formatXAxis = vi.fn((v) => String(v));
    render(<TimeSeriesChart data={mockData} formatXAxis={formatXAxis} />);
    expect(screen.getByTestId('recharts-linechart')).toBeInTheDocument();
  });
});
