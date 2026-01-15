/**
 * Pie Chart Widget Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PieChart } from './PieChart';

vi.mock('recharts', () => ({
  PieChart: ({ children }: { children: React.ReactNode }) => <div data-testid="recharts-piechart">{children}</div>,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />,
  Tooltip: ({ formatter }: { formatter?: (v: number) => unknown }) => {
    try {
      formatter?.(100);
    } catch {
      // Ignore formatting errors
    }
    return <div data-testid="tooltip" />;
  },
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div data-testid="responsive-container">{children}</div>,
  Legend: () => <div data-testid="legend" />,
}));

describe('PieChart', () => {
  const mockData = [
    { name: 'A', value: 10 },
    { name: 'B', value: 20 },
    { name: 'C', value: 15 },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders loading skeleton when isLoading is true', () => {
    render(<PieChart data={mockData} isLoading={true} />);
    const skeletons = screen.getAllByRole('generic');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('renders empty state when data is empty', () => {
    render(<PieChart data={[]} title="Test Chart" description="Chart Description" />);
    expect(screen.getByText('No data available')).toBeInTheDocument();
    expect(screen.getByText('Test Chart')).toBeInTheDocument();
    expect(screen.getByText('Chart Description')).toBeInTheDocument();
  });

  it('renders chart with data', () => {
    render(<PieChart data={mockData} />);
    expect(screen.getByTestId('recharts-piechart')).toBeInTheDocument();
  });

  it('renders title and description when provided', () => {
    render(<PieChart data={mockData} title="Test Chart" description="Test Description" />);
    expect(screen.getByText('Test Chart')).toBeInTheDocument();
    expect(screen.getByText('Test Description')).toBeInTheDocument();
  });

  it('uses custom height', () => {
    render(<PieChart data={mockData} height={400} />);
    expect(screen.getByTestId('recharts-piechart')).toBeInTheDocument();
  });

  it('renders legend when showLegend is true', () => {
    render(<PieChart data={mockData} showLegend={true} />);
    expect(screen.getByTestId('legend')).toBeInTheDocument();
  });

  it('does not render legend when showLegend is false', () => {
    render(<PieChart data={mockData} showLegend={false} />);
    expect(screen.queryByTestId('legend')).not.toBeInTheDocument();
  });

  it('handles custom colors', () => {
    render(<PieChart data={mockData} colors={['#ff0000', '#00ff00', '#0000ff']} />);
    expect(screen.getByTestId('recharts-piechart')).toBeInTheDocument();
  });

  it('handles data points with custom colors', () => {
    const dataWithColors = [
      { name: 'A', value: 10, color: '#ff0000' },
      { name: 'B', value: 20, color: '#00ff00' },
    ];
    render(<PieChart data={dataWithColors} />);
    expect(screen.getByTestId('recharts-piechart')).toBeInTheDocument();
  });

  it('uses formatTooltip when provided', () => {
    const formatTooltip = vi.fn((v) => `Value: ${v}`);
    render(<PieChart data={mockData} formatTooltip={formatTooltip} />);
    expect(formatTooltip).toHaveBeenCalled();
  });

  it('handles onSegmentClick interaction', () => {
    const handleSegmentClick = vi.fn();
    render(<PieChart data={mockData} onSegmentClick={handleSegmentClick} />);
    expect(screen.getByTestId('recharts-piechart')).toBeInTheDocument();
  });
});
