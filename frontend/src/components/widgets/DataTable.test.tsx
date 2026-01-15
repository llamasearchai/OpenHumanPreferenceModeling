/**
 * Data Table Widget Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DataTable, type Column } from './DataTable';

interface TestData {
  id: number;
  name: string;
  value: number;
  status: string;
}

describe('DataTable', () => {
  const mockData: TestData[] = [
    { id: 1, name: 'Item A', value: 100, status: 'active' },
    { id: 2, name: 'Item B', value: 200, status: 'inactive' },
    { id: 3, name: 'Item C', value: 150, status: 'active' },
  ];

  const columns: Column<TestData>[] = [
    { key: 'id', header: 'ID', sortable: true },
    { key: 'name', header: 'Name', sortable: true },
    { key: 'value', header: 'Value', sortable: true },
    { key: 'status', header: 'Status' },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders loading skeleton when isLoading is true', () => {
    render(<DataTable data={mockData} columns={columns} isLoading={true} />);
    const skeletons = screen.getAllByRole('generic');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('renders table with data', () => {
    render(<DataTable data={mockData} columns={columns} />);
    expect(screen.getByText('Item A')).toBeInTheDocument();
    expect(screen.getByText('Item B')).toBeInTheDocument();
    expect(screen.getByText('Item C')).toBeInTheDocument();
  });

  it('renders title and description when provided', () => {
    render(<DataTable data={mockData} columns={columns} title="Test Table" description="Test Description" />);
    expect(screen.getByText('Test Table')).toBeInTheDocument();
    expect(screen.getByText('Test Description')).toBeInTheDocument();
  });

  it('renders search input when searchable is true', () => {
    render(<DataTable data={mockData} columns={columns} searchable={true} />);
    expect(screen.getByPlaceholderText('Search...')).toBeInTheDocument();
  });

  it('does not render search when searchable is false', () => {
    render(<DataTable data={mockData} columns={columns} searchable={false} />);
    expect(screen.queryByPlaceholderText('Search...')).not.toBeInTheDocument();
  });

  it('filters data based on search query', async () => {
    const user = userEvent.setup();
    render(<DataTable data={mockData} columns={columns} searchable={true} />);
    
    const searchInput = screen.getByPlaceholderText('Search...');
    await user.type(searchInput, 'Item A');
    
    expect(screen.getByText('Item A')).toBeInTheDocument();
    expect(screen.queryByText('Item B')).not.toBeInTheDocument();
    expect(screen.queryByText('Item C')).not.toBeInTheDocument();
  });

  it('sorts data when sortable column header is clicked', async () => {
    const user = userEvent.setup();
    render(<DataTable data={mockData} columns={columns} />);
    
    const valueHeader = screen.getByText('Value');
    await user.click(valueHeader);
    
    const rows = screen.getAllByRole('row');
    expect(rows.length).toBeGreaterThan(1);
  });

  it('handles pagination', () => {
    const largeData = Array.from({ length: 25 }, (_, i) => ({
      id: i + 1,
      name: `Item ${i + 1}`,
      value: i * 10,
      status: 'active',
    }));

    render(
      <DataTable
        data={largeData}
        columns={columns}
        pagination={{ pageSize: 10 }}
      />
    );

    expect(screen.getByText(/Showing 1 to 10 of 25 results/i)).toBeInTheDocument();
    expect(screen.getByText(/Page 1 of 3/i)).toBeInTheDocument();
  });

  it('handles page navigation', async () => {
    const user = userEvent.setup();
    const largeData = Array.from({ length: 25 }, (_, i) => ({
      id: i + 1,
      name: `Item ${i + 1}`,
      value: i * 10,
      status: 'active',
    }));

    render(
      <DataTable
        data={largeData}
        columns={columns}
        pagination={{ pageSize: 10 }}
      />
    );

    expect(screen.getByText(/Page 1 of 3/i)).toBeInTheDocument();
    
    const buttons = screen.getAllByRole('button');
    const nextButton = buttons[buttons.length - 1];
    expect(nextButton).toBeInTheDocument();
    expect(nextButton).not.toHaveAttribute('disabled');
    
    await user.click(nextButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Page 2 of 3/i)).toBeInTheDocument();
    });
  });

  it('handles page size selector', async () => {
    const user = userEvent.setup();
    render(
      <DataTable
        data={mockData}
        columns={columns}
        pagination={{ pageSize: 10, showSizeSelector: true }}
      />
    );

    const sizeSelector = screen.getByRole('combobox');
    await user.click(sizeSelector);
    
    const option25 = screen.getByText('25');
    await user.click(option25);

    expect(screen.getByText(/Page 1 of 1/i)).toBeInTheDocument();
  });

  it('calls onRowClick when row is clicked', async () => {
    const user = userEvent.setup();
    const handleRowClick = vi.fn();
    render(<DataTable data={mockData} columns={columns} onRowClick={handleRowClick} />);
    
    const firstRow = screen.getByText('Item A').closest('tr');
    expect(firstRow).toBeInTheDocument();
    if (firstRow) {
      await user.click(firstRow);
      expect(handleRowClick).toHaveBeenCalledWith(mockData[0]);
    }
  });

  it('renders empty message when no data', () => {
    render(<DataTable data={[]} columns={columns} emptyMessage="No items found" />);
    expect(screen.getByText('No items found')).toBeInTheDocument();
  });

  it('uses custom render function for columns', () => {
    const columnsWithRender: Column<TestData>[] = [
      {
        key: 'status',
        header: 'Status',
        render: (value) => <span className="font-bold">{String(value).toUpperCase()}</span>,
      },
    ];

    render(<DataTable data={mockData} columns={columnsWithRender} />);
    const activeElements = screen.getAllByText('ACTIVE');
    expect(activeElements.length).toBeGreaterThan(0);
  });
});
