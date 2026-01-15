/**
 * Table Component Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
  Table,
  TableHeader,
  TableBody,
  TableFooter,
  TableHead,
  TableRow,
  TableCell,
  TableCaption,
} from './table';

describe('Table', () => {
  const renderTable = () => {
    return render(
      <Table>
        <TableCaption>A list of users</TableCaption>
        <TableHeader>
          <TableRow>
            <TableHead>Name</TableHead>
            <TableHead>Email</TableHead>
            <TableHead>Role</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell>John Doe</TableCell>
            <TableCell>john@example.com</TableCell>
            <TableCell>Admin</TableCell>
          </TableRow>
          <TableRow>
            <TableCell>Jane Smith</TableCell>
            <TableCell>jane@example.com</TableCell>
            <TableCell>User</TableCell>
          </TableRow>
        </TableBody>
        <TableFooter>
          <TableRow>
            <TableCell colSpan={3}>Total: 2 users</TableCell>
          </TableRow>
        </TableFooter>
      </Table>
    );
  };

  describe('Table', () => {
    it('renders table element', () => {
      renderTable();
      expect(screen.getByRole('table')).toBeInTheDocument();
    });

    it('has correct base styling', () => {
      renderTable();
      expect(screen.getByRole('table')).toHaveClass('w-full', 'caption-bottom', 'text-sm');
    });

    it('is wrapped in overflow container', () => {
      const { container } = renderTable();
      const wrapper = container.firstElementChild;
      expect(wrapper).toHaveClass('relative', 'w-full', 'overflow-auto');
    });

    it('supports custom className', () => {
      render(
        <Table className="custom-class">
          <TableBody>
            <TableRow>
              <TableCell>Test</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      );
      expect(screen.getByRole('table')).toHaveClass('custom-class');
    });
  });

  describe('TableHeader', () => {
    it('renders thead element', () => {
      renderTable();
      const thead = document.querySelector('thead');
      expect(thead).toBeInTheDocument();
    });

    it('has border styling', () => {
      renderTable();
      const thead = document.querySelector('thead');
      expect(thead).toHaveClass('[&_tr]:border-b');
    });

    it('supports custom className', () => {
      render(
        <Table>
          <TableHeader className="custom-header">
            <TableRow>
              <TableHead>Header</TableHead>
            </TableRow>
          </TableHeader>
        </Table>
      );
      expect(document.querySelector('thead')).toHaveClass('custom-header');
    });
  });

  describe('TableBody', () => {
    it('renders tbody element', () => {
      renderTable();
      const tbody = document.querySelector('tbody');
      expect(tbody).toBeInTheDocument();
    });

    it('removes border from last row', () => {
      renderTable();
      const tbody = document.querySelector('tbody');
      expect(tbody).toHaveClass('[&_tr:last-child]:border-0');
    });

    it('supports custom className', () => {
      render(
        <Table>
          <TableBody className="custom-body">
            <TableRow>
              <TableCell>Cell</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      );
      expect(document.querySelector('tbody')).toHaveClass('custom-body');
    });
  });

  describe('TableFooter', () => {
    it('renders tfoot element', () => {
      renderTable();
      const tfoot = document.querySelector('tfoot');
      expect(tfoot).toBeInTheDocument();
    });

    it('has correct styling', () => {
      renderTable();
      const tfoot = document.querySelector('tfoot');
      expect(tfoot).toHaveClass('border-t', 'bg-muted/50', 'font-medium');
    });

    it('supports custom className', () => {
      render(
        <Table>
          <TableBody>
            <TableRow>
              <TableCell>Cell</TableCell>
            </TableRow>
          </TableBody>
          <TableFooter className="custom-footer">
            <TableRow>
              <TableCell>Footer</TableCell>
            </TableRow>
          </TableFooter>
        </Table>
      );
      expect(document.querySelector('tfoot')).toHaveClass('custom-footer');
    });
  });

  describe('TableRow', () => {
    it('renders tr element', () => {
      renderTable();
      const rows = screen.getAllByRole('row');
      expect(rows.length).toBeGreaterThan(0);
    });

    it('has hover styling', () => {
      renderTable();
      const row = screen.getAllByRole('row')[1]; // First data row
      expect(row).toHaveClass('hover:bg-muted/50');
    });

    it('has selected state styling', () => {
      render(
        <Table>
          <TableBody>
            <TableRow data-state="selected">
              <TableCell>Selected row</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      );
      const row = screen.getByRole('row');
      expect(row).toHaveClass('data-[state=selected]:bg-muted');
    });

    it('supports custom className', () => {
      render(
        <Table>
          <TableBody>
            <TableRow className="custom-row">
              <TableCell>Cell</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      );
      expect(screen.getByRole('row')).toHaveClass('custom-row');
    });
  });

  describe('TableHead', () => {
    it('renders th element', () => {
      renderTable();
      const headers = screen.getAllByRole('columnheader');
      expect(headers.length).toBe(3);
    });

    it('has correct styling', () => {
      renderTable();
      const header = screen.getByRole('columnheader', { name: 'Name' });
      expect(header).toHaveClass('h-12', 'px-4', 'text-left', 'font-medium');
    });

    it('supports custom className', () => {
      render(
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="custom-head">Header</TableHead>
            </TableRow>
          </TableHeader>
        </Table>
      );
      expect(screen.getByRole('columnheader')).toHaveClass('custom-head');
    });
  });

  describe('TableCell', () => {
    it('renders td element', () => {
      renderTable();
      const cells = screen.getAllByRole('cell');
      expect(cells.length).toBeGreaterThan(0);
    });

    it('has correct styling', () => {
      renderTable();
      const cell = screen.getByRole('cell', { name: 'John Doe' });
      expect(cell).toHaveClass('p-4', 'align-middle');
    });

    it('supports colSpan', () => {
      renderTable();
      const footerCell = screen.getByRole('cell', { name: /Total: 2 users/i });
      expect(footerCell).toHaveAttribute('colspan', '3');
    });

    it('supports custom className', () => {
      render(
        <Table>
          <TableBody>
            <TableRow>
              <TableCell className="custom-cell">Cell</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      );
      expect(screen.getByRole('cell')).toHaveClass('custom-cell');
    });
  });

  describe('TableCaption', () => {
    it('renders caption element', () => {
      renderTable();
      expect(screen.getByText('A list of users')).toBeInTheDocument();
    });

    it('has correct styling', () => {
      renderTable();
      const caption = document.querySelector('caption');
      expect(caption).toHaveClass('mt-4', 'text-sm', 'text-muted-foreground');
    });

    it('supports custom className', () => {
      render(
        <Table>
          <TableCaption className="custom-caption">Caption</TableCaption>
          <TableBody>
            <TableRow>
              <TableCell>Cell</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      );
      expect(document.querySelector('caption')).toHaveClass('custom-caption');
    });
  });

  describe('ref forwarding', () => {
    it('Table forwards ref', () => {
      const ref = { current: null as HTMLTableElement | null };
      render(
        <Table ref={ref}>
          <TableBody>
            <TableRow>
              <TableCell>Cell</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      );
      expect(ref.current).toBeInstanceOf(HTMLTableElement);
    });

    it('TableHeader forwards ref', () => {
      const ref = { current: null as HTMLTableSectionElement | null };
      render(
        <Table>
          <TableHeader ref={ref}>
            <TableRow>
              <TableHead>Header</TableHead>
            </TableRow>
          </TableHeader>
        </Table>
      );
      expect(ref.current).toBeInstanceOf(HTMLTableSectionElement);
    });

    it('TableBody forwards ref', () => {
      const ref = { current: null as HTMLTableSectionElement | null };
      render(
        <Table>
          <TableBody ref={ref}>
            <TableRow>
              <TableCell>Cell</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      );
      expect(ref.current).toBeInstanceOf(HTMLTableSectionElement);
    });

    it('TableRow forwards ref', () => {
      const ref = { current: null as HTMLTableRowElement | null };
      render(
        <Table>
          <TableBody>
            <TableRow ref={ref}>
              <TableCell>Cell</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      );
      expect(ref.current).toBeInstanceOf(HTMLTableRowElement);
    });

    it('TableHead forwards ref', () => {
      const ref = { current: null as HTMLTableCellElement | null };
      render(
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead ref={ref}>Header</TableHead>
            </TableRow>
          </TableHeader>
        </Table>
      );
      expect(ref.current).toBeInstanceOf(HTMLTableCellElement);
    });

    it('TableCell forwards ref', () => {
      const ref = { current: null as HTMLTableCellElement | null };
      render(
        <Table>
          <TableBody>
            <TableRow>
              <TableCell ref={ref}>Cell</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      );
      expect(ref.current).toBeInstanceOf(HTMLTableCellElement);
    });

    it('TableCaption forwards ref', () => {
      const ref = { current: null as HTMLTableCaptionElement | null };
      render(
        <Table>
          <TableCaption ref={ref}>Caption</TableCaption>
          <TableBody>
            <TableRow>
              <TableCell>Cell</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      );
      expect(ref.current).toBeInstanceOf(HTMLTableCaptionElement);
    });
  });

  describe('display names', () => {
    it('all components have correct display names', () => {
      expect(Table.displayName).toBe('Table');
      expect(TableHeader.displayName).toBe('TableHeader');
      expect(TableBody.displayName).toBe('TableBody');
      expect(TableFooter.displayName).toBe('TableFooter');
      expect(TableRow.displayName).toBe('TableRow');
      expect(TableHead.displayName).toBe('TableHead');
      expect(TableCell.displayName).toBe('TableCell');
      expect(TableCaption.displayName).toBe('TableCaption');
    });
  });

  describe('data rendering', () => {
    it('renders data correctly', () => {
      renderTable();
      expect(screen.getByText('John Doe')).toBeInTheDocument();
      expect(screen.getByText('john@example.com')).toBeInTheDocument();
      expect(screen.getByText('Admin')).toBeInTheDocument();
    });

    it('renders empty table', () => {
      render(
        <Table>
          <TableBody data-testid="empty-body"></TableBody>
        </Table>
      );
      expect(screen.getByTestId('empty-body')).toBeInTheDocument();
    });
  });
});
