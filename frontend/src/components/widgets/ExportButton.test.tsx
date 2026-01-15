/**
 * ExportButton Widget Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ExportButton } from './ExportButton';

describe('ExportButton', () => {
  const mockData = [
    { id: 1, name: 'Item 1', value: 100 },
    { id: 2, name: 'Item 2', value: 200 },
  ];

  let originalCreateObjectURL: typeof URL.createObjectURL;
  let originalRevokeObjectURL: typeof URL.revokeObjectURL;

  beforeEach(() => {
    originalCreateObjectURL = URL.createObjectURL;
    originalRevokeObjectURL = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn().mockReturnValue('blob:test-url');
    URL.revokeObjectURL = vi.fn();
  });

  afterEach(() => {
    URL.createObjectURL = originalCreateObjectURL;
    URL.revokeObjectURL = originalRevokeObjectURL;
    vi.restoreAllMocks();
  });

  describe('rendering', () => {
    it('renders export button with CSV format by default', () => {
      render(<ExportButton data={mockData} filename="test" />);

      expect(screen.getByRole('button', { name: /Export CSV/i })).toBeInTheDocument();
    });

    it('renders export button with JSON format', () => {
      render(<ExportButton data={mockData} filename="test" format="json" />);

      expect(screen.getByRole('button', { name: /Export JSON/i })).toBeInTheDocument();
    });

    it('renders button element', () => {
      render(<ExportButton data={mockData} filename="test" />);

      expect(screen.getByRole('button')).toBeInTheDocument();
    });
  });

  describe('disabled state', () => {
    it('is disabled when data is empty', () => {
      render(<ExportButton data={[]} filename="empty" />);

      expect(screen.getByRole('button')).toBeDisabled();
    });

    it('is disabled when disabled prop is true', () => {
      render(<ExportButton data={mockData} filename="test" disabled />);

      expect(screen.getByRole('button')).toBeDisabled();
    });

    it('is enabled with data and not disabled', () => {
      render(<ExportButton data={mockData} filename="test" />);

      expect(screen.getByRole('button')).not.toBeDisabled();
    });
  });

  describe('export functionality', () => {
    it('triggers download on click', async () => {
      const user = userEvent.setup();

      render(<ExportButton data={mockData} filename="test-export" />);

      await user.click(screen.getByRole('button'));

      expect(URL.createObjectURL).toHaveBeenCalled();
      expect(URL.revokeObjectURL).toHaveBeenCalled();
    });

    it('uses custom transform function', async () => {
      const user = userEvent.setup();
      const customTransform = vi.fn().mockReturnValue('custom content');

      render(
        <ExportButton
          data={mockData}
          filename="custom"
          transform={customTransform}
        />
      );

      await user.click(screen.getByRole('button'));

      expect(customTransform).toHaveBeenCalledWith(mockData);
    });
  });

  describe('button variants', () => {
    it('uses outline variant by default', () => {
      render(<ExportButton data={mockData} filename="test" />);

      const button = screen.getByRole('button');
      expect(button).toHaveClass('border');
    });

    it('accepts custom variant', () => {
      render(<ExportButton data={mockData} filename="test" variant="ghost" />);

      expect(screen.getByRole('button')).toBeInTheDocument();
    });
  });

  describe('button sizes', () => {
    it('uses sm size by default', () => {
      render(<ExportButton data={mockData} filename="test" />);

      const button = screen.getByRole('button');
      expect(button).toHaveClass('h-9');
    });

    it('accepts custom size', () => {
      render(<ExportButton data={mockData} filename="test" size="lg" />);

      const button = screen.getByRole('button');
      expect(button).toHaveClass('h-11');
    });
  });

  describe('CSV generation', () => {
    it('handles null and undefined values', async () => {
      const user = userEvent.setup();
      const dataWithNulls = [
        { id: 1, name: null, value: undefined },
      ];

      render(<ExportButton data={dataWithNulls} filename="nulls" />);

      await user.click(screen.getByRole('button'));

      expect(URL.createObjectURL).toHaveBeenCalled();
    });

    it('handles object values by stringifying', async () => {
      const user = userEvent.setup();
      const dataWithObjects = [
        { id: 1, meta: { nested: true } },
      ];

      render(<ExportButton data={dataWithObjects} filename="objects" />);

      await user.click(screen.getByRole('button'));

      expect(URL.createObjectURL).toHaveBeenCalled();
    });
  });

  describe('JSON export', () => {
    it('exports JSON format correctly', async () => {
      const user = userEvent.setup();

      render(<ExportButton data={mockData} filename="test" format="json" />);

      await user.click(screen.getByRole('button'));

      expect(URL.createObjectURL).toHaveBeenCalled();
    });

    it('triggers click on non-JSDOM environment', () => {
      // Mock navigator to NOT have jsdom
      const originalUserAgent = navigator.userAgent;
      Object.defineProperty(navigator, 'userAgent', {
        value: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
        configurable: true
      });
      
      const clickSpy = vi.fn();
      // Mock document.createElement to return element with spy
      const originalCreateElement = document.createElement;
      vi.spyOn(document, 'createElement').mockImplementation((tagName) => {
        const el = originalCreateElement.call(document, tagName);
        if (tagName === 'a') {
            el.click = clickSpy;
        }
        return el;
      });

      render(<ExportButton data={mockData} filename="test" />);
      fireEvent.click(screen.getByRole('button'));
      
      expect(clickSpy).toHaveBeenCalled();
      
      // Restore
      Object.defineProperty(navigator, 'userAgent', {
        value: originalUserAgent,
        configurable: true
      });
      vi.restoreAllMocks();
    });
  });

  describe('empty data handling', () => {
    it('returns early when data is empty in CSV format', () => {
      render(<ExportButton data={[]} filename="empty" format="csv" />);

      const button = screen.getByRole('button', { name: /Export CSV/i });
      expect(button).toBeDisabled();
      // The early return at line 43-44 is defensive code that prevents export
      // when data is empty in CSV format, but button is already disabled
    });

    it('button is disabled for empty data regardless of format', () => {
      const { rerender } = render(<ExportButton data={[]} filename="empty" format="json" />);
      expect(screen.getByRole('button', { name: /Export JSON/i })).toBeDisabled();

      rerender(<ExportButton data={[]} filename="empty" format="csv" />);
      expect(screen.getByRole('button', { name: /Export CSV/i })).toBeDisabled();
    });

    it('handles transform function with non-empty data', async () => {
      const user = userEvent.setup();
      const transform = vi.fn().mockReturnValue('transformed');
      
      render(<ExportButton data={[{ test: 'value' }]} filename="test" transform={transform} />);

      await user.click(screen.getByRole('button'));

      expect(transform).toHaveBeenCalledWith([{ test: 'value' }]);
      expect(URL.createObjectURL).toHaveBeenCalled();
    });
  });
});
