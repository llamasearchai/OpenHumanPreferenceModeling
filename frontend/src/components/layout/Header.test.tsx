/**
 * Header Component Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { act } from '@testing-library/react';
import { Header } from './Header';
import { useUIStore } from '@/stores/ui-store';

const mockNavigate = vi.fn();

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

const renderHeader = (props = {}) => {
  return render(
    <BrowserRouter>
      <Header {...props} />
    </BrowserRouter>
  );
};

describe('Header', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    act(() => {
      useUIStore.getState().reset();
    });
  });

  describe('rendering', () => {
    it('renders header element', () => {
      const { container } = renderHeader();
      expect(container.querySelector('header')).toBeInTheDocument();
    });

    it('renders page title', () => {
      renderHeader();
      expect(screen.getByText('OpenHuman Preference Modeling')).toBeInTheDocument();
    });

    it('renders theme toggle button', () => {
      renderHeader();
      expect(screen.getByRole('button', { name: /toggle theme/i })).toBeInTheDocument();
    });

    it('contains right side actions area', () => {
      const { container } = renderHeader();
      expect(container.querySelector('.flex.items-center.gap-2')).toBeInTheDocument();
    });
  });

  describe('theme toggle', () => {
    it('toggles theme when clicked', async () => {
      const user = userEvent.setup();
      renderHeader();

      const themeButton = screen.getByRole('button', { name: /toggle theme/i });
      const initialTheme = useUIStore.getState().theme;

      await user.click(themeButton);

      const newTheme = useUIStore.getState().theme;
      expect(newTheme).not.toBe(initialTheme);
    });

    it('changes from light to dark', async () => {
      act(() => {
        useUIStore.getState().setTheme('light');
      });

      const user = userEvent.setup();
      renderHeader();

      await user.click(screen.getByRole('button', { name: /toggle theme/i }));

      expect(useUIStore.getState().theme).toBe('dark');
    });

    it('changes from dark to light', async () => {
      act(() => {
        useUIStore.getState().setTheme('dark');
      });

      const user = userEvent.setup();
      renderHeader();

      await user.click(screen.getByRole('button', { name: /toggle theme/i }));

      expect(useUIStore.getState().theme).toBe('light');
    });
  });

  describe('accessibility', () => {
    it('has proper ARIA labels on buttons', () => {
      renderHeader();
      const themeButton = screen.getByRole('button', { name: /toggle theme/i });
      expect(themeButton).toHaveAttribute('aria-label', 'Toggle theme');
    });

    it('renders semantic header element', () => {
      const { container } = renderHeader();
      const header = container.querySelector('header');
      expect(header).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    it('applies custom className', () => {
      const { container } = renderHeader({ className: 'custom-header' });
      expect(container.querySelector('.custom-header')).toBeInTheDocument();
    });

    it('header is sticky', () => {
      const { container } = renderHeader();
      const header = container.querySelector('header');
      expect(header).toHaveClass('sticky');
    });

    it('header has top-0 positioning', () => {
      const { container } = renderHeader();
      const header = container.querySelector('header');
      expect(header).toHaveClass('top-0');
    });
  });
});
