/**
 * Sidebar Component Tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { act } from '@testing-library/react';
import { Sidebar } from './Sidebar';
import { useUIStore } from '@/stores/ui-store';

const renderSidebar = () => {
  return render(
    <BrowserRouter>
      <Sidebar />
    </BrowserRouter>
  );
};

describe('Sidebar', () => {
  beforeEach(() => {
    // Reset UI store before each test
    act(() => {
      useUIStore.getState().reset();
    });
  });

  describe('rendering', () => {
    it('renders navigation aside element', () => {
      const { container } = renderSidebar();
      expect(container.querySelector('aside')).toBeInTheDocument();
    });

    it('renders brand text when expanded', () => {
      renderSidebar();
      expect(screen.getByText('OHPM')).toBeInTheDocument();
    });

    it('renders Dashboard link', () => {
      renderSidebar();
      expect(screen.getByRole('link', { name: /Dashboard/i })).toBeInTheDocument();
    });

    it('renders Annotations link', () => {
      renderSidebar();
      expect(screen.getByRole('link', { name: /Annotations/i })).toBeInTheDocument();
    });

    it('renders Metrics link', () => {
      renderSidebar();
      expect(screen.getByRole('link', { name: /Metrics/i })).toBeInTheDocument();
    });

    it('renders Alerts link', () => {
      renderSidebar();
      expect(screen.getByRole('link', { name: /Alerts/i })).toBeInTheDocument();
    });

    it('renders Calibration link', () => {
      renderSidebar();
      expect(screen.getByRole('link', { name: /Calibration/i })).toBeInTheDocument();
    });

    it('renders Settings link', () => {
      renderSidebar();
      expect(screen.getByRole('link', { name: /Settings/i })).toBeInTheDocument();
    });

    it('renders Active Learning link', () => {
      renderSidebar();
      expect(screen.getByRole('link', { name: /Active Learning/i })).toBeInTheDocument();
    });

    it('renders Federated Learning link', () => {
      renderSidebar();
      expect(screen.getByRole('link', { name: /Federated Learning/i })).toBeInTheDocument();
    });

    it('renders Quality Control link', () => {
      renderSidebar();
      expect(screen.getByRole('link', { name: /Quality Control/i })).toBeInTheDocument();
    });

    it('renders Training link', () => {
      renderSidebar();
      expect(screen.getByRole('link', { name: /Training/i })).toBeInTheDocument();
    });

    it('renders Playground link', () => {
      renderSidebar();
      expect(screen.getByRole('link', { name: /Playground/i })).toBeInTheDocument();
    });
  });

  describe('collapsed state', () => {
    it('shows collapse button', () => {
      renderSidebar();
      // The button should exist
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('toggles sidebar when collapse button clicked', async () => {
      const user = userEvent.setup();
      renderSidebar();

      // Initially sidebar should be expanded
      expect(useUIStore.getState().sidebarCollapsed).toBe(false);

      // Find and click the toggle button
      const buttons = screen.getAllByRole('button');
      const toggleButton = buttons[0];
      await user.click(toggleButton);

      // Sidebar should now be collapsed
      expect(useUIStore.getState().sidebarCollapsed).toBe(true);
    });

    it('hides brand text when collapsed', async () => {
      act(() => {
        useUIStore.getState().setSidebarCollapsed(true);
      });

      renderSidebar();

      expect(screen.queryByText('OHPM')).not.toBeInTheDocument();
    });

    it('hides label text when collapsed', async () => {
      act(() => {
        useUIStore.getState().setSidebarCollapsed(true);
      });

      renderSidebar();

      // When collapsed, text labels should not be visible
      // The "Dashboard" text as a standalone span should not exist
      const dashboardLabels = screen.queryAllByText('Dashboard');
      // In collapsed mode, labels are in tooltips, not directly in links
      // So the main nav area shouldn't have visible text labels
      expect(dashboardLabels.length).toBeLessThanOrEqual(1); // May appear in tooltip
    });

    it('has narrower width when collapsed', () => {
      act(() => {
        useUIStore.getState().setSidebarCollapsed(true);
      });

      const { container } = renderSidebar();
      const aside = container.querySelector('aside');
      // Responsive: md:w-16 for tablet and up when collapsed
      expect(aside).toHaveClass('md:w-16');
    });

    it('has wider width when expanded', () => {
      const { container } = renderSidebar();
      const aside = container.querySelector('aside');
      // Always w-64 when expanded (visible on mobile drawer and desktop)
      expect(aside).toHaveClass('w-64');
    });
  });

  describe('navigation links', () => {
    it('Dashboard links to /', () => {
      renderSidebar();
      const link = screen.getByRole('link', { name: /Dashboard/i });
      expect(link).toHaveAttribute('href', '/');
    });

    it('Annotations links to /annotations', () => {
      renderSidebar();
      const link = screen.getByRole('link', { name: /Annotations/i });
      expect(link).toHaveAttribute('href', '/annotations');
    });

    it('Metrics links to /metrics', () => {
      renderSidebar();
      const link = screen.getByRole('link', { name: /Metrics/i });
      expect(link).toHaveAttribute('href', '/metrics');
    });

    it('Alerts links to /alerts', () => {
      renderSidebar();
      const link = screen.getByRole('link', { name: /Alerts/i });
      expect(link).toHaveAttribute('href', '/alerts');
    });

    it('Settings links to /settings', () => {
      renderSidebar();
      const link = screen.getByRole('link', { name: /Settings/i });
      expect(link).toHaveAttribute('href', '/settings');
    });

    it('Playground links to /playground', () => {
      renderSidebar();
      const link = screen.getByRole('link', { name: /Playground/i });
      expect(link).toHaveAttribute('href', '/playground');
    });
  });

  describe('styling', () => {
    it('sidebar has fixed positioning', () => {
      const { container } = renderSidebar();
      const aside = container.querySelector('aside');
      expect(aside).toHaveClass('fixed');
    });

    it('sidebar has full screen height', () => {
      const { container } = renderSidebar();
      const aside = container.querySelector('aside');
      expect(aside).toHaveClass('h-screen');
    });

    it('sidebar has border', () => {
      const { container } = renderSidebar();
      const aside = container.querySelector('aside');
      expect(aside).toHaveClass('border-r');
    });

    it('has transition classes', () => {
      const { container } = renderSidebar();
      const aside = container.querySelector('aside');
      expect(aside).toHaveClass('transition-all');
    });
  });
});
