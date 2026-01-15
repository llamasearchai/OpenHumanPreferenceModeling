import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { AppLayout } from './AppLayout';
import * as UIStore from '@/stores/ui-store';

// Mock Header and Sidebar
vi.mock('./Header', () => ({
  Header: () => <div>Header</div>,
}));
vi.mock('./Sidebar', () => ({
  Sidebar: () => <div>Sidebar</div>,
}));

// Mock UI Store
vi.mock('@/stores/ui-store', () => ({
  useUIStore: vi.fn(),
  selectSidebarCollapsed: vi.fn(),
}));

describe('AppLayout', () => {
  beforeEach(() => {
    // Default to false (expanded)
    (UIStore.useUIStore as any).mockReturnValue(false);
  });

  it('renders correctly', () => {
    render(
      <MemoryRouter>
        <AppLayout />
      </MemoryRouter>
    );
    expect(screen.getByText('Header')).toBeInTheDocument();
    expect(screen.getByText('Sidebar')).toBeInTheDocument();
  });

  it('updates title based on route', async () => {
    render(
      <MemoryRouter initialEntries={['/settings']}>
        <AppLayout />
      </MemoryRouter>
    );
    await waitFor(() => {
      expect(document.title).toContain('Settings');
    });
  });

  it('handles default route title', async () => {
    render(
      <MemoryRouter initialEntries={['/']}>
        <AppLayout />
      </MemoryRouter>
    );
    await waitFor(() => {
      expect(document.title).toContain('Dashboard');
    });
  });

  it('handles nested route title', async () => {
    render(
      <MemoryRouter initialEntries={['/annotations/123']}>
        <AppLayout />
      </MemoryRouter>
    );
    await waitFor(() => {
      expect(document.title).toContain('Annotations');
    });
  });
  
  it('handles route with no specific title mapping', async () => {
    render(
      <MemoryRouter initialEntries={['/calendar']}>
        <AppLayout />
      </MemoryRouter>
    );
    // routeTitles missing /calendar, so returns default title
    // "OpenHuman Preference Modeling"
    await waitFor(() => {
      expect(document.title).toBe('OpenHuman Preference Modeling');
    });
  });
  
  it('respects sidebar collapsed state', () => {
    (UIStore.useUIStore as any).mockReturnValue(true); // Collapsed
    const { container } = render(
      <MemoryRouter>
        <AppLayout />
      </MemoryRouter>
    );
    // Check for collapsed margin class 'md:ml-16' and NOT 'lg:ml-64' (if exclusive)
    // The code: sidebarCollapsed ? 'md:ml-16' : 'md:ml-16 lg:ml-64'
    // If collapsed (true): 'md:ml-16'
    // If expanded (false): 'md:ml-16 lg:ml-64'
    
    // We can check if classList contains specific classes or structure.
    const wrapper = container.querySelector('div.transition-all');
    expect(wrapper).toHaveClass('md:ml-16');
    expect(wrapper).not.toHaveClass('lg:ml-64');

    (UIStore.useUIStore as any).mockReturnValue(false); // Expanded
    const { container: container2 } = render(
      <MemoryRouter>
        <AppLayout />
      </MemoryRouter>
    );
    const wrapper2 = container2.querySelector('div.transition-all');
    expect(wrapper2).toHaveClass('md:ml-16'); // Common
    expect(wrapper2).toHaveClass('lg:ml-64'); // Extra margin
  });
});
