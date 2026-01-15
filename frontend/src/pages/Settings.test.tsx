/**
 * Settings Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import SettingsPage from './Settings';
import { settingsApi } from '@/lib/api-client';

vi.mock('@/lib/api-client', () => ({
  settingsApi: {
    get: vi.fn(),
    update: vi.fn(),
  },
}));

vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}));

const mockSettings = {
  company_name: 'Test Company',
  company_phone: '+1 (555) 123-4567',
  address: '123 Test St',
  city: 'Test City',
  state: 'TS',
  zip_code: '12345',
  domain: 'https://test.example.com',
  allowed_file_types: '.pdf, .csv',
  site_direction: 'ltr' as const,
  footer_info: 'Test Footer',
};

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const renderSettings = () => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <SettingsPage />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('SettingsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(settingsApi.get).mockResolvedValue({
      success: true,
      data: mockSettings,
    });
    vi.mocked(settingsApi.update).mockResolvedValue({
      success: true,
      data: mockSettings,
    });
  });

  describe('rendering', () => {
    it('renders settings heading', async () => {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByRole('heading', { name: /Settings/i })).toBeInTheDocument();
      });
    });

    it('renders general information section', async () => {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByText(/General Information/i)).toBeInTheDocument();
      });
    });

    it('renders subtitle', async () => {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByText(/Manage your company and site preferences/i)).toBeInTheDocument();
      });
    });
  });

  describe('form inputs', () => {
    it('renders company name input', async () => {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByLabelText(/Company Name/i)).toBeInTheDocument();
      });
    });

    it('renders phone input', async () => {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByLabelText(/Phone/i)).toBeInTheDocument();
      });
    });

    it('renders address input', async () => {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByLabelText(/Address/i)).toBeInTheDocument();
      });
    });

    it('renders city input', async () => {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByLabelText(/City/i)).toBeInTheDocument();
      });
    });

    it('renders state input', async () => {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByLabelText(/State/i)).toBeInTheDocument();
      });
    });

    it('renders site direction radio group', async () => {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByText(/Site Direction/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/LTR/i)).toBeInTheDocument();
        expect(screen.getByLabelText(/RTL/i)).toBeInTheDocument();
      });
    });
  });

  describe('form interaction', () => {
    it('allows editing company name', async () => {
      const user = userEvent.setup();
      renderSettings();
      
      await waitFor(() => {
        expect(screen.getByLabelText(/Company Name/i)).toBeInTheDocument();
      });
      
      const input = screen.getByLabelText(/Company Name/i) as HTMLInputElement;
      await user.clear(input);
      await user.type(input, 'New Company Name');
      
      expect(input.value).toBe('New Company Name');
    });

    it('submits form with updated data', async () => {
      const user = userEvent.setup();
      renderSettings();
      
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Save Changes/i })).toBeInTheDocument();
      });
      
      const saveButton = screen.getByRole('button', { name: /Save Changes/i });
      await user.click(saveButton);
      
      await waitFor(() => {
        expect(settingsApi.update).toHaveBeenCalled();
      });
    });
  });

  describe('cards', () => {
    it('renders card sections', async () => {
      renderSettings();
      await waitFor(() => {
        const cards = document.querySelectorAll('[class*="card"]');
        expect(cards.length).toBeGreaterThan(0);
      });
    });
  });

  describe('buttons', () => {
    it('renders save button', async () => {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Save Changes/i })).toBeInTheDocument();
      });
    });

    it('renders cancel button', async () => {
      renderSettings();
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Cancel/i })).toBeInTheDocument();
      });
    });
  });

  describe('loading state', () => {
    it('shows loading skeleton while fetching', () => {
      vi.mocked(settingsApi.get).mockImplementation(
        () => new Promise(() => {}) // Never resolves
      );
      
      renderSettings();
      expect(document.querySelector('[class*="animate-pulse"]')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when fetch fails', async () => {
      vi.mocked(settingsApi.get).mockResolvedValue({
        success: false,
        error: {
          type: 'about:blank',
          title: 'Error',
          status: 500,
          detail: 'Failed to load settings',
          code: 'ERROR',
        },
      });
      
      renderSettings();
      await waitFor(() => {
        expect(screen.getByText(/Failed to load settings/i)).toBeInTheDocument();
      });
    });
  });
});
