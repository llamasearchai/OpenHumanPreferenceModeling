/**
 * Dashboard Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DashboardPage from './Dashboard';
import { healthApi, monitoringApi } from '@/lib/api-client';

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const renderDashboard = () => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <DashboardPage />
    </QueryClientProvider>
  );
};

describe('DashboardPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders loading state and disables refresh while loading', () => {
    const healthSpy = vi.spyOn(healthApi, 'check').mockImplementation(() => new Promise(() => {}));
    const alertsSpy = vi.spyOn(monitoringApi, 'getAlerts').mockImplementation(() => new Promise(() => {}));

    renderDashboard();

    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(document.querySelectorAll('.animate-pulse').length).toBeGreaterThan(0);
    expect(screen.getByRole('button', { name: /Refresh/i })).toBeDisabled();
    expect(healthSpy).toHaveBeenCalled();
    expect(alertsSpy).toHaveBeenCalled();
  });

  it('renders health and alert data when available', async () => {
    vi.spyOn(healthApi, 'check').mockResolvedValue({
      success: true,
      data: {
        encoder: 'healthy',
        dpo: 'healthy',
        monitoring: 'healthy',
        privacy: 'unhealthy',
      },
    });

    vi.spyOn(monitoringApi, 'getAlerts').mockResolvedValue({
      success: true,
      data: [
        {
          id: 'alert-1',
          rule_name: 'Latency Spike',
          severity: 'critical',
          status: 'firing',
          timestamp: '2026-01-10T12:01:00.000Z',
          message: 'P95 latency exceeded 200ms.',
        },
      ],
    });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/alert\(s\) require attention/i)).toBeInTheDocument();
    });

    expect(screen.getByText('Operational')).toBeInTheDocument();
    expect(screen.getByText('Latency Spike')).toBeInTheDocument();
    expect(screen.getByText('critical')).toBeInTheDocument();
  });

  it('uses mock data and shows warning when APIs fail', async () => {
    vi.spyOn(healthApi, 'check').mockResolvedValue({
      success: false,
      error: {
        type: 'about:blank',
        title: 'Error',
        status: 500,
        detail: 'Health API unavailable',
        code: 'HEALTH_ERROR',
      },
    });

    vi.spyOn(monitoringApi, 'getAlerts').mockResolvedValue({
      success: false,
      error: {
        type: 'about:blank',
        title: 'Error',
        status: 500,
        detail: 'Alerts API unavailable',
        code: 'ALERT_ERROR',
      },
    });

    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText(/Using mock data/i)).toBeInTheDocument();
    });

    expect(screen.queryByText('Error loading health status')).not.toBeInTheDocument();
    expect(screen.queryByText('Error loading alerts')).not.toBeInTheDocument();
  });

  it('refetches health data when refresh is clicked', async () => {
    const healthSpy = vi.spyOn(healthApi, 'check').mockResolvedValue({
      success: true,
      data: {
        encoder: 'healthy',
        dpo: 'healthy',
        monitoring: 'healthy',
        privacy: 'healthy',
      },
    });

    vi.spyOn(monitoringApi, 'getAlerts').mockResolvedValue({
      success: true,
      data: [],
    });

    renderDashboard();

    await waitFor(() => {
      expect(healthSpy).toHaveBeenCalledTimes(1);
    });

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /Refresh/i }));

    await waitFor(() => {
      expect(healthSpy).toHaveBeenCalledTimes(2);
    });
  });
});
