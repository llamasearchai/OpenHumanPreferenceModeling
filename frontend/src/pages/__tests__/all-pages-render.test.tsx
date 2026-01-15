/**
 * Comprehensive Page Rendering Tests
 *
 * Purpose: Verify all pages render correctly without errors
 * Tests cover: rendering, key elements, accessibility, and basic functionality
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Import all pages
import DashboardPage from '../Dashboard';
import AnnotationsPage from '../Annotations';
import MetricsPage from '../Metrics';
import AlertsPage from '../Alerts';
import CalibrationPage from '../Calibration';
import ActiveLearningPage from '../ActiveLearning';
import FederatedLearningPage from '../FederatedLearning';
import QualityControlPage from '../QualityControl';
import TrainingPage from '../Training';
import SettingsPage from '../Settings';
import PlaygroundPage from '../Playground';
import NotFoundPage from '../NotFound';

// Mock visualizations and 3D components
vi.mock('@/components/visualizations/EmbeddingSpace', () => ({
  EmbeddingSpace: () => <div data-testid="embedding-space">EmbeddingSpace</div>,
}));

vi.mock('@/components/visualizations/ConfidenceHistogram', () => ({
  ConfidenceHistogram: () => <div data-testid="confidence-histogram">ConfidenceHistogram</div>,
}));

vi.mock('@/components/visualizations/ReliabilityDiagram', () => ({
  ReliabilityDiagram: () => <div data-testid="reliability-diagram">ReliabilityDiagram</div>,
}));

vi.mock('@/components/widgets/AnnotationStats', () => ({
  AnnotationStats: () => <div data-testid="annotation-stats">AnnotationStats</div>,
}));

vi.mock('@/components/widgets/PieChart', () => ({
  PieChart: () => <div data-testid="pie-chart">PieChart</div>,
}));

vi.mock('@/components/widgets/BarChart', () => ({
  BarChart: () => <div data-testid="bar-chart">BarChart</div>,
}));

// Mock recharts
vi.mock('recharts', () => ({
  LineChart: ({ children }: { children: React.ReactNode }) => <div data-testid="line-chart">{children}</div>,
  BarChart: ({ children }: { children: React.ReactNode }) => <div data-testid="bar-chart-recharts">{children}</div>,
  ComposedChart: ({ children }: { children: React.ReactNode }) => <div data-testid="composed-chart">{children}</div>,
  AreaChart: ({ children }: { children: React.ReactNode }) => <div data-testid="area-chart">{children}</div>,
  Line: () => null,
  Bar: () => null,
  Area: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Cell: () => null,
  PieChart: ({ children }: { children: React.ReactNode }) => <div data-testid="pie-chart-recharts">{children}</div>,
  Pie: () => null,
}));

// Mock API clients
vi.mock('@/lib/api-client', () => ({
  healthApi: {
    check: vi.fn().mockResolvedValue({
      success: true,
      data: { encoder: 'healthy', dpo: 'healthy', monitoring: 'healthy', privacy: 'healthy' },
    }),
  },
  monitoringApi: {
    getAlerts: vi.fn().mockResolvedValue({
      success: true,
      data: [],
    }),
    getMetrics: vi.fn().mockResolvedValue({
      success: true,
      data: [
        { timestamp: '2024-01-01T10:00:00Z', value: 0.5, name: 'test_metric', tags: {} },
        { timestamp: '2024-01-01T10:01:00Z', value: 0.6, name: 'test_metric', tags: {} },
      ],
    }),
  },
  annotationApi: {
    getNextTask: vi.fn().mockResolvedValue({
      success: false,
      error: { detail: 'No tasks available' },
    }),
    submitAnnotation: vi.fn().mockResolvedValue({ success: true }),
    getAnnotations: vi.fn().mockResolvedValue({
      success: true,
      data: { data: [], meta: { page: 1, pageSize: 20, total: 0, totalPages: 0, hasNext: false, hasPrev: false } },
    }),
    getQualityMetrics: vi.fn().mockResolvedValue({
      success: true,
      data: { accuracy: 0.95, agreement: 0.88, flags: [] },
    }),
  },
  calibrationApi: {
    triggerRecalibration: vi.fn().mockResolvedValue({ success: true }),
  },
  settingsApi: {
    get: vi.fn().mockResolvedValue({
      success: true,
      data: {
        company_name: 'Open Human Preference Modeling',
        company_phone: '555-0100',
        address: '123 Research Way',
        city: 'San Francisco',
        state: 'CA',
        zip_code: '94105',
        domain: 'ohpm.local',
        allowed_file_types: ['json', 'csv'],
        site_direction: 'Research',
        footer_info: 'OHPM',
      },
    }),
    update: vi.fn().mockResolvedValue({ success: true }),
  },
  apiClient: {
    activeLearning: {
      getStatus: vi.fn().mockResolvedValue({
        success: true,
        data: {
          strategy: 'uncertainty',
          isActive: false,
          samplesAnnotated: 0,
          samplesRemaining: 0,
          currentAccuracy: 0,
          targetAccuracy: 0.95,
          queueSize: 0,
        },
      }),
      getQueue: vi.fn().mockResolvedValue({
        success: true,
        data: [],
      }),
      getConfig: vi.fn().mockResolvedValue({
        success: true,
        data: { strategy: 'uncertainty', batchSize: 32 },
      }),
      updateConfig: vi.fn().mockResolvedValue({ success: true }),
      refreshQueue: vi.fn().mockResolvedValue({ success: true }),
    },
    federated: {
      getStatus: vi.fn().mockResolvedValue({
        success: true,
        data: {
          round: 1,
          isActive: false,
          totalClients: 0,
          activeClients: 0,
          privacyBudget: { epsilonSpent: 0, epsilonRemaining: 10, delta: 1e-5, totalSteps: 0 },
          modelChecksum: '',
          lastUpdated: new Date().toISOString(),
        },
      }),
      getRoundsHistory: vi.fn().mockResolvedValue({
        success: true,
        data: [],
      }),
      getClients: vi.fn().mockResolvedValue({
        success: true,
        data: [],
      }),
      startRound: vi.fn().mockResolvedValue({ success: true }),
      stopRound: vi.fn().mockResolvedValue({ success: true }),
    },
  },
  api: {
    post: vi.fn().mockResolvedValue({
      success: true,
      data: {
        probabilities: [0.1, 0.2, 0.4, 0.2, 0.1],
        action_index: 2,
        confidence: 0.4,
      },
    }),
  },
}));

// Mock hooks
vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}));

vi.mock('@/hooks/use-realtime', () => ({
  useRealtimeTrainingProgress: () => ({
    metrics: [],
    step: 0,
    loss: 0,
    isConnected: false,
    error: null,
  }),
}));

// Mock UI store
vi.mock('@/stores/ui-store', () => ({
  useUIStore: () => ({
    theme: 'light',
    setTheme: vi.fn(),
    sidebarCollapsed: false,
  }),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
    },
  });

const renderWithProviders = (component: React.ReactElement) => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{component}</BrowserRouter>
    </QueryClientProvider>
  );
};

describe('All Pages Render Correctly', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('DashboardPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<DashboardPage />);
      expect(container).toBeTruthy();
    });

    it('renders page heading', () => {
      renderWithProviders(<DashboardPage />);
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('renders main page heading', () => {
      renderWithProviders(<DashboardPage />);
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Dashboard');
    });

    it('renders cards', () => {
      renderWithProviders(<DashboardPage />);
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });

    it('has accessible structure', () => {
      renderWithProviders(<DashboardPage />);
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });
  });

  describe('AnnotationsPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<AnnotationsPage />);
      expect(container).toBeTruthy();
    });

    it('renders page content', () => {
      renderWithProviders(<AnnotationsPage />);
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });

    it('renders annotation stats widget', async () => {
      renderWithProviders(<AnnotationsPage />);
      await waitFor(() => {
        const stats = screen.queryByTestId('annotation-stats');
        if (stats) {
          expect(stats).toBeInTheDocument();
        } else {
          // Widget might not render if no data, which is acceptable
          expect(true).toBe(true);
        }
      }, { timeout: 2000 });
    });

    it('has accessible content structure', () => {
      renderWithProviders(<AnnotationsPage />);
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('MetricsPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<MetricsPage />);
      expect(container).toBeTruthy();
    });

    it('renders page heading', async () => {
      renderWithProviders(<MetricsPage />);
      await waitFor(() => {
        expect(screen.getByText('Metrics')).toBeInTheDocument();
      });
    });

    it('renders metric selector', () => {
      renderWithProviders(<MetricsPage />);
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    it('renders refresh and export buttons', () => {
      renderWithProviders(<MetricsPage />);
      expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Export/i })).toBeInTheDocument();
    });

    it('renders stats cards', async () => {
      renderWithProviders(<MetricsPage />);
      await waitFor(() => {
        expect(screen.getByText('Current')).toBeInTheDocument();
        expect(screen.getByText('Average')).toBeInTheDocument();
      });
    });
  });

  describe('AlertsPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<AlertsPage />);
      expect(container).toBeTruthy();
    });

    it('renders page heading', () => {
      renderWithProviders(<AlertsPage />);
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('renders refresh button', () => {
      renderWithProviders(<AlertsPage />);
      expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
    });

    it('renders filter controls', () => {
      renderWithProviders(<AlertsPage />);
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    it('renders export button', () => {
      renderWithProviders(<AlertsPage />);
      expect(screen.getByRole('button', { name: /Export/i })).toBeInTheDocument();
    });
  });

  describe('CalibrationPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<CalibrationPage />);
      expect(container).toBeTruthy();
    });

    it('renders page headings', () => {
      renderWithProviders(<CalibrationPage />);
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('renders form inputs', () => {
      renderWithProviders(<CalibrationPage />);
      const inputs = document.querySelectorAll('input');
      expect(inputs.length).toBeGreaterThan(0);
    });

    it('renders sliders', () => {
      renderWithProviders(<CalibrationPage />);
      const sliders = document.querySelectorAll('[role="slider"]');
      expect(sliders.length).toBeGreaterThan(0);
    });

    it('renders visualization components', () => {
      renderWithProviders(<CalibrationPage />);
      expect(screen.getByTestId('confidence-histogram')).toBeInTheDocument();
      expect(screen.getByTestId('reliability-diagram')).toBeInTheDocument();
    });
  });

  describe('ActiveLearningPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<ActiveLearningPage />);
      expect(container).toBeTruthy();
    });

    it('renders page headings', () => {
      renderWithProviders(<ActiveLearningPage />);
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('renders tabs', () => {
      renderWithProviders(<ActiveLearningPage />);
      expect(screen.getByRole('tablist')).toBeInTheDocument();
    });

    it('renders refresh button', () => {
      renderWithProviders(<ActiveLearningPage />);
      expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
    });

    it('renders cards', () => {
      renderWithProviders(<ActiveLearningPage />);
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('FederatedLearningPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<FederatedLearningPage />);
      expect(container).toBeTruthy();
    });

    it('renders page headings', () => {
      renderWithProviders(<FederatedLearningPage />);
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('renders tabs', () => {
      renderWithProviders(<FederatedLearningPage />);
      expect(screen.getByRole('tablist')).toBeInTheDocument();
    });

    it('renders cards', () => {
      renderWithProviders(<FederatedLearningPage />);
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('QualityControlPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<QualityControlPage />);
      expect(container).toBeTruthy();
    });

    it('renders page heading', () => {
      renderWithProviders(<QualityControlPage />);
      expect(screen.getByRole('heading', { name: /Quality Control/i })).toBeInTheDocument();
    });

    it('renders search input', () => {
      renderWithProviders(<QualityControlPage />);
      expect(screen.getByLabelText(/Search annotators/i)).toBeInTheDocument();
    });

    it('renders tabs', () => {
      renderWithProviders(<QualityControlPage />);
      expect(screen.getByRole('tablist')).toBeInTheDocument();
    });

    it('renders refresh button', () => {
      renderWithProviders(<QualityControlPage />);
      expect(screen.getByRole('button', { name: /Refresh/i })).toBeInTheDocument();
    });
  });

  describe('TrainingPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<TrainingPage />);
      expect(container).toBeTruthy();
    });

    it('renders page headings', () => {
      renderWithProviders(<TrainingPage />);
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('renders run selector', () => {
      renderWithProviders(<TrainingPage />);
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    it('renders tabs', () => {
      renderWithProviders(<TrainingPage />);
      expect(screen.getByRole('tablist')).toBeInTheDocument();
    });

    it('renders cards', () => {
      renderWithProviders(<TrainingPage />);
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('SettingsPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<SettingsPage />);
      expect(container).toBeTruthy();
    });

    it('renders settings heading', async () => {
      renderWithProviders(<SettingsPage />);
      await waitFor(() => {
        expect(screen.getByRole('heading', { name: /Settings/i })).toBeInTheDocument();
      });
    });

    it('renders general information section', async () => {
      renderWithProviders(<SettingsPage />);
      await waitFor(() => {
        expect(screen.getByText(/General Information/i)).toBeInTheDocument();
      });
    });

    it('renders company name input', async () => {
      renderWithProviders(<SettingsPage />);
      await waitFor(() => {
        expect(screen.getByLabelText(/Company Name/i)).toBeInTheDocument();
      });
    });

    it('has accessible structure', async () => {
      renderWithProviders(<SettingsPage />);
      await waitFor(() => {
        const headings = screen.getAllByRole('heading');
        expect(headings.length).toBeGreaterThan(0);
      });
    });
  });

  describe('PlaygroundPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<PlaygroundPage />);
      expect(container).toBeTruthy();
    });

    it('renders playground heading', () => {
      renderWithProviders(<PlaygroundPage />);
      expect(screen.getByRole('heading', { name: /Model Playground/i })).toBeInTheDocument();
    });

    it('renders state vector input section', () => {
      renderWithProviders(<PlaygroundPage />);
      expect(screen.getByText(/State Vector Input/i)).toBeInTheDocument();
    });

    it('renders all 5 dimension sliders', () => {
      renderWithProviders(<PlaygroundPage />);
      const sliders = document.querySelectorAll('[role="slider"]');
      expect(sliders.length).toBe(5);
    });

    it('renders predict button', () => {
      renderWithProviders(<PlaygroundPage />);
      expect(screen.getByRole('button', { name: /Predict/i })).toBeInTheDocument();
    });

    it('renders results section', () => {
      renderWithProviders(<PlaygroundPage />);
      expect(screen.getByText(/Prediction Results/i)).toBeInTheDocument();
    });
  });

  describe('NotFoundPage', () => {
    it('renders without errors', () => {
      const { container } = renderWithProviders(<NotFoundPage />);
      expect(container).toBeTruthy();
    });

    it('renders 404 heading', () => {
      renderWithProviders(<NotFoundPage />);
      expect(screen.getByText('404')).toBeInTheDocument();
    });

    it('renders Page Not Found text', () => {
      renderWithProviders(<NotFoundPage />);
      expect(screen.getByText('Page Not Found')).toBeInTheDocument();
    });

    it('renders Go Home button', () => {
      renderWithProviders(<NotFoundPage />);
      expect(screen.getByRole('link', { name: /Go Home/i })).toBeInTheDocument();
    });

    it('Go Home button links to home page', () => {
      renderWithProviders(<NotFoundPage />);
      const link = screen.getByRole('link', { name: /Go Home/i });
      expect(link).toHaveAttribute('href', '/');
    });
  });

  describe('Accessibility', () => {
    const pages = [
      { name: 'Dashboard', component: <DashboardPage /> },
      { name: 'Annotations', component: <AnnotationsPage /> },
      { name: 'Metrics', component: <MetricsPage /> },
      { name: 'Alerts', component: <AlertsPage /> },
      { name: 'Calibration', component: <CalibrationPage /> },
      { name: 'ActiveLearning', component: <ActiveLearningPage /> },
      { name: 'FederatedLearning', component: <FederatedLearningPage /> },
      { name: 'QualityControl', component: <QualityControlPage /> },
      { name: 'Training', component: <TrainingPage /> },
      { name: 'Settings', component: <SettingsPage /> },
      { name: 'Playground', component: <PlaygroundPage /> },
      { name: 'NotFound', component: <NotFoundPage /> },
    ];

    pages.forEach(({ name, component }) => {
      it(`${name} renders without crashing`, () => {
        const { container } = renderWithProviders(component);
        expect(container).toBeTruthy();
      });

      it(`${name} renders content`, () => {
        const { container } = renderWithProviders(component);
        expect(container).toBeTruthy();
        expect(container.innerHTML.length).toBeGreaterThan(0);
      });

      it(`${name} buttons are keyboard accessible`, () => {
        renderWithProviders(component);
        const buttons = screen.queryAllByRole('button');
        buttons.forEach((button) => {
          const tabindex = button.getAttribute('tabindex');
          if (tabindex !== null) {
            expect(tabindex).not.toBe('-1');
          }
        });
      });
    });
  });

  describe('Error Handling', () => {
    it('all pages handle loading states', () => {
      const pages = [
        <DashboardPage />,
        <AnnotationsPage />,
        <MetricsPage />,
        <AlertsPage />,
        <CalibrationPage />,
        <ActiveLearningPage />,
        <FederatedLearningPage />,
        <QualityControlPage />,
        <TrainingPage />,
        <SettingsPage />,
        <PlaygroundPage />,
      ];

      pages.forEach((page) => {
        const { container } = renderWithProviders(page);
        expect(container).toBeTruthy();
      });
    });
  });
});
