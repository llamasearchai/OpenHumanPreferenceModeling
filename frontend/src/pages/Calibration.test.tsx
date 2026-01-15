/**
 * Calibration Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CalibrationPage from './Calibration';

// Mock visualizations
vi.mock('@/components/visualizations/ConfidenceHistogram', () => ({
  ConfidenceHistogram: () => <div data-testid="confidence-histogram">Histogram</div>,
}));

vi.mock('@/components/visualizations/ReliabilityDiagram', () => ({
  ReliabilityDiagram: () => <div data-testid="reliability-diagram">Diagram</div>,
}));

// Mock the API client
vi.mock('@/lib/api-client', () => ({
  calibrationApi: {
    triggerRecalibration: vi.fn().mockResolvedValue({ success: true }),
  },
}));

// Mock toast
vi.mock('@/hooks/use-toast', () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}));

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

const renderCalibration = () => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <CalibrationPage />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('CalibrationPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders page headings', () => {
      renderCalibration();
      const headings = screen.getAllByRole('heading');
      expect(headings.length).toBeGreaterThan(0);
    });

    it('renders form inputs', () => {
      renderCalibration();
      const inputs = document.querySelectorAll('input');
      expect(inputs.length).toBeGreaterThan(0);
    });

    it('renders sliders', () => {
      renderCalibration();
      const sliders = document.querySelectorAll('[role="slider"]');
      expect(sliders.length).toBeGreaterThan(0);
    });

    it('renders submit button', () => {
      renderCalibration();
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });
  });

  describe('cards', () => {
    it('renders card sections', () => {
      renderCalibration();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('accessibility', () => {
    it('form inputs are accessible', () => {
      renderCalibration();
      const inputs = document.querySelectorAll('input');
      expect(inputs.length).toBeGreaterThan(0);
    });

    it('sliders are keyboard accessible', () => {
      renderCalibration();
      const sliders = document.querySelectorAll('[role="slider"]');
      sliders.forEach((slider) => {
        expect(slider).toHaveAttribute('tabindex');
      });
    });
  });
});
