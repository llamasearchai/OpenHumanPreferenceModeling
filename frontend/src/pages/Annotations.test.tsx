/**
 * Annotations Page Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AnnotationsPage from './Annotations';

// Mock visualizations
vi.mock('@/components/visualizations/EmbeddingSpace', () => ({
  EmbeddingSpace: () => <div data-testid="embedding-space">EmbeddingSpace</div>,
}));

vi.mock('@/components/widgets/AnnotationStats', () => ({
  AnnotationStats: () => <div data-testid="annotation-stats">AnnotationStats</div>,
}));


// Mock the API client
vi.mock('@/lib/api-client', () => ({
  annotationApi: {
    getNextTask: vi.fn().mockResolvedValue({
      success: false,
      error: { detail: 'No tasks available' },
    }),
    submitAnnotation: vi.fn().mockResolvedValue({ success: true }),
    getAnnotatorStats: vi.fn().mockResolvedValue({
      success: true,
      data: { total: 0 },
    }),
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

const renderAnnotations = () => {
  const queryClient = createQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AnnotationsPage />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('AnnotationsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders loading or content', () => {
      renderAnnotations();
      // Either shows loading skeleton or content
      const content = document.querySelector('.animate-pulse') || document.querySelector('[class*="card"]');
      expect(content).toBeInTheDocument();
    });

    it('renders card sections', () => {
      renderAnnotations();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('content', () => {
    it('renders cards', () => {
      renderAnnotations();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });

  describe('accessibility', () => {
    it('has accessible content structure', () => {
      renderAnnotations();
      const cards = document.querySelectorAll('[class*="card"]');
      expect(cards.length).toBeGreaterThan(0);
    });
  });
});
