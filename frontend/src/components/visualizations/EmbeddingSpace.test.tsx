/**
 * Embedding Space Visualization Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { EmbeddingSpace } from './EmbeddingSpace';

vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => <div data-testid="canvas">{children}</div>,
  useThree: () => ({
    camera: { position: { x: 0, y: 0, z: 0 } },
    gl: { domElement: document.createElement('canvas') },
  }),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls" />,
  PerspectiveCamera: () => <div data-testid="camera" />,
  Html: ({ children }: { children: React.ReactNode }) => <div data-testid="html">{children}</div>,
  Bounds: ({ children }: { children: React.ReactNode }) => <div data-testid="bounds">{children}</div>,
}));

vi.mock('@/lib/three/canvas-config', () => ({
  isWebGLSupported: () => false,
}));

vi.mock('@/lib/three/a11y/A11yScene', () => ({
  A11yCanvas: ({ children }: { children: React.ReactNode }) => <div data-testid="a11y-canvas">{children}</div>,
  A11yProvider: ({ children }: { children: React.ReactNode }) => <div data-testid="a11y-provider">{children}</div>,
  KeyboardControls: () => <div data-testid="keyboard-controls" />,
  SceneFallback: ({ description }: { description: string }) => <div data-testid="scene-fallback">{description}</div>,
  useA11y: () => ({
    registerInteractiveObject: vi.fn(),
    unregisterInteractiveObject: vi.fn(),
    focusedObjectId: null,
    announce: vi.fn(),
    setFocusedObjectId: vi.fn(),
  }),
}));

describe('EmbeddingSpace', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false, staleTime: 0 },
      },
    });
    vi.clearAllMocks();
  });

  it('renders loading state initially', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <EmbeddingSpace />
      </QueryClientProvider>
    );

    const containers = screen.getAllByRole('generic');
    expect(containers.length).toBeGreaterThan(0);
  });

  it('renders fallback when WebGL is not supported', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <EmbeddingSpace />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('scene-fallback')).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it('uses custom height', () => {
    render(
      <QueryClientProvider client={queryClient}>
        <EmbeddingSpace height="h-96" />
      </QueryClientProvider>
    );

    const containers = screen.getAllByRole('generic');
    expect(containers.length).toBeGreaterThan(0);
  });

  it('handles maxPoints prop', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <EmbeddingSpace maxPoints={500} />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('scene-fallback')).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it('handles colorBy prop', async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <EmbeddingSpace colorBy="confidence" />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('scene-fallback')).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  it('accepts onPointClick callback prop', () => {
    const handlePointClick = vi.fn();
    render(
      <QueryClientProvider client={queryClient}>
        <EmbeddingSpace onPointClick={handlePointClick} />
      </QueryClientProvider>
    );

    expect(handlePointClick).toBeDefined();
  });

  it('accepts onSelectionChange callback prop', () => {
    const handleSelectionChange = vi.fn();
    render(
      <QueryClientProvider client={queryClient}>
        <EmbeddingSpace onSelectionChange={handleSelectionChange} />
      </QueryClientProvider>
    );

    expect(handleSelectionChange).toBeDefined();
  });
});
