/**
 * Toaster Component Tests
 *
 * Purpose: Test toast container and rendering
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Toaster } from './toaster';
import { useToast } from '@/hooks/use-toast';

vi.mock('@/hooks/use-toast', () => ({
  useToast: vi.fn(),
}));

vi.mock('./toast', () => ({
  Toast: ({ children }: { children: React.ReactNode }) => <div data-testid="toast">{children}</div>,
  ToastClose: () => <button data-testid="toast-close">Close</button>,
  ToastDescription: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="toast-description">{children}</div>
  ),
  ToastProvider: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="toast-provider">{children}</div>
  ),
  ToastTitle: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="toast-title">{children}</div>
  ),
  ToastViewport: () => <div data-testid="toast-viewport" />,
}));

describe('Toaster', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders provider and viewport', () => {
    vi.mocked(useToast).mockReturnValue({
      toasts: [],
      toast: vi.fn(),
      dismiss: vi.fn(),
    });

    render(<Toaster />);
    expect(screen.getByTestId('toast-provider')).toBeInTheDocument();
    expect(screen.getByTestId('toast-viewport')).toBeInTheDocument();
  });

  it('renders toasts from useToast hook', () => {
    vi.mocked(useToast).mockReturnValue({
      toasts: [
        {
          id: '1',
          title: 'Test Title',
          description: 'Test Description',
        },
      ],
      toast: vi.fn(),
      dismiss: vi.fn(),
    });

    render(<Toaster />);
    expect(screen.getByTestId('toast')).toBeInTheDocument();
    expect(screen.getByText('Test Title')).toBeInTheDocument();
    expect(screen.getByText('Test Description')).toBeInTheDocument();
  });

  it('renders multiple toasts', () => {
    vi.mocked(useToast).mockReturnValue({
      toasts: [
        { id: '1', title: 'Toast 1' },
        { id: '2', title: 'Toast 2' },
      ],
      toast: vi.fn(),
      dismiss: vi.fn(),
    });

    render(<Toaster />);
    expect(screen.getAllByTestId('toast')).toHaveLength(2);
    expect(screen.getByText('Toast 1')).toBeInTheDocument();
    expect(screen.getByText('Toast 2')).toBeInTheDocument();
  });

  it('renders toast without title', () => {
    vi.mocked(useToast).mockReturnValue({
      toasts: [
        {
          id: '1',
          description: 'Description only',
        },
      ],
      toast: vi.fn(),
      dismiss: vi.fn(),
    });

    render(<Toaster />);
    expect(screen.getByText('Description only')).toBeInTheDocument();
    expect(screen.queryByTestId('toast-title')).not.toBeInTheDocument();
  });

  it('renders toast without description', () => {
    vi.mocked(useToast).mockReturnValue({
      toasts: [
        {
          id: '1',
          title: 'Title only',
        },
      ],
      toast: vi.fn(),
      dismiss: vi.fn(),
    });

    render(<Toaster />);
    expect(screen.getByText('Title only')).toBeInTheDocument();
    expect(screen.queryByTestId('toast-description')).not.toBeInTheDocument();
  });

  it('renders action when provided', () => {
    const action = <button>Action Button</button>;
    vi.mocked(useToast).mockReturnValue({
      toasts: [
        {
          id: '1',
          title: 'Test',
          action,
        },
      ],
      toast: vi.fn(),
      dismiss: vi.fn(),
    });

    render(<Toaster />);
    expect(screen.getByText('Action Button')).toBeInTheDocument();
  });

  it('renders close button for each toast', () => {
    vi.mocked(useToast).mockReturnValue({
      toasts: [
        { id: '1', title: 'Toast 1' },
        { id: '2', title: 'Toast 2' },
      ],
      toast: vi.fn(),
      dismiss: vi.fn(),
    });

    render(<Toaster />);
    expect(screen.getAllByTestId('toast-close')).toHaveLength(2);
  });
});
