/**
 * Toast Component Tests
 *
 * Purpose: Test toast notification components
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Toast, ToastAction, ToastClose, ToastDescription, ToastTitle, ToastViewport, ToastProvider } from './toast';

// Mock Radix UI Toast
vi.mock('@radix-ui/react-toast', () => ({
  Provider: ({ children }: { children: React.ReactNode }) => <div data-testid="toast-provider">{children}</div>,
  Root: ({ children, className }: { children: React.ReactNode; className?: string }) => (
    <div data-testid="toast-root" className={className}>{children}</div>
  ),
  Viewport: ({ className }: { className?: string }) => (
    <div data-testid="toast-viewport" className={className} />
  ),
  Action: ({ children, className }: { children: React.ReactNode; className?: string }) => (
    <button data-testid="toast-action" className={className}>{children}</button>
  ),
  Close: ({ className }: { className?: string }) => (
    <button data-testid="toast-close" className={className} aria-label="Close notification" />
  ),
  Title: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="toast-title">{children}</div>
  ),
  Description: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="toast-description">{children}</div>
  ),
}));

describe('Toast Components', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Toast', () => {
    it('renders toast root', () => {
      render(<Toast>Toast content</Toast>);
      expect(screen.getByTestId('toast-root')).toBeInTheDocument();
      expect(screen.getByText('Toast content')).toBeInTheDocument();
    });

    it('applies default variant', () => {
      render(<Toast>Toast content</Toast>);
      const root = screen.getByTestId('toast-root');
      expect(root).toHaveClass('border', 'bg-background');
    });

    it('applies success variant', () => {
      render(<Toast variant="success">Success toast</Toast>);
      const root = screen.getByTestId('toast-root');
      expect(root).toHaveClass('border-green-200', 'bg-green-50');
    });

    it('applies warning variant', () => {
      render(<Toast variant="warning">Warning toast</Toast>);
      const root = screen.getByTestId('toast-root');
      expect(root).toHaveClass('border-yellow-200', 'bg-yellow-50');
    });

    it('applies destructive variant', () => {
      render(<Toast variant="destructive">Error toast</Toast>);
      const root = screen.getByTestId('toast-root');
      expect(root).toHaveClass('destructive', 'border-destructive');
    });
  });

  describe('ToastTitle', () => {
    it('renders title', () => {
      render(<ToastTitle>Toast Title</ToastTitle>);
      expect(screen.getByTestId('toast-title')).toBeInTheDocument();
      expect(screen.getByText('Toast Title')).toBeInTheDocument();
    });
  });

  describe('ToastDescription', () => {
    it('renders description', () => {
      render(<ToastDescription>Toast description</ToastDescription>);
      expect(screen.getByTestId('toast-description')).toBeInTheDocument();
      expect(screen.getByText('Toast description')).toBeInTheDocument();
    });
  });

  describe('ToastAction', () => {
    it('renders action button', () => {
      render(<ToastAction altText="Action">Action</ToastAction>);
      expect(screen.getByTestId('toast-action')).toBeInTheDocument();
      expect(screen.getByText('Action')).toBeInTheDocument();
    });
  });

  describe('ToastClose', () => {
    it('renders close button', () => {
      render(<ToastClose />);
      expect(screen.getByTestId('toast-close')).toBeInTheDocument();
      expect(screen.getByLabelText('Close notification')).toBeInTheDocument();
    });
  });

  describe('ToastViewport', () => {
    it('renders viewport', () => {
      render(<ToastViewport />);
      expect(screen.getByTestId('toast-viewport')).toBeInTheDocument();
    });
  });

  describe('ToastProvider', () => {
    it('renders provider', () => {
      render(
        <ToastProvider>
          <div>Content</div>
        </ToastProvider>
      );
      expect(screen.getByTestId('toast-provider')).toBeInTheDocument();
      expect(screen.getByText('Content')).toBeInTheDocument();
    });
  });
});
