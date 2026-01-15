/**
 * Dialog Component Tests
 *
 * Purpose: Test modal dialog components
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogFooter,
  DialogTitle,
  DialogDescription,
  DialogClose,
} from './dialog';

// Mock Radix UI Dialog
vi.mock('@radix-ui/react-dialog', () => ({
  Root: ({ children, open }: { children: React.ReactNode; open?: boolean }) => {
    return <div data-testid="dialog-root" data-open={open ?? false}>{children}</div>;
  },
  Trigger: ({ children, asChild: _asChild }: { children: React.ReactNode; asChild?: boolean }) => (
    <button data-testid="dialog-trigger">{children}</button>
  ),
  Portal: ({ children }: { children: React.ReactNode }) => <div data-testid="dialog-portal">{children}</div>,
  Overlay: ({ className }: { className?: string }) => (
    <div data-testid="dialog-overlay" className={className} />
  ),
  Content: ({ children, className }: { children: React.ReactNode; className?: string }) => (
    <div data-testid="dialog-content" className={className}>{children}</div>
  ),
  Close: ({ children, className }: { children?: React.ReactNode; className?: string }) => (
    <button data-testid="dialog-close" className={className}>{children || 'Close'}</button>
  ),
  Title: ({ children }: { children: React.ReactNode }) => (
    <h2 data-testid="dialog-title">{children}</h2>
  ),
  Description: ({ children }: { children: React.ReactNode }) => (
    <p data-testid="dialog-description">{children}</p>
  ),
}));

describe('Dialog Components', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Dialog', () => {
    it('renders dialog root', () => {
      render(
        <Dialog>
          <div>Dialog content</div>
        </Dialog>
      );
      expect(screen.getByTestId('dialog-root')).toBeInTheDocument();
    });
  });

  describe('DialogTrigger', () => {
    it('renders trigger button', () => {
      render(
        <Dialog>
          <DialogTrigger>Open Dialog</DialogTrigger>
        </Dialog>
      );
      expect(screen.getByTestId('dialog-trigger')).toBeInTheDocument();
      expect(screen.getByText('Open Dialog')).toBeInTheDocument();
    });
  });

  describe('DialogContent', () => {
    it('renders dialog content with portal and overlay', () => {
      render(
        <Dialog open={true}>
          <DialogContent>
            <div>Content</div>
          </DialogContent>
        </Dialog>
      );
      expect(screen.getByTestId('dialog-portal')).toBeInTheDocument();
      expect(screen.getByTestId('dialog-overlay')).toBeInTheDocument();
      expect(screen.getByTestId('dialog-content')).toBeInTheDocument();
      expect(screen.getByText('Content')).toBeInTheDocument();
    });

    it('renders close button', () => {
      render(
        <Dialog open={true}>
          <DialogContent>
            <div>Content</div>
          </DialogContent>
        </Dialog>
      );
      expect(screen.getByTestId('dialog-close')).toBeInTheDocument();
    });
  });

  describe('DialogHeader', () => {
    it('renders header', () => {
      render(
        <Dialog open={true}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Title</DialogTitle>
            </DialogHeader>
          </DialogContent>
        </Dialog>
      );
      expect(screen.getByTestId('dialog-title')).toBeInTheDocument();
      expect(screen.getByText('Title')).toBeInTheDocument();
    });
  });

  describe('DialogTitle', () => {
    it('renders title', () => {
      render(
        <Dialog open={true}>
          <DialogContent>
            <DialogTitle>Dialog Title</DialogTitle>
          </DialogContent>
        </Dialog>
      );
      expect(screen.getByTestId('dialog-title')).toBeInTheDocument();
      expect(screen.getByText('Dialog Title')).toBeInTheDocument();
    });
  });

  describe('DialogDescription', () => {
    it('renders description', () => {
      render(
        <Dialog open={true}>
          <DialogContent>
            <DialogDescription>Dialog description</DialogDescription>
          </DialogContent>
        </Dialog>
      );
      expect(screen.getByTestId('dialog-description')).toBeInTheDocument();
      expect(screen.getByText('Dialog description')).toBeInTheDocument();
    });
  });

  describe('DialogFooter', () => {
    it('renders footer', () => {
      render(
        <Dialog open={true}>
          <DialogContent>
            <DialogFooter>
              <button>Cancel</button>
              <button>Confirm</button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      );
      expect(screen.getByText('Cancel')).toBeInTheDocument();
      expect(screen.getByText('Confirm')).toBeInTheDocument();
    });
  });

  describe('DialogClose', () => {
    it('renders close button', () => {
      render(
        <Dialog open={true}>
          <DialogContent>
            <DialogClose>Close Button</DialogClose>
          </DialogContent>
        </Dialog>
      );
      const closeButtons = screen.getAllByTestId('dialog-close');
      expect(closeButtons.length).toBeGreaterThan(0);
      expect(screen.getByText('Close Button')).toBeInTheDocument();
    });
  });
});
