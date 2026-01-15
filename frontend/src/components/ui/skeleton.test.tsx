/**
 * Skeleton Component Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Skeleton } from './skeleton';

describe('Skeleton', () => {
  describe('rendering', () => {
    it('renders a div element', () => {
      const { container } = render(<Skeleton />);
      expect(container.querySelector('div')).toBeInTheDocument();
    });

    it('renders with custom className', () => {
      render(<Skeleton className="custom-class" data-testid="skeleton" />);
      expect(screen.getByTestId('skeleton')).toHaveClass('custom-class');
    });

    it('has animation class', () => {
      render(<Skeleton data-testid="skeleton" />);
      expect(screen.getByTestId('skeleton')).toHaveClass('animate-pulse');
    });

    it('has rounded styling', () => {
      render(<Skeleton data-testid="skeleton" />);
      expect(screen.getByTestId('skeleton')).toHaveClass('rounded-md');
    });

    it('has muted background', () => {
      render(<Skeleton data-testid="skeleton" />);
      expect(screen.getByTestId('skeleton')).toHaveClass('bg-muted');
    });
  });

  describe('sizing', () => {
    it('can have custom width', () => {
      render(<Skeleton className="w-48" data-testid="skeleton" />);
      expect(screen.getByTestId('skeleton')).toHaveClass('w-48');
    });

    it('can have custom height', () => {
      render(<Skeleton className="h-4" data-testid="skeleton" />);
      expect(screen.getByTestId('skeleton')).toHaveClass('h-4');
    });

    it('can be a circle', () => {
      render(<Skeleton className="h-12 w-12 rounded-full" data-testid="skeleton" />);
      const skeleton = screen.getByTestId('skeleton');
      expect(skeleton).toHaveClass('h-12', 'w-12', 'rounded-full');
    });
  });

  describe('HTML attributes', () => {
    it('supports data attributes', () => {
      render(<Skeleton data-testid="skeleton" data-type="text" />);
      expect(screen.getByTestId('skeleton')).toHaveAttribute('data-type', 'text');
    });

    it('supports aria-hidden', () => {
      render(<Skeleton aria-hidden="true" data-testid="skeleton" />);
      expect(screen.getByTestId('skeleton')).toHaveAttribute('aria-hidden', 'true');
    });

    it('supports role attribute', () => {
      render(<Skeleton role="presentation" data-testid="skeleton" />);
      expect(screen.getByRole('presentation')).toBeInTheDocument();
    });

    it('supports id attribute', () => {
      render(<Skeleton id="skeleton-1" />);
      expect(document.getElementById('skeleton-1')).toBeInTheDocument();
    });
  });

  describe('composition patterns', () => {
    it('renders as text placeholder', () => {
      render(<Skeleton className="h-4 w-[250px]" data-testid="skeleton" />);
      expect(screen.getByTestId('skeleton')).toBeInTheDocument();
    });

    it('renders as avatar placeholder', () => {
      render(<Skeleton className="h-12 w-12 rounded-full" data-testid="skeleton" />);
      expect(screen.getByTestId('skeleton')).toHaveClass('rounded-full');
    });

    it('renders as card placeholder', () => {
      render(
        <div className="space-y-3">
          <Skeleton className="h-[125px] w-[250px]" data-testid="card-image" />
          <Skeleton className="h-4 w-[250px]" data-testid="card-title" />
          <Skeleton className="h-4 w-[200px]" data-testid="card-desc" />
        </div>
      );
      expect(screen.getByTestId('card-image')).toBeInTheDocument();
      expect(screen.getByTestId('card-title')).toBeInTheDocument();
      expect(screen.getByTestId('card-desc')).toBeInTheDocument();
    });

    it('renders multiple lines', () => {
      render(
        <div className="space-y-2">
          <Skeleton className="h-4 w-full" data-testid="line-1" />
          <Skeleton className="h-4 w-full" data-testid="line-2" />
          <Skeleton className="h-4 w-3/4" data-testid="line-3" />
        </div>
      );
      expect(screen.getByTestId('line-1')).toBeInTheDocument();
      expect(screen.getByTestId('line-2')).toBeInTheDocument();
      expect(screen.getByTestId('line-3')).toBeInTheDocument();
    });
  });

  describe('style merging', () => {
    it('preserves custom styles with base classes', () => {
      render(<Skeleton className="my-custom-class" data-testid="skeleton" />);
      const skeleton = screen.getByTestId('skeleton');
      expect(skeleton).toHaveClass('animate-pulse', 'rounded-md', 'bg-muted', 'my-custom-class');
    });

    it('allows overriding rounded style', () => {
      render(<Skeleton className="rounded-lg" data-testid="skeleton" />);
      expect(screen.getByTestId('skeleton')).toHaveClass('rounded-lg');
    });
  });
});
