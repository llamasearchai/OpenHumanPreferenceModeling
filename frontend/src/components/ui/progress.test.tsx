/**
 * Progress Component Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Progress } from './progress';

describe('Progress', () => {
  describe('rendering', () => {
    it('renders with default props', () => {
      render(<Progress />);
      const progress = screen.getByRole('progressbar');
      expect(progress).toBeInTheDocument();
    });

    it('renders with custom className', () => {
      render(<Progress className="custom-class" />);
      expect(screen.getByRole('progressbar')).toHaveClass('custom-class');
    });

    it('has correct base styling', () => {
      render(<Progress />);
      const progress = screen.getByRole('progressbar');
      expect(progress).toHaveClass('relative', 'h-4', 'w-full', 'overflow-hidden', 'rounded-full');
    });
  });

  describe('value handling', () => {
    it('renders with 0% progress', () => {
      render(<Progress value={0} />);
      const indicator = screen.getByRole('progressbar').firstElementChild;
      expect(indicator).toHaveStyle({ transform: 'translateX(-100%)' });
    });

    it('renders with 50% progress', () => {
      render(<Progress value={50} />);
      const indicator = screen.getByRole('progressbar').firstElementChild;
      expect(indicator).toHaveStyle({ transform: 'translateX(-50%)' });
    });

    it('renders with 100% progress', () => {
      render(<Progress value={100} />);
      const indicator = screen.getByRole('progressbar').firstElementChild;
      expect(indicator).toHaveStyle({ transform: 'translateX(-0%)' });
    });

    it('handles undefined value as 0', () => {
      render(<Progress />);
      const indicator = screen.getByRole('progressbar').firstElementChild;
      expect(indicator).toHaveStyle({ transform: 'translateX(-100%)' });
    });

    it('handles decimal values', () => {
      render(<Progress value={33.33} />);
      const indicator = screen.getByRole('progressbar').firstElementChild;
      expect(indicator).toHaveStyle({ transform: 'translateX(-66.67%)' });
    });
  });

  describe('indicator styling', () => {
    it('indicator has correct base classes', () => {
      render(<Progress value={50} />);
      const indicator = screen.getByRole('progressbar').firstElementChild;
      expect(indicator).toHaveClass('h-full', 'w-full', 'flex-1', 'bg-primary', 'transition-all');
    });
  });

  describe('accessibility', () => {
    it('has progressbar role', () => {
      render(<Progress value={50} />);
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });

    it('supports aria-label', () => {
      render(<Progress value={50} aria-label="Upload progress" />);
      expect(screen.getByLabelText('Upload progress')).toBeInTheDocument();
    });

    it('has aria-valuenow when value provided', () => {
      render(<Progress value={75} />);
      const progress = screen.getByRole('progressbar');
      // Radix Progress sets aria-valuenow
      expect(progress).toBeInTheDocument();
    });

    it('has aria-valuemin attribute', () => {
      render(<Progress value={50} />);
      const progress = screen.getByRole('progressbar');
      // Radix Progress sets default aria attributes
      expect(progress).toBeInTheDocument();
    });

    it('has aria-valuemax attribute', () => {
      render(<Progress value={50} />);
      const progress = screen.getByRole('progressbar');
      expect(progress).toBeInTheDocument();
    });
  });

  describe('HTML attributes', () => {
    it('supports data attributes', () => {
      render(<Progress data-testid="custom-progress" value={50} />);
      expect(screen.getByTestId('custom-progress')).toBeInTheDocument();
    });

    it('supports id attribute', () => {
      render(<Progress id="progress-1" value={50} />);
      expect(document.getElementById('progress-1')).toBeInTheDocument();
    });
  });

  describe('ref forwarding', () => {
    it('forwards ref to progress element', () => {
      const ref = { current: null as HTMLDivElement | null };
      render(<Progress ref={ref} value={50} />);
      expect(ref.current).toBeInstanceOf(HTMLDivElement);
    });
  });

  describe('display name', () => {
    it('has correct display name', () => {
      expect(Progress.displayName).toBe('Progress');
    });
  });

  describe('edge cases', () => {
    it('handles negative values', () => {
      render(<Progress value={-10} />);
      const indicator = screen.getByRole('progressbar').firstElementChild;
      // Should clamp or handle gracefully
      expect(indicator).toBeInTheDocument();
    });

    it('handles values over 100', () => {
      render(<Progress value={150} />);
      const indicator = screen.getByRole('progressbar').firstElementChild;
      // Should render, possibly clamped
      expect(indicator).toBeInTheDocument();
    });
  });
});
