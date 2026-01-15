/**
 * ProgressRing Widget Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ProgressRing } from './ProgressRing';

describe('ProgressRing', () => {
  describe('rendering', () => {
    it('renders percentage label by default', () => {
      render(<ProgressRing value={50} />);

      expect(screen.getByText('50%')).toBeInTheDocument();
    });

    it('renders SVG element', () => {
      const { container } = render(<ProgressRing value={50} />);

      expect(container.querySelector('svg')).toBeInTheDocument();
    });

    it('renders two circles (background and progress)', () => {
      const { container } = render(<ProgressRing value={50} />);

      const circles = container.querySelectorAll('circle');
      expect(circles).toHaveLength(2);
    });
  });

  describe('percentage calculation', () => {
    it('calculates percentage correctly', () => {
      render(<ProgressRing value={25} max={100} />);

      expect(screen.getByText('25%')).toBeInTheDocument();
    });

    it('handles custom max value', () => {
      render(<ProgressRing value={50} max={200} />);

      expect(screen.getByText('25%')).toBeInTheDocument();
    });

    it('clamps percentage to 0', () => {
      render(<ProgressRing value={-10} max={100} />);

      expect(screen.getByText('0%')).toBeInTheDocument();
    });

    it('clamps percentage to 100', () => {
      render(<ProgressRing value={150} max={100} />);

      expect(screen.getByText('100%')).toBeInTheDocument();
    });

    it('handles zero value', () => {
      render(<ProgressRing value={0} max={100} />);

      expect(screen.getByText('0%')).toBeInTheDocument();
    });

    it('handles full value', () => {
      render(<ProgressRing value={100} max={100} />);

      expect(screen.getByText('100%')).toBeInTheDocument();
    });
  });

  describe('label', () => {
    it('shows custom label when provided', () => {
      render(<ProgressRing value={75} label="Completed" />);

      expect(screen.getByText('Completed')).toBeInTheDocument();
    });

    it('hides label when showLabel is false', () => {
      render(<ProgressRing value={50} showLabel={false} />);

      expect(screen.queryByText('50%')).not.toBeInTheDocument();
    });

    it('hides custom label when showLabel is false', () => {
      render(<ProgressRing value={50} label="Progress" showLabel={false} />);

      expect(screen.queryByText('Progress')).not.toBeInTheDocument();
    });
  });

  describe('sizing', () => {
    it('uses default size', () => {
      const { container } = render(<ProgressRing value={50} />);

      const svg = container.querySelector('svg');
      expect(svg).toHaveAttribute('width', '120');
      expect(svg).toHaveAttribute('height', '120');
    });

    it('accepts custom size', () => {
      const { container } = render(<ProgressRing value={50} size={200} />);

      const svg = container.querySelector('svg');
      expect(svg).toHaveAttribute('width', '200');
      expect(svg).toHaveAttribute('height', '200');
    });
  });

  describe('styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <ProgressRing value={50} className="custom-ring" />
      );

      expect(container.querySelector('.custom-ring')).toBeInTheDocument();
    });

    it('has transition class on progress circle', () => {
      const { container } = render(<ProgressRing value={50} />);

      const circles = container.querySelectorAll('circle');
      expect(circles[1]).toHaveClass('transition-all');
    });

    it('applies custom colors', () => {
      const { container } = render(
        <ProgressRing
          value={50}
          color="red"
          backgroundColor="blue"
        />
      );

      const circles = container.querySelectorAll('circle');
      expect(circles[0]).toHaveAttribute('stroke', 'blue');
      expect(circles[1]).toHaveAttribute('stroke', 'red');
    });
  });

  describe('stroke width', () => {
    it('uses default stroke width', () => {
      const { container } = render(<ProgressRing value={50} />);

      const circles = container.querySelectorAll('circle');
      circles.forEach(circle => {
        expect(circle).toHaveAttribute('stroke-width', '8');
      });
    });

    it('accepts custom stroke width', () => {
      const { container } = render(<ProgressRing value={50} strokeWidth={12} />);

      const circles = container.querySelectorAll('circle');
      circles.forEach(circle => {
        expect(circle).toHaveAttribute('stroke-width', '12');
      });
    });
  });

  describe('SVG attributes', () => {
    it('has correct circle centers', () => {
      const { container } = render(<ProgressRing value={50} size={100} />);

      const circles = container.querySelectorAll('circle');
      circles.forEach(circle => {
        expect(circle).toHaveAttribute('cx', '50');
        expect(circle).toHaveAttribute('cy', '50');
      });
    });

    it('progress circle has rounded linecap', () => {
      const { container } = render(<ProgressRing value={50} />);

      const progressCircle = container.querySelectorAll('circle')[1];
      expect(progressCircle).toHaveAttribute('stroke-linecap', 'round');
    });

    it('circles have no fill', () => {
      const { container } = render(<ProgressRing value={50} />);

      const circles = container.querySelectorAll('circle');
      circles.forEach(circle => {
        expect(circle).toHaveAttribute('fill', 'none');
      });
    });
  });
});
