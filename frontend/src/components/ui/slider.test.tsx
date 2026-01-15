/**
 * Slider Component Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Slider } from './slider';

// Helper to get slider thumb - Radix slider uses role="slider" on the thumb
const getSlider = (container: HTMLElement) => {
  return container.querySelector('[role="slider"]');
};

describe('Slider', () => {
  describe('rendering', () => {
    it('renders slider element', () => {
      const { container } = render(<Slider defaultValue={[50]} />);
      expect(getSlider(container)).toBeInTheDocument();
    });

    it('renders with default value', () => {
      const { container } = render(<Slider defaultValue={[25]} />);
      const slider = getSlider(container);
      expect(slider).toHaveAttribute('aria-valuenow', '25');
    });

    it('renders with custom className', () => {
      const { container } = render(<Slider defaultValue={[50]} className="custom-class" />);
      expect(container.querySelector('.custom-class')).toBeInTheDocument();
    });
  });

  describe('value handling', () => {
    it('supports controlled value', () => {
      const { container } = render(<Slider value={[75]} />);
      const slider = getSlider(container);
      expect(slider).toHaveAttribute('aria-valuenow', '75');
    });

    it('calls onValueChange when value changes', async () => {
      const handleChange = vi.fn();
      const { container } = render(<Slider defaultValue={[50]} onValueChange={handleChange} />);

      const slider = getSlider(container);
      fireEvent.keyDown(slider!, { key: 'ArrowRight' });

      expect(handleChange).toHaveBeenCalled();
    });

    it('respects min value', () => {
      const { container } = render(<Slider defaultValue={[0]} min={0} />);
      const slider = getSlider(container);
      expect(slider).toHaveAttribute('aria-valuemin', '0');
    });

    it('respects max value', () => {
      const { container } = render(<Slider defaultValue={[50]} max={100} />);
      const slider = getSlider(container);
      expect(slider).toHaveAttribute('aria-valuemax', '100');
    });

    it('respects step value', () => {
      const handleChange = vi.fn();
      const { container } = render(<Slider defaultValue={[0]} step={10} onValueChange={handleChange} />);

      const slider = getSlider(container);
      fireEvent.keyDown(slider!, { key: 'ArrowRight' });

      expect(handleChange).toHaveBeenCalledWith([10]);
    });
  });

  describe('keyboard navigation', () => {
    it('increases value with ArrowRight', () => {
      const handleChange = vi.fn();
      const { container } = render(<Slider defaultValue={[50]} onValueChange={handleChange} />);

      const slider = getSlider(container);
      fireEvent.keyDown(slider!, { key: 'ArrowRight' });

      expect(handleChange).toHaveBeenCalledWith([51]);
    });

    it('decreases value with ArrowLeft', () => {
      const handleChange = vi.fn();
      const { container } = render(<Slider defaultValue={[50]} onValueChange={handleChange} />);

      const slider = getSlider(container);
      fireEvent.keyDown(slider!, { key: 'ArrowLeft' });

      expect(handleChange).toHaveBeenCalledWith([49]);
    });

    it('increases value with ArrowUp', () => {
      const handleChange = vi.fn();
      const { container } = render(<Slider defaultValue={[50]} onValueChange={handleChange} />);

      const slider = getSlider(container);
      fireEvent.keyDown(slider!, { key: 'ArrowUp' });

      expect(handleChange).toHaveBeenCalledWith([51]);
    });

    it('decreases value with ArrowDown', () => {
      const handleChange = vi.fn();
      const { container } = render(<Slider defaultValue={[50]} onValueChange={handleChange} />);

      const slider = getSlider(container);
      fireEvent.keyDown(slider!, { key: 'ArrowDown' });

      expect(handleChange).toHaveBeenCalledWith([49]);
    });

    it('jumps to min with Home', () => {
      const handleChange = vi.fn();
      const { container } = render(<Slider defaultValue={[50]} min={0} max={100} onValueChange={handleChange} />);

      const slider = getSlider(container);
      fireEvent.keyDown(slider!, { key: 'Home' });

      expect(handleChange).toHaveBeenCalledWith([0]);
    });

    it('jumps to max with End', () => {
      const handleChange = vi.fn();
      const { container } = render(<Slider defaultValue={[50]} min={0} max={100} onValueChange={handleChange} />);

      const slider = getSlider(container);
      fireEvent.keyDown(slider!, { key: 'End' });

      expect(handleChange).toHaveBeenCalledWith([100]);
    });
  });

  describe('step labels', () => {
    it('renders step labels when provided', () => {
      render(
        <Slider
          defaultValue={[0]}
          min={0}
          max={4}
          stepLabels={['Very Low', 'Low', 'Medium', 'High', 'Very High']}
        />
      );

      expect(screen.getByText('Very Low')).toBeInTheDocument();
      expect(screen.getByText('Low')).toBeInTheDocument();
      expect(screen.getByText('Medium')).toBeInTheDocument();
      expect(screen.getByText('High')).toBeInTheDocument();
      expect(screen.getByText('Very High')).toBeInTheDocument();
    });

    it('highlights current step label', () => {
      render(
        <Slider
          defaultValue={[2]}
          min={0}
          max={4}
          stepLabels={['Very Low', 'Low', 'Medium', 'High', 'Very High']}
        />
      );

      expect(screen.getByText('Medium')).toHaveClass('font-medium', 'text-foreground');
    });

    it('does not render labels when not provided', () => {
      render(<Slider defaultValue={[50]} />);
      expect(screen.queryByText('Very Low')).not.toBeInTheDocument();
    });
  });

  describe('show value', () => {
    it('displays current value when showValue is true', () => {
      render(<Slider defaultValue={[42]} showValue />);
      expect(screen.getByText('42')).toBeInTheDocument();
    });

    it('does not display value when showValue is false', () => {
      render(<Slider defaultValue={[42]} />);
      // The value shouldn't be visible as a standalone text
      const wrapper = document.querySelector('.text-center.text-sm.font-medium');
      expect(wrapper).not.toBeInTheDocument();
    });

    it('updates displayed value when changed', () => {
      const { rerender } = render(<Slider value={[42]} showValue />);
      expect(screen.getByText('42')).toBeInTheDocument();

      rerender(<Slider value={[75]} showValue />);
      expect(screen.getByText('75')).toBeInTheDocument();
    });
  });

  describe('disabled state', () => {
    it('disables slider', () => {
      const { container } = render(<Slider defaultValue={[50]} disabled />);
      const slider = getSlider(container);
      expect(slider).toHaveAttribute('data-disabled');
    });

    it('does not respond to keyboard when disabled', async () => {
      const handleChange = vi.fn();
      const { container } = render(<Slider defaultValue={[50]} disabled onValueChange={handleChange} />);

      const slider = getSlider(container);
      fireEvent.keyDown(slider!, { key: 'ArrowRight' });

      expect(handleChange).not.toHaveBeenCalled();
    });
  });

  describe('styling', () => {
    it('has track element', () => {
      const { container } = render(<Slider defaultValue={[50]} />);
      expect(container.querySelector('[class*="bg-secondary"]')).toBeInTheDocument();
    });

    it('has range element', () => {
      const { container } = render(<Slider defaultValue={[50]} />);
      expect(container.querySelector('[class*="bg-primary"]')).toBeInTheDocument();
    });

    it('has thumb element', () => {
      const { container } = render(<Slider defaultValue={[50]} />);
      expect(getSlider(container)).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('has slider role', () => {
      const { container } = render(<Slider defaultValue={[50]} />);
      expect(getSlider(container)).toHaveAttribute('role', 'slider');
    });

    it('is focusable', async () => {
      const user = userEvent.setup();
      const { container } = render(<Slider defaultValue={[50]} />);

      await user.tab();
      expect(getSlider(container)).toHaveFocus();
    });

    it('has aria-valuenow', () => {
      const { container } = render(<Slider defaultValue={[75]} />);
      expect(getSlider(container)).toHaveAttribute('aria-valuenow', '75');
    });

    it('supports aria-label on container', () => {
      const { container } = render(<Slider defaultValue={[50]} aria-label="Volume" />);
      // aria-label is on the root element, not the thumb
      const root = container.querySelector('[aria-label="Volume"]');
      expect(root).toBeInTheDocument();
    });
  });

  describe('orientation', () => {
    it('defaults to horizontal', () => {
      const { container } = render(<Slider defaultValue={[50]} />);
      expect(container.querySelector('[data-orientation="horizontal"]')).toBeInTheDocument();
    });

    it('supports vertical orientation', () => {
      const { container } = render(<Slider defaultValue={[50]} orientation="vertical" />);
      expect(container.querySelector('[data-orientation="vertical"]')).toBeInTheDocument();
    });
  });

  describe('ref forwarding', () => {
    it('forwards ref to slider element', () => {
      const ref = { current: null as HTMLSpanElement | null };
      render(<Slider ref={ref} defaultValue={[50]} />);
      expect(ref.current).toBeTruthy();
    });
  });

  describe('display name', () => {
    it('has correct display name', () => {
      expect(Slider.displayName).toBe('Slider');
    });
  });
});
