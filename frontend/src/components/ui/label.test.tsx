/* eslint-disable no-undef */
/**
 * Label Component Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Label } from './label';

describe('Label', () => {
  describe('rendering', () => {
    it('renders with text content', () => {
      render(<Label>Username</Label>);
      expect(screen.getByText('Username')).toBeInTheDocument();
    });

    it('renders with custom className', () => {
      render(<Label className="custom-class">Label</Label>);
      expect(screen.getByText('Label')).toHaveClass('custom-class');
    });

    it('has correct base styling', () => {
      render(<Label>Label</Label>);
      const label = screen.getByText('Label');
      expect(label).toHaveClass('text-sm', 'font-medium', 'leading-none');
    });
  });

  describe('htmlFor association', () => {
    it('associates with input via htmlFor', () => {
      render(
        <>
          <Label htmlFor="email">Email</Label>
          <input id="email" type="email" />
        </>
      );

      const label = screen.getByText('Email');
      expect(label).toHaveAttribute('for', 'email');
    });

    it('clicking label focuses associated input', async () => {
      const user = (await import('@testing-library/user-event')).default.setup();
      const { container } = render(
        <>
          <Label htmlFor="test-input">Click me</Label>
          <input id="test-input" />
        </>
      );

      const label = screen.getByText('Click me');
      await user.click(label);

      const input = container.querySelector('#test-input');
      expect(input).toHaveFocus();
    });
  });

  describe('peer styling', () => {
    it('has peer-disabled styling classes', () => {
      render(<Label>Label</Label>);
      const label = screen.getByText('Label');
      expect(label).toHaveClass('peer-disabled:cursor-not-allowed', 'peer-disabled:opacity-70');
    });
  });

  describe('HTML attributes', () => {
    it('supports data attributes', () => {
      render(<Label data-testid="custom-label">Label</Label>);
      expect(screen.getByTestId('custom-label')).toBeInTheDocument();
    });

    it('supports id attribute', () => {
      render(<Label id="label-1">Label</Label>);
      expect(document.getElementById('label-1')).toBeInTheDocument();
    });

    it('supports aria attributes', () => {
      render(<Label aria-describedby="desc">Label</Label>);
      expect(screen.getByText('Label')).toHaveAttribute('aria-describedby', 'desc');
    });
  });

  describe('ref forwarding', () => {
    it('forwards ref to label element', () => {
      const ref = { current: null as HTMLLabelElement | null };
      render(<Label ref={ref}>Label</Label>);
      expect(ref.current).toBeInstanceOf(HTMLLabelElement);
    });
  });

  describe('children types', () => {
    it('renders string children', () => {
      render(<Label>Simple text</Label>);
      expect(screen.getByText('Simple text')).toBeInTheDocument();
    });

    it('renders JSX children', () => {
      render(
        <Label>
          <span data-testid="inner">Required</span>
          <span className="text-red-500">*</span>
        </Label>
      );
      expect(screen.getByTestId('inner')).toBeInTheDocument();
    });

    it('renders with required indicator pattern', () => {
      render(
        <Label>
          Email <span className="text-destructive">*</span>
        </Label>
      );
      expect(screen.getByText('Email')).toBeInTheDocument();
    });
  });

  describe('display name', () => {
    it('has correct display name', () => {
      expect(Label.displayName).toBe('Label');
    });
  });
});
