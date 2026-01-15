/**
 * Badge Component Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Badge } from './badge';

describe('Badge', () => {
  describe('rendering', () => {
    it('renders with default props', () => {
      render(<Badge>Status</Badge>);
      expect(screen.getByText('Status')).toBeInTheDocument();
    });

    it('renders children correctly', () => {
      render(<Badge>Active</Badge>);
      expect(screen.getByText('Active')).toBeInTheDocument();
    });

    it('renders with custom className', () => {
      render(<Badge className="custom-class">Badge</Badge>);
      expect(screen.getByText('Badge')).toHaveClass('custom-class');
    });

    it('has correct base styling', () => {
      render(<Badge>Badge</Badge>);
      const badge = screen.getByText('Badge');
      expect(badge).toHaveClass('inline-flex', 'items-center', 'rounded-full');
    });
  });

  describe('variants', () => {
    it('renders default variant', () => {
      render(<Badge variant="default">Default</Badge>);
      expect(screen.getByText('Default')).toHaveClass('bg-primary');
    });

    it('renders secondary variant', () => {
      render(<Badge variant="secondary">Secondary</Badge>);
      expect(screen.getByText('Secondary')).toHaveClass('bg-secondary');
    });

    it('renders destructive variant', () => {
      render(<Badge variant="destructive">Destructive</Badge>);
      expect(screen.getByText('Destructive')).toHaveClass('bg-destructive');
    });

    it('renders success variant', () => {
      render(<Badge variant="success">Success</Badge>);
      expect(screen.getByText('Success')).toHaveClass('bg-green-500');
    });

    it('renders warning variant', () => {
      render(<Badge variant="warning">Warning</Badge>);
      expect(screen.getByText('Warning')).toHaveClass('bg-yellow-500');
    });

    it('renders outline variant', () => {
      render(<Badge variant="outline">Outline</Badge>);
      expect(screen.getByText('Outline')).toHaveClass('text-foreground');
    });

    it('uses default variant when none specified', () => {
      render(<Badge>Default</Badge>);
      expect(screen.getByText('Default')).toHaveClass('bg-primary');
    });
  });

  describe('HTML attributes', () => {
    it('supports data attributes', () => {
      render(<Badge data-testid="custom-badge" data-status="active">Badge</Badge>);
      const badge = screen.getByTestId('custom-badge');
      expect(badge).toHaveAttribute('data-status', 'active');
    });

    it('supports id attribute', () => {
      render(<Badge id="badge-1">Badge</Badge>);
      expect(document.getElementById('badge-1')).toBeInTheDocument();
    });

    it('supports aria attributes', () => {
      render(<Badge aria-label="Status indicator">Badge</Badge>);
      expect(screen.getByLabelText('Status indicator')).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    it('has focus ring styles', () => {
      render(<Badge>Badge</Badge>);
      expect(screen.getByText('Badge')).toHaveClass('focus:ring-2');
    });

    it('has transition styles', () => {
      render(<Badge>Badge</Badge>);
      expect(screen.getByText('Badge')).toHaveClass('transition-colors');
    });

    it('has proper text sizing', () => {
      render(<Badge>Badge</Badge>);
      expect(screen.getByText('Badge')).toHaveClass('text-xs', 'font-semibold');
    });

    it('has proper padding', () => {
      render(<Badge>Badge</Badge>);
      expect(screen.getByText('Badge')).toHaveClass('px-2.5', 'py-0.5');
    });
  });

  describe('with icons and content', () => {
    it('renders with icon', () => {
      render(
        <Badge>
          <span data-testid="icon">●</span>
          <span>Online</span>
        </Badge>
      );
      expect(screen.getByTestId('icon')).toBeInTheDocument();
      expect(screen.getByText('Online')).toBeInTheDocument();
    });

    it('renders numbers', () => {
      render(<Badge>42</Badge>);
      expect(screen.getByText('42')).toBeInTheDocument();
    });

    it('renders empty badge', () => {
      const { container } = render(<Badge />);
      expect(container.querySelector('div')).toBeInTheDocument();
    });
  });
});
