/**
 * Button Component Tests
 * 
 * Purpose: Comprehensive unit tests for Button component
 * Coverage target: 90%+
 * 
 * Test categories:
 * - Rendering variants and sizes
 * - Interactive states (loading, disabled)
 * - Click handlers
 * - Keyboard interactions
 * - Accessibility attributes
 * - Icon rendering
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button } from './button';
import { Check, ChevronRight } from 'lucide-react';

describe('Button', () => {
  // ===========================================================================
  // Rendering Tests
  // ===========================================================================

  describe('rendering', () => {
    it('renders with default props', () => {
      render(<Button>Click me</Button>);
      
      const button = screen.getByRole('button', { name: /click me/i });
      expect(button).toBeInTheDocument();
      expect(button).toHaveClass('bg-primary');
    });

    it('renders children correctly', () => {
      render(<Button>Test Button</Button>);
      expect(screen.getByText('Test Button')).toBeInTheDocument();
    });

    it('renders with custom className', () => {
      render(<Button className="custom-class">Button</Button>);
      expect(screen.getByRole('button')).toHaveClass('custom-class');
    });
  });

  // ===========================================================================
  // Variant Tests
  // ===========================================================================

  describe('variants', () => {
    it('renders default variant', () => {
      render(<Button variant="default">Default</Button>);
      expect(screen.getByRole('button')).toHaveClass('bg-primary');
    });

    it('renders destructive variant', () => {
      render(<Button variant="destructive">Destructive</Button>);
      expect(screen.getByRole('button')).toHaveClass('bg-destructive');
    });

    it('renders outline variant', () => {
      render(<Button variant="outline">Outline</Button>);
      expect(screen.getByRole('button')).toHaveClass('border-input');
    });

    it('renders secondary variant', () => {
      render(<Button variant="secondary">Secondary</Button>);
      expect(screen.getByRole('button')).toHaveClass('bg-secondary');
    });

    it('renders ghost variant', () => {
      render(<Button variant="ghost">Ghost</Button>);
      expect(screen.getByRole('button')).toHaveClass('hover:bg-accent');
    });

    it('renders link variant', () => {
      render(<Button variant="link">Link</Button>);
      expect(screen.getByRole('button')).toHaveClass('underline-offset-4');
    });
  });

  // ===========================================================================
  // Size Tests
  // ===========================================================================

  describe('sizes', () => {
    it('renders default size', () => {
      render(<Button size="default">Default Size</Button>);
      expect(screen.getByRole('button')).toHaveClass('h-10', 'px-4');
    });

    it('renders small size', () => {
      render(<Button size="sm">Small</Button>);
      expect(screen.getByRole('button')).toHaveClass('h-9', 'px-3');
    });

    it('renders large size', () => {
      render(<Button size="lg">Large</Button>);
      expect(screen.getByRole('button')).toHaveClass('h-11', 'px-8');
    });

    it('renders icon size', () => {
      render(<Button size="icon">+</Button>);
      expect(screen.getByRole('button')).toHaveClass('h-10', 'w-10');
    });
  });

  // ===========================================================================
  // Loading State Tests
  // ===========================================================================

  describe('loading state', () => {
    it('shows loading spinner when isLoading is true', () => {
      render(<Button isLoading>Loading</Button>);
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-busy', 'true');
      expect(button.querySelector('.animate-spin')).toBeInTheDocument();
    });

    it('disables button when loading', () => {
      render(<Button isLoading>Loading</Button>);
      expect(screen.getByRole('button')).toBeDisabled();
    });

    it('hides content but maintains width when loading', () => {
      render(<Button isLoading>Loading Text</Button>);
      
      const content = screen.getByText('Loading Text');
      expect(content).toHaveClass('invisible');
    });

    it('announces loading state to screen readers', () => {
      render(<Button isLoading>Loading</Button>);
      expect(screen.getByText('Loading...', { selector: '.sr-only' })).toBeInTheDocument();
    });

    it('hides icons when loading', () => {
      render(
        <Button isLoading leftIcon={<Check data-testid="left-icon" />}>
          Button
        </Button>
      );
      
      expect(screen.queryByTestId('left-icon')).not.toBeInTheDocument();
    });
  });

  // ===========================================================================
  // Disabled State Tests
  // ===========================================================================

  describe('disabled state', () => {
    it('applies disabled attribute', () => {
      render(<Button disabled>Disabled</Button>);
      expect(screen.getByRole('button')).toBeDisabled();
    });

    it('applies aria-disabled attribute', () => {
      render(<Button disabled>Disabled</Button>);
      expect(screen.getByRole('button')).toHaveAttribute('aria-disabled', 'true');
    });

    it('does not trigger onClick when disabled', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();
      
      render(<Button disabled onClick={handleClick}>Disabled</Button>);
      
      await user.click(screen.getByRole('button'));
      expect(handleClick).not.toHaveBeenCalled();
    });

    it('has reduced opacity when disabled', () => {
      render(<Button disabled>Disabled</Button>);
      expect(screen.getByRole('button')).toHaveClass('disabled:opacity-50');
    });
  });

  // ===========================================================================
  // Icon Tests
  // ===========================================================================

  describe('icons', () => {
    it('renders left icon', () => {
      render(
        <Button leftIcon={<Check data-testid="left-icon" />}>
          With Icon
        </Button>
      );
      
      expect(screen.getByTestId('left-icon')).toBeInTheDocument();
    });

    it('renders right icon', () => {
      render(
        <Button rightIcon={<ChevronRight data-testid="right-icon" />}>
          With Icon
        </Button>
      );
      
      expect(screen.getByTestId('right-icon')).toBeInTheDocument();
    });

    it('renders both icons', () => {
      render(
        <Button
          leftIcon={<Check data-testid="left-icon" />}
          rightIcon={<ChevronRight data-testid="right-icon" />}
        >
          Both Icons
        </Button>
      );
      
      expect(screen.getByTestId('left-icon')).toBeInTheDocument();
      expect(screen.getByTestId('right-icon')).toBeInTheDocument();
    });

    it('icons are hidden from screen readers', () => {
      render(
        <Button leftIcon={<Check />}>
          Icon Button
        </Button>
      );
      
      const iconWrapper = screen.getByRole('button').querySelector('[aria-hidden="true"]');
      expect(iconWrapper).toBeInTheDocument();
    });
  });

  // ===========================================================================
  // Click Handler Tests
  // ===========================================================================

  describe('click handlers', () => {
    it('calls onClick when clicked', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();
      
      render(<Button onClick={handleClick}>Click me</Button>);
      
      await user.click(screen.getByRole('button'));
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('receives event object in onClick', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();
      
      render(<Button onClick={handleClick}>Click me</Button>);
      
      await user.click(screen.getByRole('button'));
      expect(handleClick).toHaveBeenCalledWith(expect.any(Object));
    });

    it('does not call onClick when loading', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();
      
      render(<Button isLoading onClick={handleClick}>Loading</Button>);
      
      await user.click(screen.getByRole('button'));
      expect(handleClick).not.toHaveBeenCalled();
    });
  });

  // ===========================================================================
  // Keyboard Interaction Tests
  // ===========================================================================

  describe('keyboard interactions', () => {
    it('is focusable', async () => {
      const user = userEvent.setup();
      render(<Button>Focusable</Button>);
      
      await user.tab();
      expect(screen.getByRole('button')).toHaveFocus();
    });

    it('activates on Enter key', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();
      
      render(<Button onClick={handleClick}>Press Enter</Button>);
      
      await user.tab();
      await user.keyboard('{Enter}');
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('activates on Space key', async () => {
      const user = userEvent.setup();
      const handleClick = vi.fn();
      
      render(<Button onClick={handleClick}>Press Space</Button>);
      
      await user.tab();
      await user.keyboard(' ');
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    it('is not focusable when disabled', async () => {
      const user = userEvent.setup();
      render(
        <>
          <Button>First</Button>
          <Button disabled>Disabled</Button>
          <Button>Third</Button>
        </>
      );
      
      await user.tab();
      expect(screen.getByRole('button', { name: 'First' })).toHaveFocus();
      
      await user.tab();
      expect(screen.getByRole('button', { name: 'Third' })).toHaveFocus();
    });
  });

  // ===========================================================================
  // Full Width Tests
  // ===========================================================================

  describe('full width', () => {
    it('applies full width class', () => {
      render(<Button fullWidth>Full Width</Button>);
      expect(screen.getByRole('button')).toHaveClass('w-full');
    });

    it('does not apply full width by default', () => {
      render(<Button>Normal Width</Button>);
      expect(screen.getByRole('button')).not.toHaveClass('w-full');
    });
  });

  // ===========================================================================
  // asChild Tests
  // ===========================================================================

  describe('asChild', () => {
    it('renders as child element when asChild is true', () => {
      render(
        <Button asChild>
          <a href="/test">Link Button</a>
        </Button>
      );
      
      const link = screen.getByRole('link', { name: /link button/i });
      expect(link).toBeInTheDocument();
      expect(link).toHaveAttribute('href', '/test');
    });

    it('applies button styles to child element', () => {
      render(
        <Button asChild variant="destructive">
          <a href="/test">Styled Link</a>
        </Button>
      );
      
      expect(screen.getByRole('link')).toHaveClass('bg-destructive');
    });
  });

  // ===========================================================================
  // Accessibility Tests
  // ===========================================================================

  describe('accessibility', () => {
    it('has correct role', () => {
      render(<Button>Accessible</Button>);
      expect(screen.getByRole('button')).toBeInTheDocument();
    });

    it('has visible focus indicator class', () => {
      render(<Button>Focus Visible</Button>);
      expect(screen.getByRole('button')).toHaveClass('focus-visible:ring-2');
    });

    it('supports aria-label', () => {
      render(<Button aria-label="Close dialog">×</Button>);
      expect(screen.getByRole('button', { name: 'Close dialog' })).toBeInTheDocument();
    });

    it('supports aria-describedby', () => {
      render(
        <>
          <Button aria-describedby="description">Described Button</Button>
          <p id="description">This button does something important</p>
        </>
      );
      
      expect(screen.getByRole('button')).toHaveAttribute('aria-describedby', 'description');
    });
  });

  // ===========================================================================
  // HTML Attribute Pass-through Tests
  // ===========================================================================

  describe('HTML attributes', () => {
    it('supports type attribute', () => {
      render(<Button type="submit">Submit</Button>);
      expect(screen.getByRole('button')).toHaveAttribute('type', 'submit');
    });

    it('supports form attribute', () => {
      render(<Button form="my-form">Submit Form</Button>);
      expect(screen.getByRole('button')).toHaveAttribute('form', 'my-form');
    });

    it('supports data attributes', () => {
      render(<Button data-testid="custom-button" data-action="save">Save</Button>);
      
      const button = screen.getByTestId('custom-button');
      expect(button).toHaveAttribute('data-action', 'save');
    });

    it('supports name attribute', () => {
      render(<Button name="action-button">Named Button</Button>);
      expect(screen.getByRole('button')).toHaveAttribute('name', 'action-button');
    });
  });
});
