/**
 * Input Component Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Input } from './input';

describe('Input', () => {
  describe('Rendering', () => {
    it('renders with default props', () => {
      render(<Input placeholder="Enter text" />);
      expect(screen.getByPlaceholderText('Enter text')).toBeInTheDocument();
    });

    it('renders with label', () => {
      render(<Input label="Username" />);
      expect(screen.getByText('Username')).toBeInTheDocument();
    });

    it('renders with helper text', () => {
      render(<Input helperText="Enter your username" />);
      expect(screen.getByText('Enter your username')).toBeInTheDocument();
    });

    it('renders with error message', () => {
      render(<Input error="This field is required" />);
      expect(screen.getByText('This field is required')).toBeInTheDocument();
    });

    it('applies error variant when error is present', () => {
      render(<Input error="Error" data-testid="input" />);
      const input = screen.getByTestId('input');
      expect(input).toHaveAttribute('aria-invalid', 'true');
    });

    it('renders with left element', () => {
      render(<Input leftElement={<span data-testid="left-icon">@</span>} />);
      expect(screen.getByTestId('left-icon')).toBeInTheDocument();
    });

    it('renders with right element', () => {
      render(<Input rightElement={<span data-testid="right-icon">#</span>} />);
      expect(screen.getByTestId('right-icon')).toBeInTheDocument();
    });

    it('renders search icon for search type', () => {
      const { container } = render(<Input type="search" />);
      // Search icon is Lucide Search.
      // We can check if wrapper div exists or icon SVG.
      const wrapper = container.querySelector('.pl-10'); // Search/Left element adds padding left
      expect(wrapper).toBeInTheDocument();
    });
  });

  describe('Interactions', () => {
    it('handles onChange events', async () => {
      const handleChange = vi.fn();
      render(<Input onChange={handleChange} />);

      const input = screen.getByRole('textbox');
      await userEvent.type(input, 'hello');

      expect(handleChange).toHaveBeenCalled();
    });

    it('can be disabled', () => {
      render(<Input disabled />);
      expect(screen.getByRole('textbox')).toBeDisabled();
    });

    it('handles clear button', async () => {
      const handleClear = vi.fn();
      render(<Input value="test" clearable onClear={handleClear} onChange={() => {}} />);

      const clearButton = screen.getByRole('button', { name: /clear/i });
      await userEvent.click(clearButton);

      expect(handleClear).toHaveBeenCalled();
    });
  });

  describe('Password Input', () => {
    it('toggles password visibility (uncontrolled)', async () => {
      render(<Input type="password" label="Password" />);

      const input = screen.getByLabelText('Password');
      expect(input).toHaveAttribute('type', 'password');

      const toggleButton = screen.getByRole('button', { name: /show password/i });
      await userEvent.click(toggleButton);

      expect(input).toHaveAttribute('type', 'text');
    });

    it('toggles password visibility (controlled)', async () => {
      const handleToggle = vi.fn();
      const { rerender } = render(
        <Input 
          type="password" 
          label="Password" 
          showPassword={false} 
          onTogglePassword={handleToggle} 
        />
      );

      const input = screen.getByLabelText('Password');
      expect(input).toHaveAttribute('type', 'password');

      const toggleButton = screen.getByRole('button', { name: /show password/i });
      await userEvent.click(toggleButton);

      expect(handleToggle).toHaveBeenCalledWith(true);
      // Should likely still be password because we haven't updated props, 
      // but let's check basic interaction first.
      expect(input).toHaveAttribute('type', 'password');

      // Now rerender with new state
      rerender(
        <Input 
          type="password" 
          label="Password" 
          showPassword={true} 
          onTogglePassword={handleToggle} 
        />
      );
      expect(input).toHaveAttribute('type', 'text');
    });
  });

  describe('Accessibility', () => {
    it('has proper label association', () => {
      render(<Input label="Email" id="email-input" />);
      const label = screen.getByText('Email');
      const input = screen.getByRole('textbox');

      expect(label).toHaveAttribute('for', 'email-input');
      expect(input).toHaveAttribute('id', 'email-input');
    });

    it('has aria-invalid when error is present', () => {
      render(<Input error="Invalid" />);
      expect(screen.getByRole('textbox')).toHaveAttribute('aria-invalid', 'true');
    });

    it('has aria-describedby pointing to error message', () => {
      render(<Input error="Error message" id="test-input" />);
      const input = screen.getByRole('textbox');
      expect(input).toHaveAttribute('aria-describedby', 'test-input-error');
    });
  });

  describe('Variants and Sizes', () => {
    it('renders default size', () => {
      render(<Input data-testid="input" />);
      const input = screen.getByTestId('input');
      expect(input).toHaveClass('h-10');
    });

    it('renders small size', () => {
      render(<Input inputSize="sm" data-testid="input" />);
      const input = screen.getByTestId('input');
      expect(input).toHaveClass('h-9');
    });

    it('renders large size', () => {
      render(<Input inputSize="lg" data-testid="input" />);
      const input = screen.getByTestId('input');
      expect(input).toHaveClass('h-11');
    });
  });
});
