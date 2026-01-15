/**
 * Textarea Component Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Textarea } from './textarea';

describe('Textarea', () => {
  describe('rendering', () => {
    it('renders with default props', () => {
      render(<Textarea placeholder="Enter text" />);
      expect(screen.getByPlaceholderText('Enter text')).toBeInTheDocument();
    });

    it('renders with custom className', () => {
      render(<Textarea className="custom-class" />);
      const textarea = screen.getByRole('textbox');
      expect(textarea).toHaveClass('custom-class');
    });

    it('renders with label', () => {
      render(<Textarea label="Description" />);
      expect(screen.getByText('Description')).toBeInTheDocument();
    });

    it('associates label with textarea via htmlFor', () => {
      render(<Textarea label="Description" id="desc" />);
      const label = screen.getByText('Description');
      expect(label).toHaveAttribute('for', 'desc');
    });
  });

  describe('value handling', () => {
    it('displays provided value', () => {
      render(<Textarea value="Hello world" onChange={() => {}} />);
      expect(screen.getByRole('textbox')).toHaveValue('Hello world');
    });

    it('calls onChange when value changes', async () => {
      const user = userEvent.setup();
      const handleChange = vi.fn();
      render(<Textarea onChange={handleChange} />);

      await user.type(screen.getByRole('textbox'), 'test');
      expect(handleChange).toHaveBeenCalled();
    });

    it('handles controlled component pattern', async () => {
      const ControlledTextarea = () => {
        const [value, setValue] = React.useState('');
        return (
          <Textarea
            value={value}
            onChange={(e) => setValue(e.target.value)}
          />
        );
      };

      const React = await import('react');
      const { render: renderControlled } = await import('@testing-library/react');
      const user = userEvent.setup();

      renderControlled(<ControlledTextarea />);
      await user.type(screen.getByRole('textbox'), 'typed text');
      expect(screen.getByRole('textbox')).toHaveValue('typed text');
    });
  });

  describe('error state', () => {
    it('displays error message', () => {
      render(<Textarea error="This field is required" />);
      expect(screen.getByText('This field is required')).toBeInTheDocument();
    });

    it('sets aria-invalid when error is present', () => {
      render(<Textarea error="Error message" />);
      expect(screen.getByRole('textbox')).toHaveAttribute('aria-invalid', 'true');
    });

    it('applies error styling', () => {
      render(<Textarea error="Error" />);
      expect(screen.getByRole('textbox')).toHaveClass('border-destructive');
    });

    it('error has role="alert" for accessibility', () => {
      render(<Textarea error="Error message" />);
      expect(screen.getByRole('alert')).toHaveTextContent('Error message');
    });

    it('sets aria-describedby to error id', () => {
      render(<Textarea error="Error" id="test-id" />);
      const textarea = screen.getByRole('textbox');
      expect(textarea).toHaveAttribute('aria-describedby', 'test-id-error');
    });
  });

  describe('helper text', () => {
    it('displays helper text', () => {
      render(<Textarea helperText="Enter at least 10 characters" />);
      expect(screen.getByText('Enter at least 10 characters')).toBeInTheDocument();
    });

    it('hides helper text when error is present', () => {
      render(<Textarea helperText="Helper" error="Error" />);
      expect(screen.queryByText('Helper')).not.toBeInTheDocument();
      expect(screen.getByText('Error')).toBeInTheDocument();
    });

    it('sets aria-describedby to helper id when no error', () => {
      render(<Textarea helperText="Helper" id="test-id" />);
      const textarea = screen.getByRole('textbox');
      expect(textarea).toHaveAttribute('aria-describedby', 'test-id-helper');
    });
  });

  describe('character count', () => {
    it('shows character count when showCount and maxLength are set', () => {
      render(<Textarea showCount maxLength={100} value="Hello" onChange={() => {}} />);
      expect(screen.getByText('5/100')).toBeInTheDocument();
    });

    it('does not show character count without showCount', () => {
      render(<Textarea maxLength={100} value="Hello" onChange={() => {}} />);
      expect(screen.queryByText('5/100')).not.toBeInTheDocument();
    });

    it('applies destructive style when at max length', () => {
      render(<Textarea showCount maxLength={5} value="Hello" onChange={() => {}} />);
      const countElement = screen.getByText('5/5');
      expect(countElement).toHaveClass('text-destructive');
    });

    it('counts empty string as 0', () => {
      render(<Textarea showCount maxLength={100} value="" onChange={() => {}} />);
      expect(screen.getByText('0/100')).toBeInTheDocument();
    });
  });

  describe('disabled state', () => {
    it('applies disabled attribute', () => {
      render(<Textarea disabled />);
      expect(screen.getByRole('textbox')).toBeDisabled();
    });

    it('has disabled styling', () => {
      render(<Textarea disabled />);
      expect(screen.getByRole('textbox')).toHaveClass('disabled:cursor-not-allowed');
    });
  });

  describe('HTML attributes', () => {
    it('supports rows attribute', () => {
      render(<Textarea rows={10} />);
      expect(screen.getByRole('textbox')).toHaveAttribute('rows', '10');
    });

    it('supports placeholder attribute', () => {
      render(<Textarea placeholder="Enter description..." />);
      expect(screen.getByPlaceholderText('Enter description...')).toBeInTheDocument();
    });

    it('supports required attribute', () => {
      render(<Textarea required />);
      expect(screen.getByRole('textbox')).toBeRequired();
    });

    it('supports readOnly attribute', () => {
      render(<Textarea readOnly />);
      expect(screen.getByRole('textbox')).toHaveAttribute('readOnly');
    });

    it('supports name attribute', () => {
      render(<Textarea name="description" />);
      expect(screen.getByRole('textbox')).toHaveAttribute('name', 'description');
    });

    it('respects maxLength attribute', () => {
      render(<Textarea maxLength={50} />);
      expect(screen.getByRole('textbox')).toHaveAttribute('maxLength', '50');
    });
  });

  describe('accessibility', () => {
    it('is focusable', async () => {
      const user = userEvent.setup();
      render(<Textarea />);

      await user.tab();
      expect(screen.getByRole('textbox')).toHaveFocus();
    });

    it('has visible focus ring', () => {
      render(<Textarea />);
      expect(screen.getByRole('textbox')).toHaveClass('focus-visible:ring-2');
    });

    it('generates unique id when not provided', () => {
      render(<Textarea label="Test" />);
      const textarea = screen.getByRole('textbox');
      expect(textarea).toHaveAttribute('id');
    });
  });

  describe('ref forwarding', () => {
    it('forwards ref to textarea element', () => {
      const ref = { current: null as HTMLTextAreaElement | null };
      render(<Textarea ref={ref} />);
      expect(ref.current).toBeInstanceOf(HTMLTextAreaElement);
    });
  });
});
