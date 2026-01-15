/**
 * Input Component
 *
 * Purpose: Text input fields with various types and states
 * Supports text, email, password, search, number inputs
 */

import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';
import { Eye, EyeOff, Search, X } from 'lucide-react';

const inputVariants = cva(
  [
    'flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm',
    'ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium',
    'placeholder:text-muted-foreground',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
    'disabled:cursor-not-allowed disabled:opacity-50',
  ],
  {
    variants: {
      variant: {
        default: '',
        error: 'border-destructive focus-visible:ring-destructive',
      },
      inputSize: {
        default: 'h-10',
        sm: 'h-9 text-xs',
        lg: 'h-11 text-base',
      },
    },
    defaultVariants: {
      variant: 'default',
      inputSize: 'default',
    },
  }
);

export interface InputProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'size'>,
    VariantProps<typeof inputVariants> {
  /** Error message to display */
  error?: string | undefined;
  /** Label for the input */
  label?: string;
  /** Helper text below input */
  helperText?: string;
  /** Left icon/element */
  leftElement?: React.ReactNode;
  /** Right icon/element */
  rightElement?: React.ReactNode;
  /** Show clear button for text inputs */
  clearable?: boolean;
  /** Callback when clear button clicked */
  /** Callback when clear button clicked */
  onClear?: () => void;
  /** Controlled password visibility state */
  showPassword?: boolean;
  /** Callback for toggling password visibility */
  onTogglePassword?: (show: boolean) => void;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      type = 'text',
      variant,
      inputSize,
      error,
      label,
      helperText,
      leftElement,
      rightElement,
      clearable,
      onClear,
      showPassword,
      onTogglePassword,
      id,
      disabled,
      value,
      ...props
    },
    ref
  ) => {
    const [internalShowPassword, setInternalShowPassword] = React.useState(false);
    const generatedId = React.useId();
    const inputId = id || generatedId;
    const errorId = `${inputId}-error`;
    const helperId = `${inputId}-helper`;

    const isPassword = type === 'password';
    const isSearch = type === 'search';
    const hasValue = value !== undefined && value !== '';

    // Determine if we are keeping track of password visibility internally or externally
    const isControlled = showPassword !== undefined;
    const currentShowPassword = isControlled ? showPassword : internalShowPassword;

    const togglePassword = () => {
      const newState = !currentShowPassword;
      if (onTogglePassword) {
        onTogglePassword(newState);
      }
      if (!isControlled) {
        setInternalShowPassword(newState);
      }
    };

    const computedVariant = error ? 'error' : variant;
    const computedType = isPassword && currentShowPassword ? 'text' : type;

    return (
      <div className="w-full space-y-1.5">
        {label && (
          <label
            htmlFor={inputId}
            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
          >
            {label}
          </label>
        )}

        <div className="relative">
          {/* Left element */}
          {(leftElement || isSearch) && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
              {leftElement || <Search className="h-4 w-4" />}
            </div>
          )}

          <input
            type={computedType}
            id={inputId}
            ref={ref}
            disabled={disabled}
            value={value}
            className={cn(
              inputVariants({ variant: computedVariant, inputSize }),
              (leftElement || isSearch) && 'pl-10',
              (rightElement || isPassword || (clearable && hasValue)) && 'pr-10',
              className
            )}
            aria-invalid={!!error}
            aria-describedby={
              error ? errorId : helperText ? helperId : undefined
            }
            {...props}
          />

          {/* Right elements */}
          <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
            {clearable && hasValue && !disabled && (
              <button
                type="button"
                onClick={onClear}
                className="text-muted-foreground hover:text-foreground"
                aria-label="Clear input"
              >
                <X className="h-4 w-4" />
              </button>
            )}

            {isPassword && (
              <button
                type="button"
                onClick={togglePassword}
                className="text-muted-foreground hover:text-foreground"
                aria-label={currentShowPassword ? 'Hide password' : 'Show password'}
              >
                {currentShowPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            )}

            {rightElement && !isPassword && rightElement}
          </div>
        </div>

        {/* Error message */}
        {error && (
          <p id={errorId} className="text-sm text-destructive" role="alert">
            {error}
          </p>
        )}

        {/* Helper text */}
        {helperText && !error && (
          <p id={helperId} className="text-sm text-muted-foreground">
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export { Input, inputVariants };
