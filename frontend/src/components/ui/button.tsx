/**
 * Button Component
 * 
 * Purpose: Primary interactive element for user actions
 * Supports multiple variants, sizes, loading states, and icons
 * 
 * Design decisions:
 * - Uses Radix Slot for asChild pattern (render as link, etc.)
 * - Class Variance Authority for variant management
 * - Accessible by default with proper ARIA attributes
 * - Loading state disables interaction and shows spinner
 * 
 * Accessibility:
 * - Proper button semantics
 * - Focus visible states
 * - Disabled state prevents interaction
 * - Loading state announces to screen readers
 * 
 * Testing strategy:
 * - Unit test all variants and sizes
 * - Test click handlers
 * - Test loading/disabled states
 * - Test keyboard interaction
 */

import * as React from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

// ============================================================================
// Styles
// ============================================================================

const buttonVariants = cva(
  // Base styles
  [
    'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium',
    'ring-offset-background transition-colors',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
    'disabled:pointer-events-none disabled:opacity-50',
    '[&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
  ],
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        outline: 'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'text-primary underline-offset-4 hover:underline',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-9 rounded-md px-3',
        lg: 'h-11 rounded-md px-8',
        icon: 'h-10 w-10',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
);

// ============================================================================
// Types
// ============================================================================

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  /** Render as a different element using Radix Slot */
  asChild?: boolean;
  /** Show loading spinner and disable interactions */
  isLoading?: boolean;
  /** Icon to show on the left side */
  leftIcon?: React.ReactNode;
  /** Icon to show on the right side */
  rightIcon?: React.ReactNode;
  /** Make button full width */
  fullWidth?: boolean;
}

// ============================================================================
// Component
// ============================================================================

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      asChild = false,
      isLoading = false,
      leftIcon,
      rightIcon,
      fullWidth = false,
      disabled,
      children,
      ...props
    },
    ref
  ) => {

    const isDisabled = disabled || isLoading;

    if (asChild) {
      return (
        <Slot
          className={cn(
            buttonVariants({ variant, size, className }),
            fullWidth && 'w-full'
          )}
          ref={ref}
          {...props}
        >
          {children}
        </Slot>
      );
    }

    return (
      <button
        className={cn(
          buttonVariants({ variant, size, className }),
          fullWidth && 'w-full',
          isLoading && 'relative'
        )}
        ref={ref}
        disabled={isDisabled}
        aria-disabled={isDisabled}
        aria-busy={isLoading}
        {...props}
      >
        {/* Loading spinner */}
        {isLoading && (
          <Loader2 className="absolute animate-spin" aria-hidden="true" />
        )}

        {/* Left icon */}
        {leftIcon && !isLoading && (
          <span className="shrink-0" aria-hidden="true">
            {leftIcon}
          </span>
        )}

        {/* Content - invisible when loading but maintains width */}
        <span className={cn(isLoading && 'invisible')}>{children}</span>

        {/* Right icon */}
        {rightIcon && !isLoading && (
          <span className="shrink-0" aria-hidden="true">
            {rightIcon}
          </span>
        )}

        {/* Screen reader loading announcement */}
        {isLoading && (
          <span className="sr-only" role="status">
            Loading...
          </span>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';

// ============================================================================
// Exports
// ============================================================================

export { Button, buttonVariants };
