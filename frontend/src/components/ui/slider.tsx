/**
 * Slider Component
 *
 * Purpose: Range input for selecting numeric values
 * Used for confidence scoring in annotations
 */

import * as React from 'react';
import * as SliderPrimitive from '@radix-ui/react-slider';
import { cn } from '@/lib/utils';

interface SliderProps
  extends React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root> {
  /** Labels for each step (optional) */
  stepLabels?: string[];
  /** Show current value label */
  showValue?: boolean;
  /**
   * Accessible label for the slider thumb (recommended).
   * If omitted, falls back to the Root's aria-label (if provided) or a generic label.
   */
  thumbAriaLabel?: string;
}

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  SliderProps
>(({ className, stepLabels, showValue, value, thumbAriaLabel, ...props }, ref) => {
  const currentValue = value?.[0] ?? props.defaultValue?.[0] ?? 0;
  const resolvedThumbLabel =
    thumbAriaLabel ??
    (props['aria-label'] as string | undefined) ??
    'Slider';

  return (
    <div className="w-full space-y-2">
      <SliderPrimitive.Root
        ref={ref}
        {...(value !== undefined ? { value } : {})}
        className={cn(
          'relative flex w-full touch-none select-none items-center',
          className
        )}
        {...props}
      >
        <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-secondary">
          <SliderPrimitive.Range className="absolute h-full bg-primary" />
        </SliderPrimitive.Track>
        <SliderPrimitive.Thumb
          aria-label={resolvedThumbLabel}
          className="block h-5 w-5 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
        />
      </SliderPrimitive.Root>

      {/* Step labels */}
      {stepLabels && stepLabels.length > 0 && (
        <div className="flex justify-between px-1">
          {stepLabels.map((label, index) => (
            <span
              key={index}
              className={cn(
                'text-xs text-muted-foreground',
                index === currentValue - (props.min ?? 0) && 'font-medium text-foreground'
              )}
            >
              {label}
            </span>
          ))}
        </div>
      )}

      {/* Current value display */}
      {showValue && (
        <div className="text-center text-sm font-medium">{currentValue}</div>
      )}
    </div>
  );
});
Slider.displayName = SliderPrimitive.Root.displayName;

export { Slider };
