/**
 * Empty State Component
 *
 * Purpose: Consistent empty state display across the application
 * with support for different variants (empty, error, no-results, etc.)
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import { Button } from './button';
import {
  InboxIcon,
  AlertTriangle,
  SearchX,
  FileX,
  RefreshCw,
  Plus,
  type LucideIcon,
} from 'lucide-react';

type EmptyStateVariant = 'empty' | 'error' | 'no-results' | 'no-data' | 'custom';

interface EmptyStateProps {
  /** Title text */
  title: string;
  /** Description text */
  description?: string;
  /** Variant determines the default icon */
  variant?: EmptyStateVariant;
  /** Custom icon to override variant default */
  icon?: LucideIcon;
  /** Primary action button */
  action?: {
    label: string;
    onClick: () => void;
    variant?: 'default' | 'outline' | 'secondary';
  };
  /** Secondary action button */
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  /** Additional className */
  className?: string;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Whether to show in compact mode (less padding) */
  compact?: boolean;
}

const variantIcons: Record<EmptyStateVariant, LucideIcon> = {
  empty: InboxIcon,
  error: AlertTriangle,
  'no-results': SearchX,
  'no-data': FileX,
  custom: InboxIcon,
};

const sizeConfig = {
  sm: {
    icon: 'h-8 w-8',
    title: 'text-sm font-medium',
    description: 'text-xs',
    padding: 'py-6',
  },
  md: {
    icon: 'h-12 w-12',
    title: 'text-lg font-semibold',
    description: 'text-sm',
    padding: 'py-12',
  },
  lg: {
    icon: 'h-16 w-16',
    title: 'text-xl font-bold',
    description: 'text-base',
    padding: 'py-16',
  },
};

export function EmptyState({
  title,
  description,
  variant = 'empty',
  icon,
  action,
  secondaryAction,
  className,
  size = 'md',
  compact = false,
}: EmptyStateProps) {
  const Icon = icon || variantIcons[variant];
  const config = sizeConfig[size];

  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center text-center',
        compact ? 'py-6' : config.padding,
        className
      )}
      role="status"
      aria-label={title}
    >
      <Icon
        className={cn(
          config.icon,
          'mb-4',
          variant === 'error' ? 'text-destructive' : 'text-muted-foreground'
        )}
        aria-hidden="true"
      />
      <h3 className={cn(config.title, 'mb-2')}>{title}</h3>
      {description && (
        <p className={cn(config.description, 'text-muted-foreground max-w-sm mb-4')}>
          {description}
        </p>
      )}
      {(action || secondaryAction) && (
        <div className="flex items-center gap-2 mt-2">
          {action && (
            <Button
              variant={action.variant || 'default'}
              size={size === 'sm' ? 'sm' : 'default'}
              onClick={action.onClick}
            >
              {action.label.toLowerCase().includes('refresh') ||
              action.label.toLowerCase().includes('retry') ? (
                <RefreshCw className="mr-2 h-4 w-4" />
              ) : action.label.toLowerCase().includes('add') ||
                action.label.toLowerCase().includes('create') ? (
                <Plus className="mr-2 h-4 w-4" />
              ) : null}
              {action.label}
            </Button>
          )}
          {secondaryAction && (
            <Button
              variant="outline"
              size={size === 'sm' ? 'sm' : 'default'}
              onClick={secondaryAction.onClick}
            >
              {secondaryAction.label}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Preset empty states for common scenarios
 */
export function NoDataState({
  onRefresh,
  ...props
}: Omit<EmptyStateProps, 'variant' | 'title'> & {
  onRefresh?: () => void;
}) {
  return (
    <EmptyState
      variant="no-data"
      title="No data available"
      description="There's no data to display at this time. Try refreshing or check back later."
      action={
        onRefresh
          ? {
              label: 'Refresh',
              onClick: onRefresh,
              variant: 'outline',
            }
          : undefined
      }
      {...props}
    />
  );
}

export function NoResultsState({
  searchTerm,
  onClear,
  ...props
}: Omit<EmptyStateProps, 'variant' | 'title'> & {
  searchTerm?: string;
  onClear?: () => void;
}) {
  return (
    <EmptyState
      variant="no-results"
      title="No results found"
      description={
        searchTerm
          ? `No results found for "${searchTerm}". Try adjusting your search or filters.`
          : 'No results match your current filters. Try adjusting your search criteria.'
      }
      action={
        onClear
          ? {
              label: 'Clear filters',
              onClick: onClear,
              variant: 'outline',
            }
          : undefined
      }
      {...props}
    />
  );
}

export function ErrorState({
  error,
  onRetry,
  ...props
}: Omit<EmptyStateProps, 'variant' | 'title'> & {
  error?: string | Error;
  onRetry?: () => void;
}) {
  const errorMessage =
    typeof error === 'string'
      ? error
      : error instanceof Error
        ? error.message
        : 'An unexpected error occurred. Please try again.';

  return (
    <EmptyState
      variant="error"
      title="Something went wrong"
      description={errorMessage}
      action={
        onRetry
          ? {
              label: 'Try again',
              onClick: onRetry,
            }
          : undefined
      }
      {...props}
    />
  );
}
