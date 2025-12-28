/**
 * Stat Card Widget
 *
 * Purpose: Simple stat display card with icon and optional comparison
 */

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { LucideIcon, TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface StatCardProps {
  title: string;
  value: string | number;
  icon?: LucideIcon;
  change?: {
    value: number;
    label: string;
  };
  isLoading?: boolean;
  className?: string;
  description?: string;
}

export function StatCard({
  title,
  value,
  icon: Icon,
  change,
  isLoading,
  className,
  description,
}: StatCardProps) {
  if (isLoading) {
    return (
      <Card className={className}>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <Skeleton className="h-4 w-24" />
          {Icon && <Skeleton className="h-4 w-4" />}
        </CardHeader>
        <CardContent>
          <Skeleton className="h-8 w-20" />
        </CardContent>
      </Card>
    );
  }

  const isPositive = change ? change.value >= 0 : null;
  const TrendIcon = isPositive === true ? TrendingUp : isPositive === false ? TrendingDown : null;

  return (
    <Card className={className}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {Icon && <Icon className="h-4 w-4 text-muted-foreground" />}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {description && (
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        )}
        {change && (
          <div className="flex items-center gap-1 mt-2 text-xs">
            {TrendIcon && (
              <TrendIcon
                className={cn(
                  'h-3 w-3',
                  isPositive ? 'text-green-500' : 'text-red-500'
                )}
              />
            )}
            <span
              className={cn(
                isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
              )}
            >
              {isPositive ? '+' : ''}
              {change.value}%
            </span>
            <span className="text-muted-foreground">{change.label}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
