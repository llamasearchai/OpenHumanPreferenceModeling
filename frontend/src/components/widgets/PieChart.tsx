/**
 * Pie Chart Widget
 *
 * Purpose: Reusable pie chart component using Recharts
 */

import {
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface PieChartDataPoint {
  name: string;
  value: number;
  color?: string;
}

interface PieChartProps {
  data: PieChartDataPoint[];
  title?: string;
  description?: string;
  isLoading?: boolean;
  height?: number;
  colors?: string[];
  showLegend?: boolean;
  showLabel?: boolean;
  formatTooltip?: (value: number) => string;
  onSegmentClick?: (data: PieChartDataPoint, index: number) => void;
}

const DEFAULT_COLORS = [
  'hsl(var(--primary))',
  '#22c55e',
  '#3b82f6',
  '#f59e0b',
  '#ef4444',
  '#8b5cf6',
  '#06b6d4',
  '#f97316',
];

export function PieChart({
  data,
  title,
  description,
  isLoading,
  height = 300,
  colors = DEFAULT_COLORS,
  showLegend = true,
  showLabel = false,
  formatTooltip,
  onSegmentClick,
}: PieChartProps) {
  if (isLoading) {
    return <Skeleton className="w-full" style={{ height }} />;
  }

  if (!data || data.length === 0) {
    return (
      <Card>
        {title && (
          <CardHeader>
            <CardTitle>{title}</CardTitle>
            {description && <CardDescription>{description}</CardDescription>}
          </CardHeader>
        )}
        <CardContent>
          <div
            className="flex items-center justify-center text-muted-foreground"
            style={{ height }}
          >
            No data available
          </div>
        </CardContent>
      </Card>
    );
  }

  const getColor = (index: number, item?: PieChartDataPoint) => {
    if (item?.color) return item.color;
    return colors[index % colors.length];
  };

  const total = data.reduce((sum, item) => sum + item.value, 0);

  return (
    <Card>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
      )}
      <CardContent>
        <div style={{ height, width: '100%' }}>
          <ResponsiveContainer width="100%" height="100%">
            <RechartsPieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={showLabel ? ({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%` : false}
                outerRadius={height * 0.3}
                fill="#8884d8"
                dataKey="value"
                {...(onSegmentClick ? { onClick: onSegmentClick } : {})}
              >
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getColor(index, entry)} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--popover))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                }}
                formatter={(value: number) => {
                  const formatted = formatTooltip ? formatTooltip(value) : value.toFixed(2);
                  const percentage = ((value / total) * 100).toFixed(1);
                  return [`${formatted} (${percentage}%)`, 'Value'];
                }}
              />
              {showLegend && <Legend />}
            </RechartsPieChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
