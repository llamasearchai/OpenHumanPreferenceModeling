/**
 * Bar Chart Widget
 *
 * Purpose: Reusable bar chart component using Recharts
 */

import {
  BarChart as RechartsBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from 'recharts';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface BarChartDataPoint {
  name: string;
  value: number;
  [key: string]: string | number;
}

interface BarChartProps {
  data: BarChartDataPoint[];
  title?: string;
  description?: string;
  isLoading?: boolean;
  height?: number;
  dataKey?: string;
  xAxisKey?: string;
  color?: string | string[];
  showGrid?: boolean;
  showLegend?: boolean;
  xAxisLabel?: string;
  yAxisLabel?: string;
  formatYAxis?: (value: number) => string;
  formatTooltip?: (value: number) => string;
  onBarClick?: (data: BarChartDataPoint, index: number) => void;
}

export function BarChart({
  data,
  title,
  description,
  isLoading,
  height = 300,
  dataKey = 'value',
  xAxisKey = 'name',
  color = 'hsl(var(--primary))',
  showGrid = true,
  showLegend = true,
  xAxisLabel,
  yAxisLabel,
  formatYAxis,
  formatTooltip,
  onBarClick,
}: BarChartProps) {
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

  const colors = Array.isArray(color) ? color : [color];
  const getColor = (index: number) => colors[index % colors.length];

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
            <RechartsBarChart
              data={data}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              {showGrid && <CartesianGrid strokeDasharray="3 3" opacity={0.3} />}
              <XAxis
                dataKey={xAxisKey}
                tick={{ fontSize: 12 }}
                {...(xAxisLabel ? { label: { value: xAxisLabel, position: 'insideBottom', offset: -5 } } : {})}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                {...(yAxisLabel ? { label: { value: yAxisLabel, angle: -90, position: 'insideLeft' } } : {})}
                {...(formatYAxis ? { tickFormatter: formatYAxis } : {})}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--popover))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '6px',
                }}
                formatter={(value: number) => [
                  formatTooltip ? formatTooltip(value) : value.toFixed(2),
                  dataKey,
                ]}
              />
              {showLegend && <Legend />}
              <Bar
                dataKey={dataKey}
                fill={colors[0]}
                radius={[4, 4, 0, 0]}
                {...(onBarClick ? { onClick: onBarClick } : {})}
              >
                {data.map((_entry, index) => (
                  <Cell key={`cell-${index}`} fill={getColor(index)} />
                ))}
              </Bar>
            </RechartsBarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
