/**
 * Time Series Chart Widget
 *
 * Purpose: Reusable time series chart component using Recharts
 */

import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface TimeSeriesDataPoint {
  timestamp: string | Date;
  value: number;
  label?: string;
}

interface TimeSeriesChartProps {
  data: TimeSeriesDataPoint[];
  title?: string;
  description?: string;
  isLoading?: boolean;
  height?: number;
  dataKey?: string;
  color?: string;
  showGrid?: boolean;
  showLegend?: boolean;
  xAxisLabel?: string;
  yAxisLabel?: string;
  formatXAxis?: (value: string | Date) => string;
  formatYAxis?: (value: number) => string;
  formatTooltip?: (value: number) => string;
}

export function TimeSeriesChart({
  data,
  title,
  description,
  isLoading,
  height = 300,
  dataKey = 'value',
  color = 'hsl(var(--primary))',
  showGrid = true,
  showLegend = true,
  xAxisLabel,
  yAxisLabel,
  formatXAxis,
  formatYAxis,
  formatTooltip,
}: TimeSeriesChartProps) {
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

  const chartData = data.map((point) => ({
    timestamp:
      formatXAxis?.(point.timestamp) ||
      (point.timestamp instanceof Date
        ? point.timestamp.toLocaleTimeString()
        : point.timestamp),
    value: point.value,
    label: point.label,
    fullTimestamp: point.timestamp instanceof Date ? point.timestamp : new Date(point.timestamp),
  }));

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
            <RechartsLineChart
              data={chartData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              {showGrid && <CartesianGrid strokeDasharray="3 3" opacity={0.3} />}
              <XAxis
                dataKey="timestamp"
                tick={{ fontSize: 12 }}
                interval="preserveStartEnd"
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
                labelFormatter={(label) => {
                  const point = chartData.find((d) => d.timestamp === label);
                  return point?.fullTimestamp
                    ? new Date(point.fullTimestamp).toLocaleString()
                    : label;
                }}
                formatter={(value: number) => [
                  formatTooltip ? formatTooltip(value) : value.toFixed(4),
                  dataKey,
                ]}
              />
              {showLegend && <Legend />}
              <Line
                type="monotone"
                dataKey={dataKey}
                stroke={color}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
                {...(title ? { name: title } : { name: dataKey })}
              />
            </RechartsLineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
