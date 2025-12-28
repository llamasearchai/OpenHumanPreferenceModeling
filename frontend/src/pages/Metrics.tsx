/**
 * Metrics Page
 *
 * Purpose: View system metrics and performance data
 */

import * as React from 'react';
import { useQuery } from '@tanstack/react-query';
import { monitoringApi } from '@/lib/api-client';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { LineChart, Activity, RefreshCw, Download } from 'lucide-react';
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

const availableMetrics = [
  { value: 'encoder_latency', label: 'Encoder Latency' },
  { value: 'model_accuracy', label: 'Model Accuracy' },
  { value: 'annotation_queue_depth', label: 'Annotation Queue' },
  { value: 'error_rate', label: 'Error Rate' },
  { value: 'ece', label: 'Expected Calibration Error' },
];

export default function MetricsPage() {
  const [selectedMetric, setSelectedMetric] = React.useState('encoder_latency');

  const {
    data: metrics,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['metrics', selectedMetric],
    queryFn: async () => {
      const result = await monitoringApi.getMetrics(selectedMetric);
      if (!result.success) throw new Error(result.error.detail);
      return result.data;
    },
    refetchInterval: 30000,
  });

  const handleExport = () => {
    if (!metrics) return;

    const csv = [
      ['timestamp', 'value', 'name'],
      ...metrics.map((m) => [m.timestamp, m.value, m.name]),
    ]
      .map((row) => row.join(','))
      .join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedMetric}_metrics.csv`;
    a.click();
  };

  const currentValue = metrics?.[metrics.length - 1]?.value;
  const avgValue = metrics?.length
    ? metrics.reduce((acc, m) => acc + m.value, 0) / metrics.length
    : 0;
  const minValue = metrics?.length ? Math.min(...metrics.map((m) => m.value)) : 0;
  const maxValue = metrics?.length ? Math.max(...metrics.map((m) => m.value)) : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Metrics</h1>
          <p className="text-muted-foreground">
            Monitor system performance and health
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={selectedMetric} onValueChange={setSelectedMetric}>
            <SelectTrigger className="w-48">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {availableMetrics.map((metric) => (
                <SelectItem key={metric.value} value={metric.value}>
                  {metric.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Current</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">
                {currentValue?.toFixed(3) || 'N/A'}
              </div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Average</CardTitle>
            <LineChart className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">{avgValue.toFixed(3)}</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Min</CardTitle>
            <LineChart className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">{minValue.toFixed(3)}</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Max</CardTitle>
            <LineChart className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">{maxValue.toFixed(3)}</div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Chart Placeholder */}
      <Card>
        <CardHeader>
          <CardTitle>
            {availableMetrics.find((m) => m.value === selectedMetric)?.label}
          </CardTitle>
          <CardDescription>
            Time series data for the selected metric
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <Skeleton className="h-64 w-full" />
          ) : error ? (
            <div className="flex items-center justify-center h-64 text-muted-foreground">
              Failed to load metrics data
            </div>
          ) : !metrics?.length ? (
            <div className="flex items-center justify-center h-64 text-muted-foreground">
              No data available for this metric
            </div>
          ) : (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsLineChart
                  data={metrics.map((m) => ({
                    timestamp: new Date(m.timestamp).toLocaleTimeString(),
                    value: m.value,
                    fullTimestamp: m.timestamp,
                  }))}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis
                    dataKey="timestamp"
                    tick={{ fontSize: 12 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--popover))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '6px',
                    }}
                    labelFormatter={(label) => {
                      const metric = metrics.find(
                        (m) => new Date(m.timestamp).toLocaleTimeString() === label
                      );
                      return metric
                        ? new Date(metric.timestamp).toLocaleString()
                        : label;
                    }}
                    formatter={(value: number) => [value.toFixed(4), 'Value']}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                    name={availableMetrics.find((m) => m.value === selectedMetric)?.label || 'Value'}
                  />
                </RechartsLineChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Data Table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Data Points</CardTitle>
          <CardDescription>
            Last {Math.min(metrics?.length || 0, 10)} measurements
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : metrics?.length ? (
            <div className="rounded-md border">
              <table className="w-full">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="p-3 text-left text-sm font-medium">Timestamp</th>
                    <th className="p-3 text-left text-sm font-medium">Value</th>
                    <th className="p-3 text-left text-sm font-medium">Tags</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.slice(-10).reverse().map((metric, i) => (
                    <tr key={i} className="border-b last:border-0">
                      <td className="p-3 text-sm">
                        {new Date(metric.timestamp).toLocaleString()}
                      </td>
                      <td className="p-3 text-sm font-mono">
                        {metric.value.toFixed(4)}
                      </td>
                      <td className="p-3 text-sm text-muted-foreground">
                        {Object.entries(metric.tags || {})
                          .map(([k, v]) => `${k}=${v}`)
                          .join(', ') || '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-center text-muted-foreground py-8">
              No data available
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
