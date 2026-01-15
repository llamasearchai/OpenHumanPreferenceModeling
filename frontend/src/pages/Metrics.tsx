/**
 * Metrics Page
 *
 * Purpose: View system metrics and performance data
 */

import * as React from 'react';
import { useQuery } from '@tanstack/react-query';
import { monitoringApi } from '@/lib/api-client';
import { makeMockMetrics } from '@/lib/mock-data';
import { ApiRequestError, extractErrorMessage } from '@/lib/errors';
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
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { LineChart, Activity, RefreshCw, Download } from 'lucide-react';
import { TimeSeriesChart } from '@/components/widgets/TimeSeriesChart';
import { MetricCard } from '@/components/widgets/MetricCard';

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
      if (!result.success) {
        throw ApiRequestError.fromResponse(result);
      }
      return result.data;
    },
    refetchInterval: 30000,
  });

  const resolvedMetrics = metrics?.length
    ? metrics
    : error
      ? makeMockMetrics(selectedMetric)
      : [];
  const usingMockData = !!error && (!metrics || metrics.length === 0);

  const handleExport = () => {
    if (!resolvedMetrics.length) return;

    const csv = [
      ['timestamp', 'value', 'name'],
      ...resolvedMetrics.map((m) => [m.timestamp, m.value, m.name]),
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

  const currentValue = resolvedMetrics[resolvedMetrics.length - 1]?.value;
  const avgValue = resolvedMetrics.length
    ? resolvedMetrics.reduce((acc, m) => acc + m.value, 0) / resolvedMetrics.length
    : 0;
  const minValue = resolvedMetrics.length
    ? Math.min(...resolvedMetrics.map((m) => m.value))
    : 0;
  const maxValue = resolvedMetrics.length
    ? Math.max(...resolvedMetrics.map((m) => m.value))
    : 0;

  // Memoize chart data transformation - must be called unconditionally
  const chartData = React.useMemo(() => {
    if (!resolvedMetrics.length) return [];
    return resolvedMetrics.map((m, index) => ({
      timestamp: new Date(m.timestamp).toLocaleTimeString(),
      value: m.value,
      fullTimestamp: m.timestamp,
      index,
    }));
  }, [resolvedMetrics]);

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
            <SelectTrigger className="w-48" aria-label="Metric">
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
        <MetricCard
          title="Current"
          value={currentValue?.toFixed(3) || 'N/A'}
          icon={Activity}
          isLoading={isLoading}
          data-testid="current-value"
        />
        <MetricCard
          title="Average"
          value={avgValue.toFixed(3)}
          icon={LineChart}
          isLoading={isLoading}
        />
        <MetricCard
          title="Min"
          value={minValue.toFixed(3)}
          icon={LineChart}
          isLoading={isLoading}
        />
        <MetricCard
          title="Max"
          value={maxValue.toFixed(3)}
          icon={LineChart}
          isLoading={isLoading}
        />
      </div>

      {usingMockData && (
        <Alert variant="warning">
          <AlertTitle>Using mock data</AlertTitle>
          <AlertDescription>
            Metrics data could not be loaded. Showing mock values to validate charts.
          </AlertDescription>
        </Alert>
      )}

      {error && !usingMockData && (
        <Alert variant="destructive">
          <AlertTitle>Metrics unavailable</AlertTitle>
          <AlertDescription>
            {extractErrorMessage(error, 'Failed to load metrics')}
          </AlertDescription>
        </Alert>
      )}

      {/* Chart */}
      <TimeSeriesChart
        title={availableMetrics.find((m) => m.value === selectedMetric)?.label || 'Metrics'}
        description="Time series data for the selected metric"
        data={chartData.map((d) => ({ timestamp: d.fullTimestamp, value: d.value }))}
        isLoading={isLoading}
        height={300}
        formatYAxis={(value) => value.toFixed(3)}
        formatTooltip={(value) => value.toFixed(4)}
      />

      {/* Data Table */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Data Points</CardTitle>
          <CardDescription>
            Last {Math.min(resolvedMetrics.length || 0, 10)} measurements
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-10 w-full" />
              ))}
            </div>
          ) : resolvedMetrics.length ? (
            <div className="rounded-md border">
              <table className="w-full" aria-label="Recent metrics data">
                <caption className="sr-only">
                  Table showing the last {Math.min(resolvedMetrics.length || 0, 10)} metric measurements
                </caption>
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th scope="col" className="p-3 text-left text-sm font-medium">Timestamp</th>
                    <th scope="col" className="p-3 text-left text-sm font-medium">Value</th>
                    <th scope="col" className="p-3 text-left text-sm font-medium">Tags</th>
                  </tr>
                </thead>
                <tbody>
                  {resolvedMetrics.slice(-10).reverse().map((metric, i) => (
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
