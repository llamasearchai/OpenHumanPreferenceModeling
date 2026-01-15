/**
 * Alerts Page
 *
 * Purpose: View and manage system alerts
 */

import * as React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { monitoringApi } from '@/lib/api-client';
import { mockAlerts } from '@/lib/mock-data';
import { ApiRequestError, extractErrorMessage } from '@/lib/errors';
import { useToast } from '@/hooks/use-toast';
import { Button } from '@/components/ui/button';
import { ExportButton } from '@/components/widgets/ExportButton';
import { MetricCard } from '@/components/widgets/MetricCard';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  AlertTriangle,
  Bell,
  CheckCircle2,
  Clock,
  RefreshCw,
  Check,
} from 'lucide-react';

type SeverityFilter = 'all' | 'critical' | 'warning' | 'info';

export default function AlertsPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [severityFilter, setSeverityFilter] = React.useState<SeverityFilter>('all');

  const {
    data: alerts,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['alerts'],
    queryFn: async () => {
      const result = await monitoringApi.getAlerts();
      if (!result.success) {
        throw ApiRequestError.fromResponse(result);
      }
      return result.data;
    },
    refetchInterval: 30000,
  });

  const acknowledgeMutation = useMutation({
    mutationFn: async (alertId: string) => {
      const result = await monitoringApi.acknowledgeAlert(alertId);
      if (!result.success) {
        throw ApiRequestError.fromResponse(result);
      }
      return result.data;
    },
    onSuccess: () => {
      toast({ title: 'Alert acknowledged', variant: 'success' });
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
    },
    onError: (error) => {
      toast({
        title: 'Failed to acknowledge alert',
        description: extractErrorMessage(error),
        variant: 'destructive',
      });
    },
  });

  const resolvedAlerts = alerts?.length ? alerts : error ? mockAlerts : [];
  const usingMockData = !!error && (!alerts || alerts.length === 0);

  const filteredAlerts = React.useMemo(() => {
    if (!resolvedAlerts.length) return [];
    if (severityFilter === 'all') return resolvedAlerts;
    return resolvedAlerts.filter((a) => a.severity === severityFilter);
  }, [resolvedAlerts, severityFilter]);

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <AlertTriangle className="h-4 w-4 text-destructive" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      default:
        return <Bell className="h-4 w-4 text-blue-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'firing':
        return <Badge variant="destructive">Firing</Badge>;
      case 'acknowledged':
        return <Badge variant="secondary">Acknowledged</Badge>;
      case 'resolved':
        return <Badge variant="success">Resolved</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Alerts</h1>
          <p className="text-muted-foreground">
            Monitor and manage system alerts
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select
            value={severityFilter}
            onValueChange={(value) => setSeverityFilter(value as SeverityFilter)}
          >
            <SelectTrigger className="w-32" aria-label="Severity filter">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
              <SelectItem value="warning">Warning</SelectItem>
              <SelectItem value="info">Info</SelectItem>
            </SelectContent>
          </Select>
          <ExportButton
            data={resolvedAlerts}
            filename="alerts"
            format="csv"
            variant="outline"
            size="sm"
          />
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {usingMockData && (
        <Alert variant="warning">
          <AlertTitle>Using mock data</AlertTitle>
          <AlertDescription>
            Alerts could not be loaded. Showing mock alerts for validation.
          </AlertDescription>
        </Alert>
      )}

      {error && !usingMockData && (
        <Alert variant="destructive">
          <AlertTitle>Alerts unavailable</AlertTitle>
          <AlertDescription>
            {extractErrorMessage(error, 'Failed to load alerts')}
          </AlertDescription>
        </Alert>
      )}

      {/* Alert Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <MetricCard
          title="Total Alerts"
          value={resolvedAlerts.length}
          icon={Bell}
          isLoading={isLoading}
        />
        <MetricCard
          title="Critical"
          value={resolvedAlerts.filter((a) => a.severity === 'critical').length || 0}
          icon={AlertTriangle}
          variant="destructive"
          isLoading={isLoading}
        />
        <MetricCard
          title="Firing"
          value={resolvedAlerts.filter((a) => a.status === 'firing').length || 0}
          icon={Clock}
          variant="warning"
          isLoading={isLoading}
        />
        <MetricCard
          title="Resolved"
          value={resolvedAlerts.filter((a) => a.status === 'resolved').length || 0}
          icon={CheckCircle2}
          variant="success"
          isLoading={isLoading}
        />
      </div>

      {/* Alert List */}
      <Card>
        <CardHeader>
          <CardTitle>Alert History</CardTitle>
          <CardDescription>
            {filteredAlerts.length} alert(s) matching filter
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : filteredAlerts.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <CheckCircle2 className="h-12 w-12 text-green-500 mb-4" />
              <p className="text-lg font-medium">No alerts</p>
              <p className="text-sm text-muted-foreground">
                {severityFilter === 'all'
                  ? 'All systems are operating normally'
                  : `No ${severityFilter} alerts`}
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {filteredAlerts.map((alert) => (
                <div
                  key={alert.id}
                  className="flex items-center justify-between rounded-lg border p-4"
                >
                  <div className="flex items-center gap-4">
                    {getSeverityIcon(alert.severity)}
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="font-medium">{alert.rule_name}</p>
                        {getStatusBadge(alert.status)}
                        <Badge variant="outline">{alert.severity}</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {alert.message}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        {new Date(alert.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  {alert.status === 'firing' && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => acknowledgeMutation.mutate(alert.id)}
                      disabled={acknowledgeMutation.isPending}
                    >
                      <Check className="mr-2 h-4 w-4" />
                      Acknowledge
                    </Button>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
