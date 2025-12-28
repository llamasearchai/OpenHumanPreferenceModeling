/**
 * Dashboard Page
 *
 * Purpose: Main overview with health status, metrics, and quick actions
 */

import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  Bell,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import { healthApi, monitoringApi } from '@/lib/api-client';
import { Button } from '@/components/ui/button';
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
import { StatCard } from '@/components/widgets/StatCard';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export default function DashboardPage() {
  const {
    data: healthData,
    isLoading: healthLoading,
    error: healthError,
    refetch: refetchHealth,
  } = useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const result = await healthApi.check();
      if (!result.success) throw new Error(result.error.detail);
      return result.data;
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const {
    data: alertsData,
    isLoading: alertsLoading,
  } = useQuery({
    queryKey: ['alerts'],
    queryFn: async () => {
      const result = await monitoringApi.getAlerts();
      if (!result.success) throw new Error(result.error.detail);
      return result.data;
    },
    refetchInterval: 30000,
  });

  const getStatusIcon = (status: string) => {
    return status === 'healthy' ? (
      <CheckCircle2 className="h-5 w-5 text-green-500" />
    ) : (
      <XCircle className="h-5 w-5 text-destructive" />
    );
  };

  const getStatusBadge = (status: string) => {
    return (
      <Badge variant={status === 'healthy' ? 'success' : 'destructive'}>
        {status}
      </Badge>
    );
  };

  const services = healthData
    ? [
        { name: 'Encoder', status: healthData.encoder },
        { name: 'DPO Pipeline', status: healthData.dpo },
        { name: 'Monitoring', status: healthData.monitoring },
        { name: 'Privacy Engine', status: healthData.privacy },
      ]
    : [];

  const firingAlerts = alertsData?.filter((a) => a.status === 'firing') || [];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            System overview and health status
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetchHealth()}
          disabled={healthLoading}
        >
          <RefreshCw className={`mr-2 h-4 w-4 ${healthLoading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Health Status Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {healthLoading
          ? Array.from({ length: 4 }).map((_, i) => (
              <Card key={i}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-5 w-5 rounded-full" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-6 w-16" />
                </CardContent>
              </Card>
            ))
          : services.map((service) => (
              <Card key={service.name}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    {service.name}
                  </CardTitle>
                  {getStatusIcon(service.status)}
                </CardHeader>
                <CardContent>
                  {getStatusBadge(service.status)}
                </CardContent>
              </Card>
            ))}
      </div>

      {/* Error display */}
      {healthError && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error loading health status</AlertTitle>
          <AlertDescription>
            {healthError instanceof Error ? healthError.message : 'Failed to connect to server'}
          </AlertDescription>
        </Alert>
      )}

      {/* Active Alerts */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Bell className="h-5 w-5" />
                Active Alerts
              </CardTitle>
              <CardDescription>
                {firingAlerts.length > 0
                  ? `${firingAlerts.length} alert(s) require attention`
                  : 'No active alerts'}
              </CardDescription>
            </div>
            {firingAlerts.length > 0 && (
              <Badge variant="destructive">{firingAlerts.length}</Badge>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {alertsLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : firingAlerts.length > 0 ? (
            <div className="space-y-2">
              {firingAlerts.slice(0, 5).map((alert) => (
                <div
                  key={alert.id}
                  className="flex items-center justify-between rounded-lg border p-3"
                >
                  <div className="flex items-center gap-3">
                    <AlertTriangle
                      className={`h-4 w-4 ${
                        alert.severity === 'critical'
                          ? 'text-destructive'
                          : 'text-yellow-500'
                      }`}
                    />
                    <div>
                      <p className="text-sm font-medium">{alert.rule_name}</p>
                      <p className="text-xs text-muted-foreground">
                        {alert.message}
                      </p>
                    </div>
                  </div>
                  <Badge
                    variant={
                      alert.severity === 'critical' ? 'destructive' : 'warning'
                    }
                  >
                    {alert.severity}
                  </Badge>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <CheckCircle2 className="h-12 w-12 text-green-500 mb-4" />
              <p className="text-lg font-medium">All Clear</p>
              <p className="text-sm text-muted-foreground">
                No active alerts at this time
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quick Stats */}
      <div className="grid gap-4 md:grid-cols-3">
        <StatCard
          title="System Status"
          value={healthError ? 'Offline' : 'Operational'}
          icon={Activity}
          description={healthError ? 'Connection issues detected' : 'All systems running normally'}
          isLoading={healthLoading}
        />

        <StatCard
          title="Healthy Services"
          value={`${services.filter((s) => s.status === 'healthy').length}/${services.length}`}
          icon={CheckCircle2}
          description="Services online"
          isLoading={healthLoading}
        />

        <StatCard
          title="Active Alerts"
          value={firingAlerts.length}
          icon={Bell}
          description="Require attention"
          isLoading={alertsLoading}
        />
      </div>

      {/* Additional Dashboard Sections */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="activity">Recent Activity</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>System Health</CardTitle>
                <CardDescription>Current status of all services</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {services.map((service) => (
                    <div
                      key={service.name}
                      className="flex items-center justify-between p-3 rounded-lg border"
                    >
                      <div className="flex items-center gap-3">
                        {getStatusIcon(service.status)}
                        <span className="font-medium">{service.name}</span>
                      </div>
                      {getStatusBadge(service.status)}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Quick Actions</CardTitle>
                <CardDescription>Common tasks and shortcuts</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-2">
                  <Button variant="outline" className="justify-start" asChild>
                    <a href="/annotations">Start Annotating</a>
                  </Button>
                  <Button variant="outline" className="justify-start" asChild>
                    <a href="/metrics">View Metrics</a>
                  </Button>
                  <Button variant="outline" className="justify-start" asChild>
                    <a href="/calibration">Check Calibration</a>
                  </Button>
                  <Button variant="outline" className="justify-start" asChild>
                    <a href="/training">Monitor Training</a>
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
              <CardDescription>Key performance indicators</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-4">
                <StatCard
                  title="Avg Response Time"
                  value="142ms"
                  description="Last 24 hours"
                  change={{ value: -5.2, label: 'vs yesterday' }}
                />
                <StatCard
                  title="Success Rate"
                  value="99.8%"
                  description="Last 24 hours"
                  change={{ value: 0.1, label: 'vs yesterday' }}
                />
                <StatCard
                  title="Active Users"
                  value="47"
                  description="Currently online"
                  change={{ value: 12, label: 'vs last hour' }}
                />
                <StatCard
                  title="Tasks Completed"
                  value="1,234"
                  description="Today"
                  change={{ value: 8.3, label: 'vs yesterday' }}
                />
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="activity" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
              <CardDescription>Latest system events and updates</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { time: '2 minutes ago', event: 'New annotation submitted', type: 'success' },
                  { time: '15 minutes ago', event: 'Calibration completed', type: 'info' },
                  { time: '1 hour ago', event: 'Training run started', type: 'info' },
                  { time: '2 hours ago', event: 'Alert resolved', type: 'success' },
                ].map((activity, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-3 rounded-lg border"
                  >
                    <div>
                      <p className="text-sm font-medium">{activity.event}</p>
                      <p className="text-xs text-muted-foreground">{activity.time}</p>
                    </div>
                    <Badge variant={activity.type === 'success' ? 'success' : 'secondary'}>
                      {activity.type}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
