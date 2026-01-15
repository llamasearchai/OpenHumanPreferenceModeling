/**
 * Federated Learning Operations Console
 *
 * Purpose: Monitor and manage federated learning rounds,
 * privacy budget consumption, and client participation.
 */

import * as React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Server,
  Users,
  Shield,
  Play,
  Pause,
  AlertTriangle,
  CheckCircle2,
  Activity,
  Lock,
  Eye,
} from 'lucide-react';
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
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { useToast } from '@/hooks/use-toast';
import { ApiRequestError, extractErrorMessage } from '@/lib/errors';
import {
  mockFederatedClients,
  mockFederatedRounds,
  mockFederatedStatus,
} from '@/lib/mock-data';

import {

  RoundDetails,
  ClientParticipation,
} from '@/types/api';
import apiClient from '@/lib/api-client';

// Privacy budget gauge component
function PrivacyGauge({
  spent,
  remaining,
}: {
  spent: number;
  remaining: number;
}) {
  const total = spent + remaining;
  const percentage = (spent / total) * 100;
  const isWarning = percentage > 75;
  const isCritical = percentage > 90;

  // SVG gauge
  const radius = 80;
  const strokeWidth = 12;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = 2 * Math.PI * normalizedRadius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <svg width={radius * 2} height={radius * 2} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={radius}
          cy={radius}
          r={normalizedRadius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-muted"
        />
        {/* Progress circle */}
        <circle
          cx={radius}
          cy={radius}
          r={normalizedRadius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className={
            isCritical
              ? 'text-destructive'
              : isWarning
                ? 'text-yellow-500'
                : 'text-primary'
          }
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center">
        <span className="text-3xl font-bold">{percentage.toFixed(0)}%</span>
        <span className="text-xs text-muted-foreground">Budget Used</span>
      </div>
      <div className="mt-4 text-center">
        <div className="text-sm">
          <span className="font-mono">{spent.toFixed(2)}</span>
          <span className="text-muted-foreground"> / </span>
          <span className="font-mono">{total.toFixed(2)}</span>
          <span className="text-muted-foreground ml-1">ε</span>
        </div>
        <div className="text-xs text-muted-foreground mt-1">
          {remaining.toFixed(2)} ε remaining
        </div>
      </div>
    </div>
  );
}

// Client heatmap
function ClientHeatmap({ clients }: { clients: ClientParticipation[] }) {
  const maxRounds = 50;

  return (
    <div className="space-y-2">
      <div className="text-xs text-muted-foreground flex justify-between">
        <span>Client</span>
        <span>Recent Rounds →</span>
      </div>
      <div className="space-y-1">
        {clients.slice(0, 15).map((client) => (
          <div key={client.clientId} className="flex items-center gap-2">
            <span className="text-xs font-mono w-20 truncate">{client.clientId}</span>
            <div className="flex-1 flex gap-px">
              {Array.from({ length: Math.min(maxRounds, 30) }, (_, i) => {
                // Heuristic mapping for visualization since real history might be sparse
                const roundNum = 42 - i; // This logic might need adjustment based on real round numbering
                const participated = client.rounds.includes(roundNum);
                return (
                  <div
                    key={i}
                    className={`h-3 flex-1 rounded-sm ${
                      participated ? 'bg-primary' : 'bg-muted'
                    }`}
                    title={`Round ${roundNum}: ${participated ? 'Participated' : 'Missed'}`}
                  />
                );
              })}
            </div>
            <Badge
              variant={
                client.status === 'active'
                  ? 'success'
                  : client.status === 'straggler'
                    ? 'warning'
                    : 'destructive'
              }
              className="text-xs w-16 justify-center"
            >
              {client.status}
            </Badge>
          </div>
        ))}
      </div>
    </div>
  );
}

// Gradient distribution (simplified histogram)
function GradientDistribution({
  stats,
}: {
  stats: { meanNorm: number; maxNorm: number; noiseScale: number };
}) {
  // Generate mock distribution data based on stats for visualization
  // In a real app we'd fetch the histogram buckets from backend
  const bins = 10;
  const preNoise = Array.from({ length: bins }, (_, i) =>
    Math.exp(-((i - 3) ** 2) / 2) * stats.meanNorm
  );
  const postNoise = Array.from({ length: bins }, (_, i) =>
    Math.exp(-((i - 3) ** 2) / 4) * (stats.meanNorm + stats.noiseScale)
  );

  const maxVal = Math.max(...preNoise, ...postNoise) || 1;

  return (
    <div className="space-y-4">
      <div className="flex items-end gap-1 h-32">
        {preNoise.map((val, i) => (
          <div key={i} className="flex-1 flex flex-col items-center gap-1">
            <div
              className="w-full bg-primary/30 rounded-t"
              style={{ height: `${((postNoise[i] ?? 0) / maxVal) * 100}%` }}
            />
            <div
              className="w-full bg-primary rounded-t -mt-1"
              style={{ height: `${(val / maxVal) * 100}%` }}
            />
          </div>
        ))}
      </div>
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-primary rounded" />
          <span>Pre-noise</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-primary/30 rounded" />
          <span>Post-noise</span>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <div className="text-muted-foreground">Mean Norm</div>
          <div className="font-mono">{stats.meanNorm.toFixed(4)}</div>
        </div>
        <div>
          <div className="text-muted-foreground">Max Norm</div>
          <div className="font-mono">{stats.maxNorm.toFixed(4)}</div>
        </div>
        <div>
          <div className="text-muted-foreground">Noise Scale</div>
          <div className="font-mono">{stats.noiseScale.toFixed(4)}</div>
        </div>
      </div>
    </div>
  );
}

// Round history list
function RoundHistory({ rounds }: { rounds: RoundDetails[] }) {
  return (
    <div className="space-y-2">
      {rounds.map((round) => (
        <div
          key={round.roundId}
          className="flex items-center justify-between p-3 border rounded-lg"
        >
          <div className="flex items-center gap-3">
            {round.status === 'in_progress' ? (
              <Activity className="h-4 w-4 text-primary animate-pulse" />
            ) : round.status === 'completed' ? (
              <CheckCircle2 className="h-4 w-4 text-green-500" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-destructive" />
            )}
            <div>
              <span className="font-medium">Round {round.roundId}</span>
              <div className="text-xs text-muted-foreground">
                {round.participatingClients} / {round.selectedClients} clients
              </div>
            </div>
          </div>
          <div className="text-right">
            <Badge variant={round.status === 'in_progress' ? 'default' : 'secondary'}>
              {round.status.replace('_', ' ')}
            </Badge>
            <div className="text-xs text-muted-foreground mt-1">
              {new Date(round.startedAt).toLocaleTimeString()}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// Main component
export default function FederatedLearningPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Queries
  const { data: status, isLoading: statusLoading, error: statusError } = useQuery({
    queryKey: ['federated', 'status'],
    queryFn: async () => {
      const result = await apiClient.federated.getStatus();
      if (!result.success) {
        throw ApiRequestError.fromResponse(result);
      }
      return result.data;
    },
    refetchInterval: 2000,
  });

  const { data: rounds, isLoading: roundsLoading, error: roundsError } = useQuery({
    queryKey: ['federated', 'rounds'],
    queryFn: async () => {
      const result = await apiClient.federated.getRounds();
      if (!result.success) {
        throw ApiRequestError.fromResponse(result);
      }
      return result.data;
    },
    refetchInterval: 5000,
  });

  const { data: clients, isLoading: clientsLoading, error: clientsError } = useQuery({
    queryKey: ['federated', 'clients'],
    queryFn: async () => {
      const result = await apiClient.federated.getClients();
      if (!result.success) {
        throw ApiRequestError.fromResponse(result);
      }
      return result.data;
    },
    refetchInterval: 10000,
  });

  // Mutations
  const startRoundMutation = useMutation({
    mutationFn: async () => {
      const result = await apiClient.federated.startRound();
      if (!result.success) {
        throw ApiRequestError.fromResponse(result);
      }
      return result.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['federated'] });
      toast({ title: 'Round started', variant: 'success' });
    },
    onError: (err) => {
      toast({ title: 'Error', description: extractErrorMessage(err), variant: 'destructive' });
    }
  });

  const pauseTrainingMutation = useMutation({
    mutationFn: async () => {
      const result = await apiClient.federated.pauseTraining();
      if (!result.success) {
        throw ApiRequestError.fromResponse(result);
      }
      return result.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['federated'] });
      toast({ title: 'Training paused', variant: 'default' });
    },
    onError: (err) => {
      toast({ title: 'Error', description: extractErrorMessage(err), variant: 'destructive' });
    }
  });

  const resolvedStatus = status ?? (statusError ? mockFederatedStatus : undefined);
  const resolvedRounds = rounds?.length ? rounds : roundsError ? mockFederatedRounds : [];
  const resolvedClients = clients?.length ? clients : clientsError ? mockFederatedClients : [];
  const usingMockData = !!statusError || !!roundsError || !!clientsError;

  const latestRound = resolvedRounds?.[0];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Federated Learning</h1>
          <p className="text-muted-foreground">
            Privacy-preserving distributed training across edge devices
          </p>
        </div>
        <div className="flex items-center gap-2">
          {resolvedStatus?.isActive ? (
            <Button
              variant="outline"
              onClick={() => pauseTrainingMutation.mutate()}
              disabled={pauseTrainingMutation.isPending}
            >
              <Pause className="mr-2 h-4 w-4" />
              Pause Training
            </Button>
          ) : (
            <Button
              onClick={() => startRoundMutation.mutate()}
              disabled={startRoundMutation.isPending}
            >
              <Play className="mr-2 h-4 w-4" />
              Start Round
            </Button>
          )}
        </div>
      </div>

      {usingMockData && (
        <Alert variant="warning">
          <AlertTitle>Using mock data</AlertTitle>
          <AlertDescription>
            Federated learning data could not be loaded. Showing mock data for validation.
          </AlertDescription>
        </Alert>
      )}

      {(statusError || roundsError || clientsError) && !usingMockData && (
        <Alert variant="destructive">
          <AlertTitle>Federated data unavailable</AlertTitle>
          <AlertDescription>
            {extractErrorMessage(statusError || roundsError || clientsError, 'Failed to load federated learning data')}
          </AlertDescription>
        </Alert>
      )}

      {/* Status Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Current Round</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <>
                <div className="text-2xl font-bold">{resolvedStatus?.round}</div>
                <Badge variant={resolvedStatus?.isActive ? 'default' : 'secondary'}>
                  {resolvedStatus?.isActive ? 'Active' : 'Paused'}
                </Badge>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Clients Online</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <>
                <div className="text-2xl font-bold">
                  {resolvedStatus?.activeClients} / {resolvedStatus?.totalClients}
                </div>
                <Progress
                  value={((resolvedStatus?.activeClients || 0) / (resolvedStatus?.totalClients || 1)) * 100}
                  className="mt-2"
                />
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Privacy Steps</CardTitle>
            <Lock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <>
                <div className="text-2xl font-bold font-mono">
                  {resolvedStatus?.privacyBudget.totalSteps.toLocaleString()}
                </div>
                <p className="text-xs text-muted-foreground">
                  δ = {resolvedStatus?.privacyBudget.delta}
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Version</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <>
                <div className="text-lg font-mono">{resolvedStatus?.modelChecksum}</div>
                <p className="text-xs text-muted-foreground">
                  Updated {new Date(resolvedStatus?.lastUpdated || '').toLocaleTimeString()}
                </p>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Privacy Budget */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Privacy Budget
            </CardTitle>
            <CardDescription>
              Differential privacy epsilon consumption
            </CardDescription>
          </CardHeader>
          <CardContent className="flex justify-center py-6">
            {statusLoading ? (
              <Skeleton className="h-40 w-40 rounded-full" />
            ) : resolvedStatus ? (
              <PrivacyGauge
                spent={resolvedStatus.privacyBudget.epsilonSpent}
                remaining={resolvedStatus.privacyBudget.epsilonRemaining}
              />
            ) : null}
          </CardContent>
        </Card>

        {/* Gradient Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Gradient Analysis
            </CardTitle>
            <CardDescription>
              Pre vs post-noise gradient distributions
            </CardDescription>
          </CardHeader>
          <CardContent>
            {roundsLoading || !latestRound ? (
              <Skeleton className="h-48 w-full" />
            ) : (
              <GradientDistribution stats={latestRound.gradientStats} />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Tabs for details */}
      <Tabs defaultValue="rounds">
        <TabsList>
          <TabsTrigger value="rounds">Round History</TabsTrigger>
          <TabsTrigger value="clients">Client Participation</TabsTrigger>
        </TabsList>

        <TabsContent value="rounds">
          <Card>
            <CardHeader>
              <CardTitle>Training Rounds</CardTitle>
              <CardDescription>
                History of federated learning rounds
              </CardDescription>
            </CardHeader>
            <CardContent>
              {roundsLoading ? (
                <div className="space-y-2">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : resolvedRounds.length ? (
                <RoundHistory rounds={resolvedRounds} />
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="clients">
          <Card>
            <CardHeader>
              <CardTitle>Client Participation Heatmap</CardTitle>
              <CardDescription>
                Participation history across training rounds
              </CardDescription>
            </CardHeader>
            <CardContent>
              {clientsLoading ? (
                <div className="space-y-2">
                  {Array.from({ length: 10 }).map((_, i) => (
                    <Skeleton key={i} className="h-6 w-full" />
                  ))}
                </div>
              ) : resolvedClients.length ? (
                <ClientHeatmap clients={resolvedClients} />
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Privacy Alert */}
      {resolvedStatus && resolvedStatus.privacyBudget.epsilonRemaining < 2 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Privacy Budget Low</AlertTitle>
          <AlertDescription>
            Only {resolvedStatus.privacyBudget.epsilonRemaining.toFixed(2)} ε remaining.
            Training will automatically stop when budget is exhausted.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
