/**
 * Training Progress Tracker
 *
 * Purpose: Monitor DPO and SFT training runs with
 * real-time loss curves and gradient statistics.
 */

import * as React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  TrendingUp,
  Activity,
  Clock,
  Zap,
  AlertTriangle,
} from 'lucide-react';
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useRealtimeTrainingProgress } from '@/hooks/use-realtime';
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

// Types
interface TrainingRun {
  id: string;
  name: string;
  type: 'dpo' | 'sft';
  status: 'running' | 'completed' | 'failed' | 'paused';
  startedAt: string;
  completedAt: string | null;
  currentStep: number;
  totalSteps: number;
  config: {
    learningRate: number;
    batchSize: number;
    beta?: number;
    epochs: number;
  };
}

interface TrainingMetrics {
  step: number;
  loss: number;
  chosenReward?: number;
  rejectedReward?: number;
  rewardMargin?: number;
  learningRate: number;
  gradNorm: number;
  timestamp: string;
}

interface GradientStats {
  layerName: string;
  meanNorm: number;
  maxNorm: number;
  minNorm: number;
}

// Mock API functions
async function fetchRuns(): Promise<TrainingRun[]> {
  return [
    {
      id: 'run-1',
      name: 'DPO Fine-tune v2.3',
      type: 'dpo',
      status: 'running',
      startedAt: new Date(Date.now() - 3600000).toISOString(),
      completedAt: null,
      currentStep: 450,
      totalSteps: 1000,
      config: {
        learningRate: 5e-6,
        batchSize: 4,
        beta: 0.1,
        epochs: 3,
      },
    },
    {
      id: 'run-2',
      name: 'SFT Baseline',
      type: 'sft',
      status: 'completed',
      startedAt: new Date(Date.now() - 86400000).toISOString(),
      completedAt: new Date(Date.now() - 82800000).toISOString(),
      currentStep: 2000,
      totalSteps: 2000,
      config: {
        learningRate: 2e-5,
        batchSize: 8,
        epochs: 1,
      },
    },
  ];
}

async function fetchMetrics(_runId: string): Promise<TrainingMetrics[]> {
  const steps = 100;
  return Array.from({ length: steps }, (_, i) => {
    const step = i * 5;
    const baseLoss = 2.5 * Math.exp(-step / 200) + 0.3 + (Math.random() - 0.5) * 0.1;
    const chosenReward = -1 + step / 300 + (Math.random() - 0.5) * 0.2;
    const rejectedReward = -2 + step / 500 + (Math.random() - 0.5) * 0.2;

    return {
      step,
      loss: baseLoss,
      chosenReward,
      rejectedReward,
      rewardMargin: chosenReward - rejectedReward,
      learningRate: 5e-6 * (1 - step / 1000 / 2),
      gradNorm: 0.5 + Math.random() * 0.5,
      timestamp: new Date(Date.now() - (steps - i) * 60000).toISOString(),
    };
  });
}

async function fetchGradientStats(): Promise<GradientStats[]> {
  const layers = [
    'embed_tokens',
    'layers.0',
    'layers.1',
    'layers.2',
    'layers.3',
    'lm_head',
  ];
  return layers.map((name) => ({
    layerName: name,
    meanNorm: Math.random() * 0.3,
    maxNorm: Math.random() * 0.8,
    minNorm: Math.random() * 0.1,
  }));
}

// Loss chart component using Recharts
function LossChart({ metrics }: { metrics: TrainingMetrics[] }) {
  if (!metrics.length) return null;

  const chartData = metrics.map((m) => ({
    step: m.step,
    loss: m.loss,
    learningRate: m.learningRate,
    gradNorm: m.gradNorm,
  }));

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RechartsLineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="step"
            label={{ value: 'Step', position: 'insideBottom', offset: -5 }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--popover))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
            formatter={(value: number) => [value.toFixed(4), 'Loss']}
            labelFormatter={(label) => `Step: ${label}`}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
            name="Training Loss"
          />
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  );
}

// Reward margin chart using Recharts
function RewardMarginChart({ metrics }: { metrics: TrainingMetrics[] }) {
  const dpoMetrics = metrics.filter((m) => m.rewardMargin !== undefined);
  if (!dpoMetrics.length) return null;

  const chartData = dpoMetrics.map((m) => ({
    step: m.step,
    margin: m.rewardMargin!,
    chosenReward: m.chosenReward,
    rejectedReward: m.rejectedReward,
  }));

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RechartsLineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="step"
            label={{ value: 'Step', position: 'insideBottom', offset: -5 }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            label={{ value: 'Reward Margin', angle: -90, position: 'insideLeft' }}
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--popover))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
            formatter={(value: number, name: string) => {
              if (name === 'margin') return [value.toFixed(4), 'Reward Margin'];
              return [value.toFixed(4), name];
            }}
            labelFormatter={(label) => `Step: ${label}`}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="margin"
            stroke="#22c55e"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
            name="Reward Margin"
          />
          <Line
            type="monotone"
            dataKey="chosenReward"
            stroke="#3b82f6"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="5 5"
            name="Chosen Reward"
          />
          <Line
            type="monotone"
            dataKey="rejectedReward"
            stroke="#ef4444"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="5 5"
            name="Rejected Reward"
          />
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  );
}

// Gradient stats visualization
function GradientStatsViz({ stats }: { stats: GradientStats[] }) {
  const maxNorm = Math.max(...stats.map((s) => s.maxNorm));

  return (
    <div className="space-y-3">
      {stats.map((stat) => (
        <div key={stat.layerName} className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <span className="font-mono text-xs">{stat.layerName}</span>
            <span className="text-muted-foreground text-xs">
              mean: {stat.meanNorm.toFixed(4)}
            </span>
          </div>
          <div className="flex items-center gap-1 h-4">
            <div
              className="h-full bg-muted-foreground/20 rounded"
              style={{ width: `${(stat.minNorm / maxNorm) * 100}%` }}
            />
            <div
              className="h-full bg-primary rounded"
              style={{ width: `${((stat.meanNorm - stat.minNorm) / maxNorm) * 100}%` }}
            />
            <div
              className="h-full bg-primary/50 rounded"
              style={{ width: `${((stat.maxNorm - stat.meanNorm) / maxNorm) * 100}%` }}
            />
          </div>
        </div>
      ))}
      <div className="flex items-center gap-4 text-xs mt-4">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-muted-foreground/20" />
          <span>Min</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-primary" />
          <span>Mean</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-primary/50" />
          <span>Max</span>
        </div>
      </div>
    </div>
  );
}

// Main component
export default function TrainingPage() {
  const [selectedRunId, setSelectedRunId] = React.useState<string | null>(null);

  // Real-time training progress from WebSocket
  const realtimeProgress = useRealtimeTrainingProgress();

  // Queries
  const { data: runs } = useQuery({
    queryKey: ['training', 'runs'],
    queryFn: fetchRuns,
    refetchInterval: 10000,
  });

  const selectedRun = runs?.find((r) => r.id === selectedRunId) || runs?.[0];

  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['training', 'metrics', selectedRun?.id],
    queryFn: () => fetchMetrics(selectedRun!.id),
    enabled: !!selectedRun,
    refetchInterval: selectedRun?.status === 'running' ? 5000 : false,
  });

  const { data: gradStats, isLoading: gradLoading } = useQuery({
    queryKey: ['training', 'gradients', selectedRun?.id],
    queryFn: fetchGradientStats,
    enabled: !!selectedRun && selectedRun.status === 'running',
    refetchInterval: 10000,
  });

  // Calculate ETA
  const eta = React.useMemo(() => {
    if (!selectedRun || selectedRun.status !== 'running') return null;
    const elapsed = Date.now() - new Date(selectedRun.startedAt).getTime();
    const progress = selectedRun.currentStep / selectedRun.totalSteps;
    if (progress === 0) return null;
    const remaining = (elapsed / progress) * (1 - progress);
    return Math.round(remaining / 60000); // minutes
  }, [selectedRun]);

  // Latest metrics
  const latestMetrics = metrics?.[metrics.length - 1];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Training</h1>
          <p className="text-muted-foreground">
            Monitor DPO and SFT training progress
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select
            value={selectedRunId || selectedRun?.id || ''}
            onValueChange={setSelectedRunId}
          >
            <SelectTrigger className="w-[250px]">
              <SelectValue placeholder="Select training run" />
            </SelectTrigger>
            <SelectContent>
              {runs?.map((run) => (
                <SelectItem key={run.id} value={run.id}>
                  {run.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Run Status Cards */}
      {selectedRun && (
        <div className="grid gap-4 md:grid-cols-5">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Status</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <Badge
                variant={
                  selectedRun.status === 'running'
                    ? 'default'
                    : selectedRun.status === 'completed'
                      ? 'success'
                      : selectedRun.status === 'failed'
                        ? 'destructive'
                        : 'secondary'
                }
              >
                {selectedRun.status}
              </Badge>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Progress</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {((selectedRun.currentStep / selectedRun.totalSteps) * 100).toFixed(0)}%
              </div>
              <Progress
                value={(selectedRun.currentStep / selectedRun.totalSteps) * 100}
                className="mt-2"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Current Loss</CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold font-mono">
                {latestMetrics?.loss.toFixed(4) || '—'}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Step</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold font-mono">
                {selectedRun.currentStep} / {selectedRun.totalSteps}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">ETA</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {eta ? `${eta}m` : selectedRun.status === 'completed' ? 'Done' : '—'}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Charts */}
      <Tabs defaultValue="loss">
        <TabsList>
          <TabsTrigger value="loss">Loss Curve</TabsTrigger>
          {selectedRun?.type === 'dpo' && (
            <TabsTrigger value="rewards">Reward Margins</TabsTrigger>
          )}
          <TabsTrigger value="gradients">Gradient Norms</TabsTrigger>
          <TabsTrigger value="config">Configuration</TabsTrigger>
        </TabsList>

        <TabsContent value="loss">
          <Card>
            <CardHeader>
              <CardTitle>Training Loss</CardTitle>
              <CardDescription>
                {selectedRun?.type === 'dpo'
                  ? 'DPO loss: -log σ(β × (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))'
                  : 'Cross-entropy loss over training steps'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {metricsLoading ? (
                <Skeleton className="h-[200px] w-full" />
              ) : metrics ? (
                <LossChart metrics={metrics} />
              ) : (
                <div className="h-[200px] flex items-center justify-center text-muted-foreground">
                  No metrics available
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="rewards">
          <Card>
            <CardHeader>
              <CardTitle>Reward Margin</CardTitle>
              <CardDescription>
                Difference between chosen and rejected response rewards. Should increase during
                successful DPO training.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {metricsLoading ? (
                <Skeleton className="h-[200px] w-full" />
              ) : metrics ? (
                <RewardMarginChart metrics={metrics} />
              ) : null}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="gradients">
          <Card>
            <CardHeader>
              <CardTitle>Gradient Norms by Layer</CardTitle>
              <CardDescription>
                Distribution of gradient norms across model layers. Watch for vanishing or exploding
                gradients.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {gradLoading ? (
                <Skeleton className="h-[200px] w-full" />
              ) : gradStats ? (
                <GradientStatsViz stats={gradStats} />
              ) : (
                <div className="h-[200px] flex items-center justify-center text-muted-foreground">
                  Gradient stats only available for running jobs
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="config">
          <Card>
            <CardHeader>
              <CardTitle>Training Configuration</CardTitle>
              <CardDescription>Hyperparameters for this training run</CardDescription>
            </CardHeader>
            <CardContent>
              {selectedRun && (
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <div className="text-sm text-muted-foreground">Learning Rate</div>
                    <div className="font-mono text-lg">
                      {selectedRun.config.learningRate.toExponential(1)}
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm text-muted-foreground">Batch Size</div>
                    <div className="font-mono text-lg">{selectedRun.config.batchSize}</div>
                  </div>
                  <div className="space-y-2">
                    <div className="text-sm text-muted-foreground">Epochs</div>
                    <div className="font-mono text-lg">{selectedRun.config.epochs}</div>
                  </div>
                  {selectedRun.config.beta && (
                    <div className="space-y-2">
                      <div className="text-sm text-muted-foreground">DPO Beta (β)</div>
                      <div className="font-mono text-lg">{selectedRun.config.beta}</div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Anomaly alerts */}
      {latestMetrics && latestMetrics.gradNorm > 1.0 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Gradient Explosion Detected</AlertTitle>
          <AlertDescription>
            Gradient norm ({latestMetrics.gradNorm.toFixed(2)}) exceeds threshold. Consider reducing
            learning rate or adding gradient clipping.
          </AlertDescription>
        </Alert>
      )}

      {/* Real-time updates indicator */}
      {realtimeProgress && (
        <div className="fixed bottom-4 right-4 bg-background border rounded-lg shadow-lg p-3">
          <div className="flex items-center gap-2 text-sm">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span>Live: Step {realtimeProgress.step}</span>
            <span className="text-muted-foreground">|</span>
            <span>Loss: {realtimeProgress.loss.toFixed(4)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
