/* eslint-disable no-undef */
/**
 * Active Learning Dashboard Page
 *
 * Purpose: Manage and monitor active learning strategies,
 * view priority queues, and compare strategy effectiveness.
 */

import * as React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Brain,
  Settings,
  RefreshCw,
  TrendingUp,
  Target,
  Layers,
  Info,
  ChevronRight,
  BarChart3,
  Shuffle,
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
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { useToast } from '@/hooks/use-toast';
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
interface QueueItem {
  id: string;
  text: string;
  uncertaintyScore: number;
  diversityScore: number;
  iidScore: number;
  compositeRank: number;
  createdAt: string;
}

interface StrategyMetrics {
  strategy: string;
  accuracyGain: number[];
  samplesUsed: number[];
  iterations: number;
}

interface ActiveLearningConfig {
  budget: number;
  batchSize: number;
  strategy: string;
  seedSize: number;
}

interface ActiveLearningStatus {
  labeledCount: number;
  unlabeledCount: number;
  budgetRemaining: number;
  currentStrategy: string;
  lastUpdated: string;
}

// Mock API functions
async function fetchQueue(): Promise<QueueItem[]> {
  // In production: return fetch('/api/active-learning/queue').then(r => r.json());
  return Array.from({ length: 20 }, (_, i) => ({
    id: `sample-${i}`,
    text: `Sample text ${i}: This is a candidate for annotation...`,
    uncertaintyScore: Math.random(),
    diversityScore: Math.random(),
    iidScore: Math.random(),
    compositeRank: i + 1,
    createdAt: new Date(Date.now() - Math.random() * 86400000).toISOString(),
  }));
}

async function fetchStrategyComparison(): Promise<StrategyMetrics[]> {
  const iterations = 10;
  return [
    {
      strategy: 'Uncertainty',
      accuracyGain: Array.from({ length: iterations }, (_, i) =>
        0.5 + 0.4 * (1 - Math.exp(-i / 3))
      ),
      samplesUsed: Array.from({ length: iterations }, (_, i) => (i + 1) * 10),
      iterations,
    },
    {
      strategy: 'Diversity',
      accuracyGain: Array.from({ length: iterations }, (_, i) =>
        0.5 + 0.35 * (1 - Math.exp(-i / 4))
      ),
      samplesUsed: Array.from({ length: iterations }, (_, i) => (i + 1) * 10),
      iterations,
    },
    {
      strategy: 'IID',
      accuracyGain: Array.from({ length: iterations }, (_, i) =>
        0.5 + 0.42 * (1 - Math.exp(-i / 2.5))
      ),
      samplesUsed: Array.from({ length: iterations }, (_, i) => (i + 1) * 10),
      iterations,
    },
  ];
}

async function fetchStatus(): Promise<ActiveLearningStatus> {
  return {
    labeledCount: 150,
    unlabeledCount: 850,
    budgetRemaining: 50,
    currentStrategy: 'iid',
    lastUpdated: new Date().toISOString(),
  };
}

async function fetchConfig(): Promise<ActiveLearningConfig> {
  return {
    budget: 200,
    batchSize: 10,
    strategy: 'iid',
    seedSize: 20,
  };
}

// Strategy explanations
const strategyInfo: Record<string, { name: string; description: string; formula: string }> = {
  uncertainty: {
    name: 'Uncertainty Sampling',
    description:
      'Selects samples where the model is most uncertain about the prediction. Best for exploiting areas where the model needs improvement.',
    formula: 'score = 1 - max(P(y|x))',
  },
  diversity: {
    name: 'Diversity Sampling',
    description:
      'Selects samples that are most different from the current labeled set. Uses k-means++ initialization to ensure coverage.',
    formula: 'score = min_j ||x_i - c_j||²',
  },
  iid: {
    name: 'Inverse Information Density',
    description:
      'Balances uncertainty with representativeness. Prioritizes uncertain samples that are also representative of the unlabeled pool.',
    formula: 'score = uncertainty × (1 / density)',
  },
};

// Queue item component
function QueueItemRow({
  item,
  onAnnotate,
}: {
  item: QueueItem;
  onAnnotate: (id: string) => void;
}) {
  return (
    <div className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 transition-colors">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <Badge variant="outline" className="font-mono text-xs">
            #{item.compositeRank}
          </Badge>
          <span className="text-sm font-medium truncate">{item.text}</span>
        </div>
        <div className="flex gap-4 text-xs text-muted-foreground">
          <span>
            Uncertainty:{' '}
            <span className="font-mono">{(item.uncertaintyScore * 100).toFixed(1)}%</span>
          </span>
          <span>
            Diversity:{' '}
            <span className="font-mono">{(item.diversityScore * 100).toFixed(1)}%</span>
          </span>
          <span>
            IID: <span className="font-mono">{(item.iidScore * 100).toFixed(1)}%</span>
          </span>
        </div>
      </div>
      <Button size="sm" variant="outline" onClick={() => onAnnotate(item.id)}>
        Annotate
        <ChevronRight className="ml-1 h-4 w-4" />
      </Button>
    </div>
  );
}

// Strategy comparison chart with Recharts
function StrategyChart({ data }: { data: StrategyMetrics[] }) {
  // Prepare data for Recharts
  const chartData = data[0]?.samplesUsed.map((samples, index) => {
    const point: Record<string, number | string> = {
      samples,
    };
    data.forEach((strategy) => {
      point[strategy.strategy] = strategy.accuracyGain[index] ?? 0;
    });
    return point;
  }) || [];

  const colors = {
    Uncertainty: 'hsl(var(--primary))',
    Diversity: '#22c55e',
    IID: '#3b82f6',
  };

  return (
    <div className="h-[400px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <RechartsLineChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="samples"
            label={{ value: 'Samples Used', position: 'insideBottom', offset: -5 }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            label={{ value: 'Accuracy Gain', angle: -90, position: 'insideLeft' }}
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--popover))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
            formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Accuracy']}
            labelFormatter={(label) => `Samples: ${label}`}
          />
          <Legend />
          {data.map((strategy) => (
            <Line
              key={strategy.strategy}
              type="monotone"
              dataKey={strategy.strategy}
              stroke={colors[strategy.strategy as keyof typeof colors] || 'hsl(var(--primary))'}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          ))}
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  );
}

// Configuration dialog
function ConfigDialog({
  config,
  onSave,
}: {
  config: ActiveLearningConfig;
  onSave: (config: ActiveLearningConfig) => void;
}) {
  const [localConfig, setLocalConfig] = React.useState(config);

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Settings className="mr-2 h-4 w-4" />
          Configure
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Active Learning Configuration</DialogTitle>
          <DialogDescription>
            Adjust the active learning parameters for your annotation workflow.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Strategy</label>
            <Select
              value={localConfig.strategy}
              onValueChange={(v) => setLocalConfig((c) => ({ ...c, strategy: v }))}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="uncertainty">Uncertainty Sampling</SelectItem>
                <SelectItem value="diversity">Diversity Sampling</SelectItem>
                <SelectItem value="iid">Inverse Information Density</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Batch Size</label>
              <Badge variant="outline">{localConfig.batchSize}</Badge>
            </div>
            <Slider
              value={[localConfig.batchSize]}
              onValueChange={([v]) => setLocalConfig((c) => ({ ...c, batchSize: v ?? c.batchSize }))}
              min={1}
              max={50}
              step={1}
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Budget</label>
              <Badge variant="outline">{localConfig.budget}</Badge>
            </div>
            <Slider
              value={[localConfig.budget]}
              onValueChange={([v]) => setLocalConfig((c) => ({ ...c, budget: v ?? c.budget }))}
              min={10}
              max={1000}
              step={10}
            />
          </div>

          <Button className="w-full" onClick={() => onSave(localConfig)}>
            Save Configuration
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// Main component
export default function ActiveLearningPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [selectedStrategy, setSelectedStrategy] = React.useState<string>('iid');

  // Queries
  const { data: queue, isLoading: queueLoading } = useQuery({
    queryKey: ['active-learning', 'queue'],
    queryFn: fetchQueue,
    refetchInterval: 30000,
  });

  const { data: comparison, isLoading: comparisonLoading } = useQuery({
    queryKey: ['active-learning', 'comparison'],
    queryFn: fetchStrategyComparison,
  });

  const { data: status, isLoading: statusLoading } = useQuery({
    queryKey: ['active-learning', 'status'],
    queryFn: fetchStatus,
    refetchInterval: 10000,
  });

  const { data: config } = useQuery({
    queryKey: ['active-learning', 'config'],
    queryFn: fetchConfig,
  });

  // Mutations
  const updateConfigMutation = useMutation({
    mutationFn: async (newConfig: ActiveLearningConfig) => {
      // In production: return fetch('/api/active-learning/config', { method: 'PATCH', body: JSON.stringify(newConfig) });
      return newConfig;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['active-learning'] });
      toast({ title: 'Configuration updated', variant: 'success' });
    },
  });

  const refreshPredictionsMutation = useMutation({
    mutationFn: async () => {
      // In production: return fetch('/api/active-learning/refresh', { method: 'POST' });
      await new Promise((r) => window.setTimeout(r, 1000));
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['active-learning', 'queue'] });
      toast({ title: 'Predictions refreshed', variant: 'success' });
    },
  });

  const handleAnnotate = (id: string) => {
    // Navigate to annotation with pre-selected task
    window.location.href = `/annotations?taskId=${id}`;
  };

  const budgetProgress = status
    ? ((status.budgetRemaining / (config?.budget || 200)) * 100)
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Active Learning</h1>
          <p className="text-muted-foreground">
            Optimize annotation efficiency with intelligent sampling
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refreshPredictionsMutation.mutate()}
            disabled={refreshPredictionsMutation.isPending}
          >
            <RefreshCw
              className={`mr-2 h-4 w-4 ${refreshPredictionsMutation.isPending ? 'animate-spin' : ''}`}
            />
            Refresh Predictions
          </Button>
          {config && (
            <ConfigDialog config={config} onSave={(c) => updateConfigMutation.mutate(c)} />
          )}
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Labeled</CardTitle>
            <Layers className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <>
                <div className="text-2xl font-bold">{status?.labeledCount}</div>
                <p className="text-xs text-muted-foreground">samples annotated</p>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Unlabeled</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <>
                <div className="text-2xl font-bold">{status?.unlabeledCount}</div>
                <p className="text-xs text-muted-foreground">samples remaining</p>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Budget</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <>
                <div className="text-2xl font-bold">{status?.budgetRemaining}</div>
                <Progress value={budgetProgress} className="mt-2" />
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Strategy</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statusLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <>
                <div className="text-2xl font-bold capitalize">
                  {status?.currentStrategy || 'IID'}
                </div>
                <p className="text-xs text-muted-foreground">active strategy</p>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="queue" className="space-y-4">
        <TabsList>
          <TabsTrigger value="queue">Priority Queue</TabsTrigger>
          <TabsTrigger value="comparison">Strategy Comparison</TabsTrigger>
          <TabsTrigger value="explanation">How It Works</TabsTrigger>
        </TabsList>

        {/* Priority Queue Tab */}
        <TabsContent value="queue" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Annotation Queue</CardTitle>
                  <CardDescription>
                    Samples ranked by the selected active learning strategy
                  </CardDescription>
                </div>
                <Select value={selectedStrategy} onValueChange={setSelectedStrategy}>
                  <SelectTrigger className="w-[200px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="uncertainty">Sort by Uncertainty</SelectItem>
                    <SelectItem value="diversity">Sort by Diversity</SelectItem>
                    <SelectItem value="iid">Sort by IID Score</SelectItem>
                    <SelectItem value="composite">Composite Ranking</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              {queueLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <Skeleton key={i} className="h-20 w-full" />
                  ))}
                </div>
              ) : queue && queue.length > 0 ? (
                <div className="space-y-3">
                  {queue.slice(0, 10).map((item) => (
                    <QueueItemRow key={item.id} item={item} onAnnotate={handleAnnotate} />
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Shuffle className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                  <p className="text-lg font-medium">Queue Empty</p>
                  <p className="text-sm text-muted-foreground">
                    All samples have been labeled or budget is exhausted
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Strategy Comparison Tab */}
        <TabsContent value="comparison" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Strategy Performance
              </CardTitle>
              <CardDescription>
                Compare the effectiveness of different sampling strategies
              </CardDescription>
            </CardHeader>
            <CardContent>
              {comparisonLoading ? (
                <div className="space-y-4">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : comparison ? (
                <StrategyChart data={comparison} />
              ) : null}
            </CardContent>
          </Card>

          <Alert>
            <Info className="h-4 w-4" />
            <AlertTitle>Interpretation</AlertTitle>
            <AlertDescription>
              Higher gain/sample indicates more efficient learning. IID typically performs
              best by balancing exploration (diversity) and exploitation (uncertainty).
            </AlertDescription>
          </Alert>
        </TabsContent>

        {/* Explanation Tab */}
        <TabsContent value="explanation" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            {Object.entries(strategyInfo).map(([key, info]) => (
              <Card key={key}>
                <CardHeader>
                  <CardTitle className="text-lg">{info.name}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">{info.description}</p>
                  <div className="bg-muted rounded-md p-3">
                    <code className="text-xs font-mono">{info.formula}</code>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Active Learning Workflow</CardTitle>
            </CardHeader>
            <CardContent>
              <ol className="space-y-4">
                <li className="flex gap-4">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-medium">
                    1
                  </div>
                  <div>
                    <p className="font-medium">Initial Seed</p>
                    <p className="text-sm text-muted-foreground">
                      Start with a small random sample to bootstrap the model
                    </p>
                  </div>
                </li>
                <li className="flex gap-4">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-medium">
                    2
                  </div>
                  <div>
                    <p className="font-medium">Train Model</p>
                    <p className="text-sm text-muted-foreground">
                      Train on current labeled data to get predictions on unlabeled samples
                    </p>
                  </div>
                </li>
                <li className="flex gap-4">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-medium">
                    3
                  </div>
                  <div>
                    <p className="font-medium">Select Samples</p>
                    <p className="text-sm text-muted-foreground">
                      Use the active learning strategy to pick the most informative samples
                    </p>
                  </div>
                </li>
                <li className="flex gap-4">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-medium">
                    4
                  </div>
                  <div>
                    <p className="font-medium">Annotate</p>
                    <p className="text-sm text-muted-foreground">
                      Human annotators label the selected samples
                    </p>
                  </div>
                </li>
                <li className="flex gap-4">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground text-sm font-medium">
                    5
                  </div>
                  <div>
                    <p className="font-medium">Repeat</p>
                    <p className="text-sm text-muted-foreground">
                      Continue until budget is exhausted or target accuracy is reached
                    </p>
                  </div>
                </li>
              </ol>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
