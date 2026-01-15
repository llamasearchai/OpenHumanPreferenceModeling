/**
 * Quality Control Dashboard
 *
 * Purpose: Monitor annotator quality, detect spam,
 * and manage annotator status.
 */

import * as React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Users,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  TrendingUp,
  TrendingDown,
  Eye,
  Ban,
  RefreshCw,
  Search,
  Filter,
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
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { PieChart } from '@/components/widgets/PieChart';
import { BarChart } from '@/components/widgets/BarChart';

// Types
interface AnnotatorMetrics {
  id: string;
  name: string;
  email: string;
  totalAnnotations: number;
  agreementScore: number;
  goldPassRate: number;
  avgTimePerTask: number;
  status: 'active' | 'probation' | 'suspended';
  flags: string[];
  trend: 'up' | 'down' | 'stable';
  lastActive: string;
}

interface SpamFlag {
  id: string;
  annotatorId: string;
  annotatorName: string;
  flagType: 'random_responding' | 'too_fast' | 'repetitive' | 'low_agreement';
  description: string;
  evidence: string[];
  createdAt: string;
  reviewed: boolean;
}

interface AgreementPair {
  annotator1: string;
  annotator2: string;
  kappa: number;
  taskCount: number;
}

const baseTime = new Date('2026-01-10T12:00:00.000Z');

function minutesAgo(minutes: number) {
  return new Date(baseTime.getTime() - minutes * 60 * 1000).toISOString();
}

const mockAnnotators: AnnotatorMetrics[] = Array.from({ length: 15 }, (_, i) => {
  const idx = i + 1;
  const status: AnnotatorMetrics['status'] =
    idx > 13 ? 'suspended' : idx > 11 ? 'probation' : 'active';
  const trend: AnnotatorMetrics['trend'] = idx % 3 === 0 ? 'down' : idx % 2 === 0 ? 'up' : 'stable';
  const flags = status === 'suspended' ? ['low_agreement', 'too_fast'] : status === 'probation' ? ['low_agreement'] : [];

  return {
    id: `annotator-${idx}`,
    name: `Annotator ${idx}`,
    email: `annotator${idx}@ohpm.local`,
    totalAnnotations: 180 + idx * 24,
    agreementScore: Number((0.68 + idx * 0.015).toFixed(3)),
    goldPassRate: Number((0.78 + idx * 0.01).toFixed(3)),
    avgTimePerTask: Number((18 + idx * 0.9).toFixed(1)),
    status,
    flags,
    trend,
    lastActive: minutesAgo(idx * 37),
  };
}).sort((a, b) => b.agreementScore - a.agreementScore);

const mockSpamFlags: SpamFlag[] = [
  {
    id: 'flag-1',
    annotatorId: 'annotator-13',
    annotatorName: 'Annotator 13',
    flagType: 'low_agreement',
    description: 'Agreement dropped below threshold for three consecutive shifts.',
    evidence: [
      'Agreement with gold: 52%',
      'Two consecutive audit failures',
      'High variance in pairwise ratings',
    ],
    createdAt: minutesAgo(210),
    reviewed: false,
  },
  {
    id: 'flag-2',
    annotatorId: 'annotator-14',
    annotatorName: 'Annotator 14',
    flagType: 'too_fast',
    description: 'Median task completion time is far below baseline.',
    evidence: ['Median time: 3.2s', 'Baseline: 22.5s', 'Consistency score: 0.41'],
    createdAt: minutesAgo(430),
    reviewed: true,
  },
  {
    id: 'flag-3',
    annotatorId: 'annotator-12',
    annotatorName: 'Annotator 12',
    flagType: 'repetitive',
    description: 'Repeatedly selects the same option across diverse tasks.',
    evidence: ['92% identical choices', 'Response entropy: 0.08', 'Repeated rationale phrases'],
    createdAt: minutesAgo(95),
    reviewed: false,
  },
];

const mockAgreementPairs: AgreementPair[] = Array.from({ length: 10 }, (_, i) => {
  const pairs: AgreementPair[] = [];
  for (let j = i + 1; j < 10; j++) {
    const kappa = 0.35 + ((i + j) % 6) * 0.07;
    pairs.push({
      annotator1: `Annotator ${i + 1}`,
      annotator2: `Annotator ${j + 1}`,
      kappa: Number(kappa.toFixed(2)),
      taskCount: 20 + ((i + j) % 5) * 8,
    });
  }
  return pairs;
}).flat();

async function fetchAnnotators(): Promise<AnnotatorMetrics[]> {
  return mockAnnotators;
}

async function fetchSpamFlags(): Promise<SpamFlag[]> {
  return mockSpamFlags;
}

async function fetchAgreementMatrix(): Promise<AgreementPair[]> {
  return mockAgreementPairs;
}

// Flag type display
function getFlagInfo(type: SpamFlag['flagType']) {
  switch (type) {
    case 'random_responding':
      return { label: 'Random Responding', color: 'destructive' as const };
    case 'too_fast':
      return { label: 'Too Fast', color: 'warning' as const };
    case 'repetitive':
      return { label: 'Repetitive Pattern', color: 'warning' as const };
    case 'low_agreement':
      return { label: 'Low Agreement', color: 'secondary' as const };
    default:
      return { label: type, color: 'default' as const };
  }
}

// Agreement matrix heatmap
function AgreementHeatmap({ pairs }: { pairs: AgreementPair[] }) {
  // Get unique annotators
  const annotators = Array.from(
    new Set(pairs.flatMap((p) => [p.annotator1, p.annotator2]))
  ).sort();

  // Build matrix
  const matrix: Record<string, Record<string, number>> = {};
  annotators.forEach((a1) => {
    matrix[a1] = {};
    annotators.forEach((a2) => {
      if (a1 === a2) {
        if (matrix[a1]) matrix[a1]![a2] = 1;
      } else {
        const pair = pairs.find(
          (p) =>
            (p.annotator1 === a1 && p.annotator2 === a2) ||
            (p.annotator1 === a2 && p.annotator2 === a1)
        );
        if (matrix[a1]) matrix[a1]![a2] = pair?.kappa || 0;
      }
    });
  });

  const getColor = (kappa: number) => {
    if (kappa >= 0.8) return 'bg-green-500';
    if (kappa >= 0.6) return 'bg-green-400';
    if (kappa >= 0.4) return 'bg-yellow-400';
    if (kappa >= 0.2) return 'bg-orange-400';
    return 'bg-red-400';
  };

  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="p-1" />
            {annotators.slice(0, 8).map((a) => (
              <th key={a} className="p-1 font-normal text-muted-foreground truncate max-w-[60px]">
                {a.replace('Annotator ', 'A')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {annotators.slice(0, 8).map((a1) => (
            <tr key={a1}>
              <td className="p-1 font-normal text-muted-foreground truncate max-w-[60px]">
                {a1.replace('Annotator ', 'A')}
              </td>
              {annotators.slice(0, 8).map((a2) => (
                <td key={a2} className="p-1">
                  <div
                    className={`w-8 h-8 rounded flex items-center justify-center text-white font-mono ${
                      a1 === a2 ? 'bg-muted' : getColor(matrix[a1]?.[a2] ?? 0)
                    }`}
                    title={`${a1} vs ${a2}: κ=${(matrix[a1]?.[a2] ?? 0).toFixed(2)}`}
                  >
                    {a1 !== a2 && (matrix[a1]?.[a2] ?? 0).toFixed(1)}
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center gap-4 mt-4 text-xs">
        <span className="text-muted-foreground">Cohen's Kappa:</span>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-red-400" />
          <span>&lt;0.2</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-orange-400" />
          <span>0.2-0.4</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-yellow-400" />
          <span>0.4-0.6</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-green-400" />
          <span>0.6-0.8</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 rounded bg-green-500" />
          <span>&gt;0.8</span>
        </div>
      </div>
    </div>
  );
}

// Review dialog
function ReviewDialog({
  flag,
  open,
  onClose,
  onConfirm,
  onDismiss,
}: {
  flag: SpamFlag | null;
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  onDismiss: () => void;
}) {
  if (!flag) return null;

  const info = getFlagInfo(flag.flagType);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Review Spam Flag</DialogTitle>
          <DialogDescription>
            Review the evidence and take action on this flag.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">{flag.annotatorName}</span>
            <Badge variant={info.color}>{info.label}</Badge>
          </div>

          <div>
            <div className="text-sm font-medium mb-2">Evidence</div>
            <ul className="text-sm text-muted-foreground space-y-1">
              {flag.evidence.map((e, i) => (
                <li key={i} className="flex items-center gap-2">
                  <span className="w-1 h-1 rounded-full bg-muted-foreground" />
                  {e}
                </li>
              ))}
            </ul>
          </div>

          <div className="text-xs text-muted-foreground">
            Flagged {new Date(flag.createdAt).toLocaleString()}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onDismiss}>
            Dismiss Flag
          </Button>
          <Button variant="destructive" onClick={onConfirm}>
            <Ban className="mr-2 h-4 w-4" />
            Suspend Annotator
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// Main component
export default function QualityControlPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [searchQuery, setSearchQuery] = React.useState('');
  const [statusFilter, setStatusFilter] = React.useState<string>('all');
  const [selectedFlag, setSelectedFlag] = React.useState<SpamFlag | null>(null);

  // Queries
  const { data: annotators, isLoading: annotatorsLoading } = useQuery({
    queryKey: ['quality', 'annotators'],
    queryFn: fetchAnnotators,
    refetchInterval: 30000,
  });

  const { data: flags, isLoading: flagsLoading } = useQuery({
    queryKey: ['quality', 'spam-flags'],
    queryFn: fetchSpamFlags,
    refetchInterval: 30000,
  });

  const { data: agreementPairs, isLoading: agreementLoading } = useQuery({
    queryKey: ['quality', 'agreement-matrix'],
    queryFn: fetchAgreementMatrix,
  });

  // Mutations
  const updateStatusMutation = useMutation({
    mutationFn: async ({
      annotatorId,
      status,
    }: {
      annotatorId: string;
      status: string;
    }) => {
      await new Promise((r) => setTimeout(r, 500));
      return { annotatorId, status };
    },
    onSuccess: (_, { status }) => {
      queryClient.invalidateQueries({ queryKey: ['quality'] });
      toast({
        title: `Annotator ${status}`,
        variant: status === 'suspended' ? 'destructive' : 'success',
      });
    },
  });

  const dismissFlagMutation = useMutation({
    mutationFn: async (flagId: string) => {
      await new Promise((r) => setTimeout(r, 500));
      return flagId;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['quality', 'spam-flags'] });
      toast({ title: 'Flag dismissed', variant: 'default' });
      setSelectedFlag(null);
    },
  });

  // Filtered annotators
  const filteredAnnotators = React.useMemo(() => {
    if (!annotators) return [];
    return annotators.filter((a) => {
      const matchesSearch =
        a.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        a.email.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus = statusFilter === 'all' || a.status === statusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [annotators, searchQuery, statusFilter]);

  const unreviewedFlags = flags?.filter((f) => !f.reviewed) || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Quality Control</h1>
          <p className="text-muted-foreground">
            Monitor annotator performance and detect quality issues
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => queryClient.invalidateQueries({ queryKey: ['quality'] })}
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Refresh
        </Button>
      </div>

      <Alert variant="info">
        <AlertTitle>Mock quality data</AlertTitle>
        <AlertDescription>
          Quality metrics are currently mocked to validate dashboards and charts.
        </AlertDescription>
      </Alert>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Annotators</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {annotatorsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">{annotators?.length || 0}</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Annotators</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {annotatorsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">
                {annotators?.filter((a) => a.status === 'active').length}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Agreement</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {annotatorsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">
                {(
                  (annotators?.reduce((s, a) => s + a.agreementScore, 0) || 0) /
                  (annotators?.length || 1) *
                  100
                ).toFixed(0)}
                %
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Flags</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {flagsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">{unreviewedFlags.length}</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">On Probation</CardTitle>
            <XCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {annotatorsLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <div className="text-2xl font-bold">
                {annotators?.filter((a) => a.status === 'probation').length}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="leaderboard">
        <TabsList>
          <TabsTrigger value="leaderboard">Annotator Leaderboard</TabsTrigger>
          <TabsTrigger value="flags">
            Spam Flags
            {unreviewedFlags.length > 0 && (
              <Badge variant="destructive" className="ml-2">
                {unreviewedFlags.length}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="agreement">Agreement Matrix</TabsTrigger>
        </TabsList>

        {/* Leaderboard Tab */}
        <TabsContent value="leaderboard">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Annotator Performance</CardTitle>
                  <CardDescription>Ranked by agreement score</CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      aria-label="Search annotators"
                      className="pl-9 w-[200px]"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  </div>
                  <Select value={statusFilter} onValueChange={setStatusFilter}>
                    <SelectTrigger className="w-[130px]" aria-label="Status filter">
                      <Filter className="mr-2 h-4 w-4" />
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Status</SelectItem>
                      <SelectItem value="active">Active</SelectItem>
                      <SelectItem value="probation">Probation</SelectItem>
                      <SelectItem value="suspended">Suspended</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              {annotatorsLoading ? (
                <div className="space-y-2">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Rank</TableHead>
                      <TableHead>Annotator</TableHead>
                      <TableHead>Tasks</TableHead>
                      <TableHead>Agreement</TableHead>
                      <TableHead>Gold Pass</TableHead>
                      <TableHead>Avg Time</TableHead>
                      <TableHead>Trend</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredAnnotators.map((annotator, index) => (
                      <TableRow key={annotator.id}>
                        <TableCell className="font-medium">#{index + 1}</TableCell>
                        <TableCell>
                          <div>
                            <div className="font-medium">{annotator.name}</div>
                            <div className="text-xs text-muted-foreground">
                              {annotator.email}
                            </div>
                          </div>
                        </TableCell>
                        <TableCell className="font-mono">
                          {annotator.totalAnnotations}
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <Progress
                              value={annotator.agreementScore * 100}
                              className="w-16"
                            />
                            <span className="font-mono text-sm">
                              {(annotator.agreementScore * 100).toFixed(0)}%
                            </span>
                          </div>
                        </TableCell>
                        <TableCell className="font-mono">
                          {(annotator.goldPassRate * 100).toFixed(0)}%
                        </TableCell>
                        <TableCell className="font-mono">
                          {annotator.avgTimePerTask.toFixed(0)}s
                        </TableCell>
                        <TableCell>
                          {annotator.trend === 'up' ? (
                            <TrendingUp className="h-4 w-4 text-green-500" />
                          ) : annotator.trend === 'down' ? (
                            <TrendingDown className="h-4 w-4 text-red-500" />
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              annotator.status === 'active'
                                ? 'success'
                                : annotator.status === 'probation'
                                  ? 'warning'
                                  : 'destructive'
                            }
                          >
                            {annotator.status}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Button variant="ghost" size="sm">
                            <Eye className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Spam Flags Tab */}
        <TabsContent value="flags">
          <Card>
            <CardHeader>
              <CardTitle>Spam Detection Alerts</CardTitle>
              <CardDescription>
                Review and take action on suspicious annotation patterns
              </CardDescription>
            </CardHeader>
            <CardContent>
              {flagsLoading ? (
                <div className="space-y-2">
                  {Array.from({ length: 3 }).map((_, i) => (
                    <Skeleton key={i} className="h-16 w-full" />
                  ))}
                </div>
              ) : flags && flags.length > 0 ? (
                <div className="space-y-3">
                  {flags.map((flag) => {
                    const info = getFlagInfo(flag.flagType);
                    return (
                      <div
                        key={flag.id}
                        className={`flex items-center justify-between p-4 border rounded-lg ${
                          flag.reviewed ? 'opacity-60' : ''
                        }`}
                      >
                        <div className="flex items-center gap-4">
                          <AlertTriangle
                            className={`h-5 w-5 ${
                              flag.reviewed
                                ? 'text-muted-foreground'
                                : 'text-destructive'
                            }`}
                          />
                          <div>
                            <div className="font-medium">{flag.annotatorName}</div>
                            <div className="text-sm text-muted-foreground">
                              {flag.description}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <Badge variant={info.color}>{info.label}</Badge>
                          {flag.reviewed ? (
                            <Badge variant="outline">Reviewed</Badge>
                          ) : (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => setSelectedFlag(flag)}
                            >
                              Review
                            </Button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-8">
                  <CheckCircle2 className="h-12 w-12 mx-auto text-green-500 mb-4" />
                  <p className="text-lg font-medium">No Spam Detected</p>
                  <p className="text-sm text-muted-foreground">
                    All annotations appear legitimate
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Agreement Matrix Tab */}
        <TabsContent value="agreement" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Inter-Annotator Agreement</CardTitle>
              <CardDescription>
                Pairwise Cohen's Kappa scores between annotators
              </CardDescription>
            </CardHeader>
            <CardContent>
              {agreementLoading ? (
                <Skeleton className="h-64 w-full" />
              ) : agreementPairs ? (
                <AgreementHeatmap pairs={agreementPairs} />
              ) : null}
            </CardContent>
          </Card>

          {/* Status Distribution */}
          {annotators && (
            <div className="grid gap-4 md:grid-cols-2">
              <PieChart
                title="Annotator Status Distribution"
                description="Breakdown of annotators by status"
                data={[
                  {
                    name: 'Active',
                    value: annotators.filter((a) => a.status === 'active').length,
                  },
                  {
                    name: 'Probation',
                    value: annotators.filter((a) => a.status === 'probation').length,
                  },
                  {
                    name: 'Suspended',
                    value: annotators.filter((a) => a.status === 'suspended').length,
                  },
                ]}
                height={300}
              />

              <BarChart
                title="Agreement Score Distribution"
                description="Distribution of agreement scores across annotators"
                data={annotators.slice(0, 10).map((a) => ({
                  name: a.name.replace('Annotator ', 'A'),
                  value: a.agreementScore * 100,
                }))}
                height={300}
                formatYAxis={(value) => `${value.toFixed(0)}%`}
                formatTooltip={(value) => `${value.toFixed(1)}%`}
              />
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Review Dialog */}
      <ReviewDialog
        flag={selectedFlag}
        open={!!selectedFlag}
        onClose={() => setSelectedFlag(null)}
        onConfirm={() => {
          if (selectedFlag) {
            updateStatusMutation.mutate({
              annotatorId: selectedFlag.annotatorId,
              status: 'suspended',
            });
            setSelectedFlag(null);
          }
        }}
        onDismiss={() => {
          if (selectedFlag) {
            dismissFlagMutation.mutate(selectedFlag.id);
          }
        }}
      />
    </div>
  );
}
