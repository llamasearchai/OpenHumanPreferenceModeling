/**
 * Annotations Page
 *
 * Purpose: Complete annotation workspace supporting all task types
 */

import * as React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { annotationApi } from '@/lib/api-client';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/hooks/use-toast';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  CheckCircle2,
  XCircle,
  Minus,
  RotateCcw,
  SkipForward,
  Clock,
  BarChart3,
  Layers,
} from 'lucide-react';
import type { PairwiseWinner, TaskContent } from '@/types/api';
import { EmbeddingSpace } from '@/components/visualizations/EmbeddingSpace';
import { AnnotationStats } from '@/components/widgets/AnnotationStats';

const confidenceLabels = ['Very Low', 'Low', 'Medium', 'High', 'Very High'];

export default function AnnotationsPage() {
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const [startTime, setStartTime] = React.useState<number>(0);
  const [confidence, setConfidence] = React.useState<number[]>([3]);
  const [rationale, setRationale] = React.useState('');
  const [selectedWinner, setSelectedWinner] = React.useState<PairwiseWinner | null>(null);
  const [ranking, setRanking] = React.useState<string[]>([]);
  const [likertScore, setLikertScore] = React.useState<number[]>([4]);

  const annotatorId = user?.id || 'demo_user';
  const confidenceValue = confidence[0] ?? 3;

  const {
    data: taskData,
    isLoading: taskLoading,
    error: taskError,
    refetch: refetchTask,
  } = useQuery({
    queryKey: ['tasks', 'next', annotatorId],
    queryFn: async () => {
      const result = await annotationApi.getNextTask(annotatorId);
      if (!result.success) throw new Error(result.error.detail);
      return result.data;
    },
    retry: false,
  });

  React.useEffect(() => {
    if (taskData) {
      setStartTime(Date.now());
      setSelectedWinner(null);
      setConfidence([3]);
      setRationale('');
      setLikertScore([4]);

      // Initialize ranking order if ranking task
      if (taskData.type === 'ranking' && 'responses' in taskData.content) {
        const responses = (taskData.content as { responses: string[] }).responses;
        setRanking(responses.map((_r: string, i: number) => `${i}`));
      }
    }
  }, [taskData]);

  const submitMutation = useMutation({
    mutationFn: async (responseData: Record<string, unknown>) => {
      const timeSpent = (Date.now() - startTime) / 1000;
      const result = await annotationApi.submitAnnotation({
        task_id: taskData!.id,
        annotator_id: annotatorId,
        annotation_type: taskData!.type,
        response_data: responseData,
        time_spent_seconds: timeSpent,
        confidence: confidenceValue,
      });
      if (!result.success) throw new Error(result.error.detail);
      return result.data;
    },
    onSuccess: () => {
      toast({
        title: 'Annotation submitted',
        description: 'Loading next task...',
        variant: 'success',
      });
      queryClient.invalidateQueries({ queryKey: ['tasks', 'next'] });
    },
    onError: (error) => {
      toast({
        title: 'Failed to submit',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    },
  });

  const handlePairwiseSubmit = (winner: PairwiseWinner) => {
    setSelectedWinner(winner);
  };

  const handleSubmit = () => {
    if (!taskData) return;

    let responseData: Record<string, unknown>;

    switch (taskData.type) {
      case 'pairwise':
        if (!selectedWinner) {
          toast({ title: 'Please select a winner', variant: 'destructive' });
          return;
        }
        responseData = {
          winner: selectedWinner,
          rationale: rationale || undefined,
        };
        break;

      case 'ranking':
        responseData = {
          ranking,
          rationale: rationale || undefined,
        };
        break;

      case 'likert':
        responseData = {
          rating: likertScore[0],
          rationale: rationale || undefined,
        };
        break;

      case 'critique':
        if (!rationale.trim()) {
          toast({ title: 'Please provide feedback', variant: 'destructive' });
          return;
        }
        responseData = {
          critique: rationale,
        };
        break;

      default:
        responseData = { rationale };
    }

    submitMutation.mutate(responseData);
  };

  const handleSkip = () => {
    refetchTask();
  };

  if (taskLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <Card>
          <CardHeader>
            <Skeleton className="h-6 w-48" />
          </CardHeader>
          <CardContent className="space-y-4">
            <Skeleton className="h-32 w-full" />
            <Skeleton className="h-32 w-full" />
          </CardContent>
        </Card>
      </div>
    );
  }

  if (taskError) {
    return (
      <div className="space-y-6">
        <h1 className="text-3xl font-bold tracking-tight">Annotations</h1>
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <CheckCircle2 className="h-16 w-16 text-green-500 mb-4" />
            <h2 className="text-xl font-semibold mb-2">All caught up!</h2>
            <p className="text-muted-foreground text-center max-w-md mb-4">
              No annotation tasks are available right now. Check back later or refresh to try again.
            </p>
            <Button onClick={() => refetchTask()}>
              <RotateCcw className="mr-2 h-4 w-4" />
              Check for tasks
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!taskData) return null;
  const content: TaskContent = taskData.content;
  const prompt = ((): string => {
    const maybePrompt = (content as Record<string, unknown>).prompt;
    return typeof maybePrompt === 'string' ? maybePrompt : 'Task';
  })();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Annotations</h1>
          <p className="text-muted-foreground">
            Complete annotation tasks to improve model quality
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline">
            <Clock className="mr-1 h-3 w-3" />
            Task: {taskData.type}
          </Badge>
          <Button variant="outline" size="sm" onClick={handleSkip}>
            <SkipForward className="mr-2 h-4 w-4" />
            Skip
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="task" className="space-y-4">
        <TabsList>
          <TabsTrigger value="task">Current Task</TabsTrigger>
          <TabsTrigger value="history">My Annotations</TabsTrigger>
          <TabsTrigger value="visualization">Embedding Space</TabsTrigger>
        </TabsList>

        <TabsContent value="task" className="space-y-6">
          {/* Task Content */}
      <Card>
        <CardHeader>
          <CardTitle>
            {prompt}
          </CardTitle>
          <CardDescription>
            {taskData.type === 'pairwise' && 'Compare the two responses and select the better one'}
            {taskData.type === 'ranking' && 'Rank the responses from best to worst'}
            {taskData.type === 'likert' && 'Rate the response quality on a scale'}
            {taskData.type === 'critique' && 'Provide detailed feedback on the response'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {/* Pairwise Task */}
          {taskData.type === 'pairwise' && 'response_a' in taskData.content && (
            <div className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <Card
                  className={`cursor-pointer transition-all ${
                    selectedWinner === 'A'
                      ? 'ring-2 ring-primary border-primary'
                      : 'hover:border-muted-foreground'
                  }`}
                  onClick={() => handlePairwiseSubmit('A')}
                >
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center justify-between">
                      Response A
                      {selectedWinner === 'A' && (
                        <CheckCircle2 className="h-5 w-5 text-primary" />
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="whitespace-pre-wrap">
                      {String((taskData.content as Record<string, unknown>).response_a ?? '')}
                    </p>
                  </CardContent>
                </Card>

                <Card
                  className={`cursor-pointer transition-all ${
                    selectedWinner === 'B'
                      ? 'ring-2 ring-primary border-primary'
                      : 'hover:border-muted-foreground'
                  }`}
                  onClick={() => handlePairwiseSubmit('B')}
                >
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center justify-between">
                      Response B
                      {selectedWinner === 'B' && (
                        <CheckCircle2 className="h-5 w-5 text-primary" />
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="whitespace-pre-wrap">
                      {String((taskData.content as Record<string, unknown>).response_b ?? '')}
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Additional Options */}
              <div className="flex gap-2 justify-center">
                <Button
                  variant={selectedWinner === 'tie' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => handlePairwiseSubmit('tie')}
                >
                  <Minus className="mr-2 h-4 w-4" />
                  Tie
                </Button>
                <Button
                  variant={selectedWinner === 'both_poor' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => handlePairwiseSubmit('both_poor')}
                >
                  <XCircle className="mr-2 h-4 w-4" />
                  Both Poor
                </Button>
              </div>
            </div>
          )}

          {/* Likert Task */}
          {taskData.type === 'likert' && (
            <div className="space-y-6">
              <Card>
                <CardContent className="pt-6">
                  <p className="whitespace-pre-wrap">
                    {String((taskData.content as Record<string, unknown>).response ?? '')}
                  </p>
                </CardContent>
              </Card>

              <div className="space-y-4">
                <label className="text-sm font-medium">
                  Rate the quality (1-7):
                </label>
                <Slider
                  value={likertScore}
                  onValueChange={setLikertScore}
                  min={1}
                  max={7}
                  step={1}
                  stepLabels={['1', '2', '3', '4', '5', '6', '7']}
                  showValue
                />
              </div>
            </div>
          )}

          {/* Critique Task */}
          {taskData.type === 'critique' && (
            <div className="space-y-4">
              <Card>
                <CardContent className="pt-6">
                  <p className="whitespace-pre-wrap">
                    {String((taskData.content as Record<string, unknown>).response ?? '')}
                  </p>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Ranking Task */}
          {taskData.type === 'ranking' && 'responses' in taskData.content && (
            <div className="space-y-4">
              <Alert>
                <AlertDescription>
                  Drag responses to reorder them from best (top) to worst (bottom).
                </AlertDescription>
              </Alert>
              <div className="space-y-2">
                {((taskData.content as { responses: string[] }).responses || []).map(
                  (response: string, index: number) => (
                  <Card key={index} className="cursor-move">
                    <CardHeader className="py-3">
                      <div className="flex items-center gap-3">
                        <Badge variant="outline">{index + 1}</Badge>
                        <p className="text-sm flex-1">{response}</p>
                      </div>
                    </CardHeader>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Confidence & Rationale */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Your Assessment</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Confidence Slider */}
          <div className="space-y-4">
            <label className="text-sm font-medium">
              Confidence in your answer:
            </label>
            <Slider
              value={confidence}
              onValueChange={setConfidence}
              min={1}
              max={5}
              step={1}
              stepLabels={confidenceLabels}
            />
          </div>

          {/* Rationale */}
          <Textarea
            label="Rationale (optional)"
            placeholder="Explain your reasoning..."
            value={rationale}
            onChange={(e) => setRationale(e.target.value)}
            maxLength={500}
            showCount
            helperText="Providing rationale helps improve model training"
          />

          {/* Submit Button */}
          <Button
            onClick={handleSubmit}
            className="w-full"
            size="lg"
            isLoading={submitMutation.isPending}
            disabled={
              (taskData.type === 'pairwise' && !selectedWinner) ||
              (taskData.type === 'critique' && !rationale.trim())
            }
          >
            Submit Annotation
          </Button>
        </CardContent>
      </Card>
        </TabsContent>

        <TabsContent value="history" className="space-y-4">
          <AnnotationHistory annotatorId={annotatorId} />
          <AnnotationStats annotatorId={annotatorId} />
        </TabsContent>

        <TabsContent value="visualization" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Layers className="h-5 w-5" />
                Annotation Embedding Space
              </CardTitle>
              <CardDescription>
                Explore your annotations in 3D embedding space
              </CardDescription>
            </CardHeader>
            <CardContent>
              <EmbeddingSpace
                maxPoints={1000}
                colorBy="taskType"
                onPointClick={(point) => {
                  toast({
                    title: 'Point selected',
                    description: `Task: ${point.taskId}`,
                  });
                }}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Annotation History Component
function AnnotationHistory({ annotatorId }: { annotatorId: string }) {
  const {
    data: annotationsData,
    isLoading,
  } = useQuery({
    queryKey: ['annotations', annotatorId],
    queryFn: async () => {
      const result = await annotationApi.listAnnotations({
        annotator_id: annotatorId,
        page: 1,
        page_size: 20,
      });
      if (!result.success) throw new Error(result.error.detail);
      return result.data;
    },
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Annotation History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          My Annotation History
        </CardTitle>
        <CardDescription>
          {annotationsData?.meta.total || 0} total annotations
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!annotationsData?.data.length ? (
          <div className="text-center py-8 text-muted-foreground">
            No annotations yet. Start annotating to see your history here.
          </div>
        ) : (
          <div className="space-y-3">
            {annotationsData.data.map((annotation) => (
              <div
                key={annotation.id}
                className="flex items-center justify-between p-3 border rounded-lg"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{annotation.annotation_type}</Badge>
                    <span className="text-sm text-muted-foreground">
                      Task: {annotation.task_id.slice(0, 8)}...
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {new Date(annotation.created_at).toLocaleString()}
                  </p>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium">
                    {annotation.time_spent_seconds.toFixed(1)}s
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Confidence: {annotation.confidence}/5
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
