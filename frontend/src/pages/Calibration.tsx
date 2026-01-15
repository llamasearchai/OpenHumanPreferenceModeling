/**
 * Calibration Page
 *
 * Purpose: Model calibration controls and status
 */

import * as React from 'react';
import { useMutation } from '@tanstack/react-query';
import { calibrationApi } from '@/lib/api-client';
import { useToast } from '@/hooks/use-toast';
import { ApiRequestError, extractErrorMessage } from '@/lib/errors';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import {
  Gauge,
  Play,
  CheckCircle2,
  AlertTriangle,
  Info,
} from 'lucide-react';
import type { RecalibrationResponse } from '@/types/api';
import { ConfidenceHistogram } from '@/components/visualizations/ConfidenceHistogram';
import { ReliabilityDiagram } from '@/components/visualizations/ReliabilityDiagram';

export default function CalibrationPage() {
  const { toast } = useToast();

  const [validationUri, setValidationUri] = React.useState('');
  const [targetEce, setTargetEce] = React.useState<number[]>([0.08]);
  const [maxIterations, setMaxIterations] = React.useState<number[]>([100]);
  const [result, setResult] = React.useState<RecalibrationResponse | null>(null);

  const recalibrateMutation = useMutation({
    mutationFn: async () => {
      const response = await calibrationApi.triggerRecalibration({
        validation_data_uri: validationUri,
        target_ece: targetEce[0] ?? 0.08,
        max_iterations: maxIterations[0] ?? 100,
      });
      if (!response.success) {
        throw ApiRequestError.fromResponse(response);
      }
      return response.data;
    },
    onSuccess: (data) => {
      setResult(data);
      toast({
        title: 'Recalibration complete',
        description: `New temperature: ${data.temperature.toFixed(4)}`,
        variant: 'success',
      });
    },
    onError: (error) => {
      toast({
        title: 'Recalibration failed',
        description: extractErrorMessage(error),
        variant: 'destructive',
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validationUri.trim()) {
      toast({
        title: 'Validation required',
        description: 'Please enter a validation data URI',
        variant: 'destructive',
      });
      return;
    }
    recalibrateMutation.mutate();
  };

  const eceImprovement = result && result.pre_ece > 0
    ? ((result.pre_ece - result.post_ece) / result.pre_ece) * 100
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Calibration</h1>
        <p className="text-muted-foreground">
          Manage model confidence calibration
        </p>
      </div>

      {/* Info Alert */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertTitle>About Calibration</AlertTitle>
        <AlertDescription>
          Temperature scaling adjusts model confidence to better match actual
          accuracy. A well-calibrated model's predicted probabilities align with
          observed frequencies.
        </AlertDescription>
      </Alert>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Recalibration Form */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Gauge className="h-5 w-5" />
              Trigger Recalibration
            </CardTitle>
            <CardDescription>
              Run temperature scaling on validation data
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <Input
                label="Validation Data URI"
                value={validationUri}
                onChange={(e) => setValidationUri(e.target.value)}
                helperText="S3 or local path to validation dataset"
              />

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Target ECE</label>
                  <Badge variant="outline">{(targetEce[0] ?? 0.08).toFixed(2)}</Badge>
                </div>
                <Slider
                  value={targetEce}
                  onValueChange={setTargetEce}
                  min={0.01}
                  max={0.2}
                  step={0.01}
                />
                <p className="text-xs text-muted-foreground">
                  Expected Calibration Error threshold (lower is better)
                </p>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Max Iterations</label>
                  <Badge variant="outline">{maxIterations[0] ?? 100}</Badge>
                </div>
                <Slider
                  value={maxIterations}
                  onValueChange={setMaxIterations}
                  min={10}
                  max={500}
                  step={10}
                />
                <p className="text-xs text-muted-foreground">
                  Maximum optimization iterations
                </p>
              </div>

              <Button
                type="submit"
                className="w-full"
                isLoading={recalibrateMutation.isPending}
              >
                <Play className="mr-2 h-4 w-4" />
                Start Recalibration
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Results Card */}
        <Card>
          <CardHeader>
            <CardTitle>Calibration Results</CardTitle>
            <CardDescription>
              {result
                ? 'Latest recalibration outcome'
                : 'Run recalibration to see results'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {recalibrateMutation.isPending ? (
              <div className="space-y-4 py-8">
                <Progress value={undefined} className="w-full" />
                <p className="text-center text-sm text-muted-foreground">
                  Optimizing temperature parameter...
                </p>
              </div>
            ) : result ? (
              <div className="space-y-6">
                {/* Success indicator */}
                <div className="flex items-center gap-3 p-4 rounded-lg bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800">
                  <CheckCircle2 className="h-8 w-8 text-green-600" />
                  <div>
                    <p className="font-medium text-green-900 dark:text-green-100">
                      Calibration Successful
                    </p>
                    <p className="text-sm text-green-700 dark:text-green-300">
                      Completed in {result.iterations} iterations
                    </p>
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid gap-4">
                  <div className="flex items-center justify-between p-3 rounded-lg border">
                    <span className="text-sm font-medium">Temperature</span>
                    <span className="font-mono text-lg">
                      {result.temperature.toFixed(4)}
                    </span>
                  </div>

                  <div className="flex items-center justify-between p-3 rounded-lg border">
                    <span className="text-sm font-medium">Pre-calibration ECE</span>
                    <span className="font-mono text-lg text-destructive">
                      {result.pre_ece.toFixed(4)}
                    </span>
                  </div>

                  <div className="flex items-center justify-between p-3 rounded-lg border">
                    <span className="text-sm font-medium">Post-calibration ECE</span>
                    <span className="font-mono text-lg text-green-600">
                      {result.post_ece.toFixed(4)}
                    </span>
                  </div>

                  <div className="flex items-center justify-between p-3 rounded-lg border bg-muted/50">
                    <span className="text-sm font-medium">Improvement</span>
                    <Badge variant="success">
                      {eceImprovement.toFixed(1)}% reduction
                    </Badge>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <Gauge className="h-16 w-16 text-muted-foreground mb-4" />
                <p className="text-lg font-medium">No Results Yet</p>
                <p className="text-sm text-muted-foreground max-w-xs">
                  Configure and run recalibration to optimize model confidence
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Error display */}
      {recalibrateMutation.error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Recalibration Failed</AlertTitle>
          <AlertDescription>
            {recalibrateMutation.error instanceof Error
              ? recalibrateMutation.error.message
              : 'An unknown error occurred'}
          </AlertDescription>
        </Alert>
      )}

      {/* Visualizations */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Confidence Distribution</CardTitle>
            <CardDescription>
              Distribution of prediction confidences before and after calibration
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ConfidenceHistogram height={300} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Reliability Diagram</CardTitle>
            <CardDescription>
              Predicted confidence vs observed accuracy per bin
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ReliabilityDiagram height={300} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
