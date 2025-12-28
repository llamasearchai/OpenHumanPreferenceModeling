/**
 * Annotation Statistics Widget
 *
 * Purpose: Display annotation statistics and progress
 */

import { useQuery } from '@tanstack/react-query';
import { annotationApi } from '@/lib/api-client';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { ProgressRing } from './ProgressRing';
import { StatCard } from './StatCard';
import { BarChart } from './BarChart';
import { Clock, CheckCircle2, TrendingUp, Target } from 'lucide-react';

interface AnnotationStatsProps {
  annotatorId: string;
}

export function AnnotationStats({ annotatorId }: AnnotationStatsProps) {
  const { data: annotationsData, isLoading } = useQuery({
    queryKey: ['annotations', 'stats', annotatorId],
    queryFn: async () => {
      const result = await annotationApi.listAnnotations({
        annotator_id: annotatorId,
        page: 1,
        page_size: 1000,
      });
      if (!result.success) throw new Error(result.error.detail);
      return result.data;
    },
  });

  const { data: qualityData, isLoading: qualityLoading } = useQuery({
    queryKey: ['quality', 'metrics', annotatorId],
    queryFn: async () => {
      const result = await annotationApi.getQualityMetrics(annotatorId);
      if (!result.success) throw new Error(result.error.detail);
      return result.data;
    },
  });

  if (isLoading || qualityLoading) {
    return (
      <div className="grid gap-4 md:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-32" />
        ))}
      </div>
    );
  }

  const annotations = annotationsData?.data || [];
  const totalAnnotations = annotations.length;
  const avgTime = annotations.length
    ? annotations.reduce((sum, a) => sum + a.time_spent_seconds, 0) / annotations.length
    : 0;
  const avgConfidence = annotations.length
    ? annotations.reduce((sum, a) => sum + a.confidence, 0) / annotations.length
    : 0;

  // Group by type
  const typeDistribution = annotations.reduce((acc, a) => {
    acc[a.annotation_type] = (acc[a.annotation_type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const typeChartData = Object.entries(typeDistribution).map(([name, value]) => ({
    name,
    value,
  }));

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <StatCard
          title="Total Annotations"
          value={totalAnnotations}
          icon={CheckCircle2}
          description="All time"
        />
        <StatCard
          title="Avg Time per Task"
          value={`${avgTime.toFixed(1)}s`}
          icon={Clock}
          description="Average completion time"
        />
        <StatCard
          title="Avg Confidence"
          value={`${avgConfidence.toFixed(1)}/5`}
          icon={Target}
          description="Average confidence score"
        />
        <div className="flex items-center justify-center">
          <ProgressRing
            value={qualityData?.agreement_score ? qualityData.agreement_score * 100 : 0}
            max={100}
            size={100}
            label="Agreement"
            showLabel
          />
        </div>
      </div>

      {/* Charts */}
      {typeChartData.length > 0 && (
        <BarChart
          title="Annotations by Type"
          description="Distribution of annotation types"
          data={typeChartData}
          height={250}
        />
      )}

      {/* Quality Metrics */}
      {qualityData && (
        <Card>
          <CardHeader>
            <CardTitle>Quality Metrics</CardTitle>
            <CardDescription>Your annotation quality scores</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <div className="text-sm text-muted-foreground">Agreement Score</div>
                <div className="text-2xl font-bold">
                  {(qualityData.agreement_score * 100).toFixed(1)}%
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-sm text-muted-foreground">Gold Pass Rate</div>
                <div className="text-2xl font-bold">
                  {(qualityData.gold_pass_rate * 100).toFixed(1)}%
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-sm text-muted-foreground">Avg Time</div>
                <div className="text-2xl font-bold">
                  {qualityData.avg_time_per_task.toFixed(1)}s
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
