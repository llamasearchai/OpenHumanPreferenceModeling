/**
 * Confidence Histogram Component
 *
 * Purpose: Display distribution of prediction confidences
 * before and after temperature scaling calibration.
 */

import { useQuery } from '@tanstack/react-query';
import { Skeleton } from '@/components/ui/skeleton';

// Types
interface ConfidenceDistribution {
  binStart: number;
  binEnd: number;
  preCount: number;
  postCount: number;
}

interface ConfidenceHistogramProps {
  preDistribution?: number[];
  postDistribution?: number[];
  height?: number;
  numBins?: number;
  showOverlay?: boolean;
}

// Generate mock distribution data
function generateMockDistribution(): ConfidenceDistribution[] {
  const numBins = 20;
  const bins: ConfidenceDistribution[] = [];

  for (let i = 0; i < numBins; i++) {
    const binStart = i / numBins;
    const binEnd = (i + 1) / numBins;
    const midpoint = (binStart + binEnd) / 2;

    // Pre-calibration: skewed toward high confidence (overconfident model)
    const preDist = Math.exp(-((midpoint - 0.85) ** 2) / 0.05);
    // Post-calibration: more spread out (better calibrated)
    const postDist = Math.exp(-((midpoint - 0.7) ** 2) / 0.1);

    bins.push({
      binStart,
      binEnd,
      preCount: Math.floor(preDist * 1000 + Math.random() * 50),
      postCount: Math.floor(postDist * 800 + Math.random() * 50),
    });
  }

  return bins;
}

export function ConfidenceHistogram({
  height = 200,
  showOverlay = true,
}: ConfidenceHistogramProps) {
  // Fetch distribution data
  const { data: distribution, isLoading } = useQuery({
    queryKey: ['calibration', 'confidence-distribution'],
    queryFn: async () => {
      // In production: return fetch('/api/calibration/confidence-distribution').then(r => r.json());
      return generateMockDistribution();
    },
  });

  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const width = 500;
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  if (isLoading) {
    return <Skeleton className="w-full" style={{ height }} />;
  }

  if (!distribution || distribution.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-muted-foreground"
        style={{ height }}
      >
        No distribution data available
      </div>
    );
  }

  const maxCount = Math.max(
    ...distribution.flatMap((d) => [d.preCount, d.postCount])
  );
  const binWidth = innerWidth / distribution.length;
  const barWidth = showOverlay ? binWidth - 2 : (binWidth - 4) / 2;

  // Calculate means for accessibility description
  const preMean = (
    distribution.reduce(
      (sum, d) => sum + d.preCount * ((d.binStart + d.binEnd) / 2),
      0
    ) / distribution.reduce((sum, d) => sum + d.preCount, 0) * 100
  ).toFixed(1);

  const postMean = (
    distribution.reduce(
      (sum, d) => sum + d.postCount * ((d.binStart + d.binEnd) / 2),
      0
    ) / distribution.reduce((sum, d) => sum + d.postCount, 0) * 100
  ).toFixed(1);

  return (
    <div className="space-y-4">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        style={{ maxHeight: height }}
        role="img"
        aria-label={`Confidence distribution histogram comparing pre-calibration (mean ${preMean}%) and post-calibration (mean ${postMean}%) prediction confidences across ${distribution.length} bins.`}
      >
        <title>Prediction Confidence Distribution</title>
        <desc>
          A histogram showing the distribution of prediction confidences before and after temperature scaling calibration.
          Pre-calibration mean: {preMean}%. Post-calibration mean: {postMean}%.
        </desc>
        <g transform={`translate(${padding.left}, ${padding.top})`}>
          {distribution.map((bin, i) => {
            const preHeight = (bin.preCount / maxCount) * innerHeight;
            const postHeight = (bin.postCount / maxCount) * innerHeight;

            if (showOverlay) {
              // Overlapping bars
              return (
                <g key={i}>
                  <rect
                    x={i * binWidth + 1}
                    y={innerHeight - preHeight}
                    width={barWidth}
                    height={preHeight}
                    className="fill-primary/50"
                  />
                  <rect
                    x={i * binWidth + 1}
                    y={innerHeight - postHeight}
                    width={barWidth}
                    height={postHeight}
                    className="fill-green-500/70"
                  />
                </g>
              );
            } else {
              // Side-by-side bars
              return (
                <g key={i}>
                  <rect
                    x={i * binWidth + 1}
                    y={innerHeight - preHeight}
                    width={barWidth}
                    height={preHeight}
                    className="fill-primary"
                  />
                  <rect
                    x={i * binWidth + barWidth + 2}
                    y={innerHeight - postHeight}
                    width={barWidth}
                    height={postHeight}
                    className="fill-green-500"
                  />
                </g>
              );
            }
          })}

          {/* X-axis */}
          <line
            x1={0}
            y1={innerHeight}
            x2={innerWidth}
            y2={innerHeight}
            className="stroke-border"
            strokeWidth={1}
          />

          {/* X-axis labels */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
            <g key={tick} transform={`translate(${tick * innerWidth}, ${innerHeight})`}>
              <line y1={0} y2={5} className="stroke-border" strokeWidth={1} />
              <text
                y={20}
                textAnchor="middle"
                className="fill-muted-foreground text-xs"
              >
                {(tick * 100).toFixed(0)}%
              </text>
            </g>
          ))}

          {/* X-axis title */}
          <text
            x={innerWidth / 2}
            y={innerHeight + 35}
            textAnchor="middle"
            className="fill-foreground text-xs font-medium"
          >
            Prediction Confidence
          </text>

          {/* Y-axis */}
          <line x1={0} y1={0} x2={0} y2={innerHeight} className="stroke-border" strokeWidth={1} />

          {/* Y-axis title */}
          <text
            x={-innerHeight / 2}
            y={-35}
            transform="rotate(-90)"
            textAnchor="middle"
            className="fill-foreground text-xs font-medium"
          >
            Count
          </text>
        </g>
      </svg>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div className={`w-4 h-4 rounded ${showOverlay ? 'bg-primary/50' : 'bg-primary'}`} />
          <span className="text-muted-foreground">Pre-calibration</span>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-4 h-4 rounded ${showOverlay ? 'bg-green-500/70' : 'bg-green-500'}`} />
          <span className="text-muted-foreground">Post-calibration</span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="text-center p-3 bg-muted/50 rounded-lg">
          <div className="text-muted-foreground mb-1">Pre-calibration Mean</div>
          <div className="font-mono font-medium">{preMean}%</div>
        </div>
        <div className="text-center p-3 bg-muted/50 rounded-lg">
          <div className="text-muted-foreground mb-1">Post-calibration Mean</div>
          <div className="font-mono font-medium">{postMean}%</div>
        </div>
      </div>
    </div>
  );
}

export default ConfidenceHistogram;
