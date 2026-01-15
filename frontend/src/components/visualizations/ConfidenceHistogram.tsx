/**
 * Confidence Histogram Component
 *
 * Purpose: Display distribution of prediction confidences
 * before and after temperature scaling calibration.
 */

import * as React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Skeleton } from '@/components/ui/skeleton';
import { useChartDimensions } from '@/lib/visualizations/use-chart-dimensions';

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

    const preCount = Math.round(preDist * 1000 + (i % 4) * 12);
    const postCount = Math.round(postDist * 800 + (i % 5) * 10);

    bins.push({
      binStart,
      binEnd,
      preCount,
      postCount,
    });
  }

  return bins;
}

export function ConfidenceHistogram({
  height = 200,
  showOverlay = true,
}: ConfidenceHistogramProps) {
  const patternBaseId = React.useId();
  const safePatternId = patternBaseId.replace(/:/g, '');
  const prePatternId = `${safePatternId}-pre`;
  const postPatternId = `${safePatternId}-post`;

  const preLegendStyle = {
    backgroundImage:
      'repeating-linear-gradient(45deg, rgba(255,255,255,0.7) 0, rgba(255,255,255,0.7) 2px, transparent 2px, transparent 6px)',
  };
  const postLegendStyle = {
    backgroundImage:
      'radial-gradient(circle at 2px 2px, rgba(255,255,255,0.7) 1.2px, transparent 1.2px)',
    backgroundSize: '6px 6px',
  };

  const {
    ref: chartRef,
    width,
    height: measuredHeight,
    boundedWidth,
    boundedHeight,
    margins,
    isReady,
  } = useChartDimensions({
    marginTop: 20,
    marginRight: 20,
    marginBottom: 40,
    marginLeft: 50,
  });

  // Fetch distribution data
  const { data: distribution, isLoading } = useQuery({
    queryKey: ['calibration', 'confidence-distribution'],
    queryFn: async () => {
      // In production: return fetch('/api/calibration/confidence-distribution').then(r => r.json());
      return generateMockDistribution();
    },
  });

  const innerWidth = boundedWidth;
  const innerHeight = boundedHeight;

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

  if (!isReady || width === 0 || measuredHeight === 0) {
    return <Skeleton className="w-full" style={{ height }} />;
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
      <div ref={chartRef} className="w-full" style={{ height }}>
        <svg
          viewBox={`0 0 ${width} ${measuredHeight}`}
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
          <defs>
            <pattern
              id={prePatternId}
              width="6"
              height="6"
              patternUnits="userSpaceOnUse"
              className="text-primary"
            >
              <rect width="6" height="6" fill="currentColor" opacity="0.2" />
              <path d="M0 6 L6 0" stroke="currentColor" strokeWidth="1" />
            </pattern>
            <pattern
              id={postPatternId}
              width="6"
              height="6"
              patternUnits="userSpaceOnUse"
              className="text-green-500"
            >
              <rect width="6" height="6" fill="currentColor" opacity="0.2" />
              <circle cx="3" cy="3" r="1.2" fill="currentColor" />
            </pattern>
          </defs>
          <g transform={`translate(${margins.left}, ${margins.top})`}>
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
                      fill={`url(#${prePatternId})`}
                      opacity={0.6}
                    />
                    <rect
                      x={i * binWidth + 1}
                      y={innerHeight - postHeight}
                      width={barWidth}
                      height={postHeight}
                      fill={`url(#${postPatternId})`}
                      opacity={0.8}
                    />
                  </g>
                );
              }

              // Side-by-side bars
              return (
                <g key={i}>
                  <rect
                    x={i * binWidth + 1}
                    y={innerHeight - preHeight}
                    width={barWidth}
                    height={preHeight}
                    fill={`url(#${prePatternId})`}
                  />
                  <rect
                    x={i * binWidth + barWidth + 2}
                    y={innerHeight - postHeight}
                    width={barWidth}
                    height={postHeight}
                    fill={`url(#${postPatternId})`}
                  />
                </g>
              );
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
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <div
            className={`w-4 h-4 rounded ${showOverlay ? 'bg-primary/50' : 'bg-primary'}`}
            style={{ ...preLegendStyle, opacity: showOverlay ? 0.6 : 1 }}
            aria-hidden="true"
          />
          <span className="text-muted-foreground">Pre-calibration (striped)</span>
        </div>
        <div className="flex items-center gap-2">
          <div
            className={`w-4 h-4 rounded ${showOverlay ? 'bg-green-500/70' : 'bg-green-500'}`}
            style={{ ...postLegendStyle, opacity: showOverlay ? 0.8 : 1 }}
            aria-hidden="true"
          />
          <span className="text-muted-foreground">Post-calibration (dotted)</span>
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

      <details className="rounded-lg border bg-background/60 p-3">
        <summary className="cursor-pointer text-sm font-medium">View data table</summary>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-xs">
            <caption className="sr-only">Confidence distribution data by bin</caption>
            <thead>
              <tr className="text-left text-muted-foreground border-b">
                <th scope="col" className="py-1 pr-2">
                  Bin range
                </th>
                <th scope="col" className="py-1 pr-2 text-right">
                  Pre count
                </th>
                <th scope="col" className="py-1 pr-2 text-right">
                  Post count
                </th>
              </tr>
            </thead>
            <tbody>
              {distribution.map((bin) => (
                <tr key={`${bin.binStart}-${bin.binEnd}`} className="border-b last:border-b-0">
                  <td className="py-1 pr-2 font-mono">
                    {(bin.binStart * 100).toFixed(0)}% - {(bin.binEnd * 100).toFixed(0)}%
                  </td>
                  <td className="py-1 pr-2 text-right font-mono">
                    {bin.preCount.toLocaleString()}
                  </td>
                  <td className="py-1 pr-2 text-right font-mono">
                    {bin.postCount.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

export default ConfidenceHistogram;
