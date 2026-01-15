/**
 * Reliability Diagram Component
 *
 * Purpose: Visualize model calibration using a reliability diagram
 * showing predicted confidence vs observed accuracy per bin.
 */

import { useQuery } from '@tanstack/react-query';
import { useId, type KeyboardEvent } from 'react';
import { Info } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Skeleton } from '@/components/ui/skeleton';
import { useChartDimensions } from '@/lib/visualizations/use-chart-dimensions';

// Types
interface CalibrationBin {
  binIndex: number;
  binStart: number;
  binEnd: number;
  count: number;
  avgConfidence: number;
  avgAccuracy: number;
  calibrationError: number;
}

interface ReliabilityDiagramProps {
  bins?: CalibrationBin[];
  height?: number;
  showGap?: boolean;
  onBinClick?: (bin: CalibrationBin) => void;
  selectedBin?: number | null;
}

// Generate mock calibration data
function generateMockBins(): CalibrationBin[] {
  const numBins = 10;
  const bins: CalibrationBin[] = [];

  for (let i = 0; i < numBins; i++) {
    const binStart = i / numBins;
    const binEnd = (i + 1) / numBins;
    const midpoint = (binStart + binEnd) / 2;

    // Simulate typical overconfidence at high confidence levels
    const variance = ((i % 3) - 1) * 0.02;
    const accuracy =
      midpoint < 0.5
        ? midpoint + 0.04 + variance
        : midpoint - 0.08 + variance;

    const count = Math.round(
      1000 * Math.exp(-((midpoint - 0.7) ** 2) / 0.2) + (i % 4) * 18
    );

    bins.push({
      binIndex: i,
      binStart,
      binEnd,
      count: Math.max(10, count),
      avgConfidence: midpoint + ((i % 5) - 2) * 0.01,
      avgAccuracy: Math.max(0, Math.min(1, accuracy)),
      calibrationError: Math.abs(midpoint - accuracy),
    });
  }

  return bins;
}

// Bin bar component
interface BinBarProps {
  bin: CalibrationBin;
  maxCount: number;
  height: number;
  binWidth: number;
  showGap: boolean;
  isSelected: boolean;
  onClick: () => void;
  patternIds: {
    expected: string;
    observed: string;
    over: string;
    under: string;
  };
}

function BinBar({
  bin,
  maxCount,
  height,
  binWidth,
  showGap,
  isSelected,
  onClick,
  patternIds,
}: BinBarProps) {
  const barHeight = (bin.avgAccuracy / 1) * height;
  const expectedHeight = (bin.avgConfidence / 1) * height;
  const gap = barHeight - expectedHeight;
  const isOverconfident = gap < 0;

  // Bar width with padding
  const barPadding = 2;
  const actualBarWidth = binWidth - barPadding * 2;

  // Opacity based on sample count
  const opacity = 0.4 + (bin.count / maxCount) * 0.6;

  const handleKeyDown = (event: KeyboardEvent<SVGGElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onClick();
    }
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <g
            className={`cursor-pointer transition-opacity ${isSelected ? 'opacity-100' : 'hover:opacity-80'}`}
            onClick={onClick}
            onKeyDown={handleKeyDown}
            role="button"
            tabIndex={0}
            aria-label={`Confidence ${Math.round(bin.binStart * 100)} to ${Math.round(
              bin.binEnd * 100
            )} percent. Accuracy ${(bin.avgAccuracy * 100).toFixed(1)} percent.`}
          >
            {/* Background (expected = confidence) */}
            <rect
              x={bin.binIndex * binWidth + barPadding}
              y={height - expectedHeight}
              width={actualBarWidth}
              height={expectedHeight}
              fill={`url(#${patternIds.expected})`}
            />

            {/* Actual accuracy bar */}
            <rect
              x={bin.binIndex * binWidth + barPadding}
              y={height - barHeight}
              width={actualBarWidth}
              height={barHeight}
              fill={`url(#${patternIds.observed})`}
              style={{ opacity }}
            />

            {/* Gap indicator - using semantic colors for accessibility */}
            {showGap && Math.abs(gap) > 5 && (
              <rect
                x={bin.binIndex * binWidth + barPadding}
                y={isOverconfident ? height - barHeight : height - expectedHeight}
                width={actualBarWidth}
                height={Math.abs(gap)}
                fill={`url(#${isOverconfident ? patternIds.over : patternIds.under})`}
                role="img"
                aria-label={isOverconfident ? 'Overconfident gap' : 'Underconfident gap'}
              />
            )}

            {/* Selection indicator */}
            {isSelected && (
              <rect
                x={bin.binIndex * binWidth}
                y={0}
                width={binWidth}
                height={height}
                className="fill-primary/10 stroke-primary"
                strokeWidth={2}
              />
            )}

            {/* Low sample count indicator */}
            {bin.count < 30 && (
              <line
                x1={bin.binIndex * binWidth + barPadding}
                x2={bin.binIndex * binWidth + actualBarWidth}
                y1={height - barHeight}
                y2={height - barHeight}
                strokeDasharray="4,4"
                className="stroke-muted-foreground"
                strokeWidth={2}
              />
            )}
          </g>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs">
          <div className="space-y-1 text-xs">
            <div className="font-medium">
              Confidence: {(bin.binStart * 100).toFixed(0)}% -{' '}
              {(bin.binEnd * 100).toFixed(0)}%
            </div>
            <div className="grid grid-cols-2 gap-x-4">
              <span className="text-muted-foreground">Accuracy:</span>
              <span className="font-mono">{(bin.avgAccuracy * 100).toFixed(1)}%</span>
              <span className="text-muted-foreground">Avg Confidence:</span>
              <span className="font-mono">{(bin.avgConfidence * 100).toFixed(1)}%</span>
              <span className="text-muted-foreground">Samples:</span>
              <span className="font-mono">{bin.count.toLocaleString()}</span>
              <span className="text-muted-foreground">Calibration Error:</span>
              <span className="font-mono">{(bin.calibrationError * 100).toFixed(1)}%</span>
            </div>
            {bin.count < 30 && (
              <div className="text-yellow-600 dark:text-yellow-400 mt-1">
                Low sample count - interpret with caution
              </div>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

// Main component
export function ReliabilityDiagram({
  bins: propBins,
  height = 300,
  showGap = true,
  onBinClick,
  selectedBin,
}: ReliabilityDiagramProps) {
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
    marginRight: 30,
    marginBottom: 40,
    marginLeft: 50,
  });

  const patternBaseId = useId();
  const safePatternId = patternBaseId.replace(/:/g, '');
  const patternIds = {
    expected: `${safePatternId}-expected`,
    observed: `${safePatternId}-observed`,
    over: `${safePatternId}-over`,
    under: `${safePatternId}-under`,
  };

  const observedLegendStyle = {
    backgroundImage:
      'repeating-linear-gradient(90deg, currentColor 0, currentColor 2px, transparent 2px, transparent 6px)',
  };
  const expectedLegendStyle = {
    backgroundImage:
      'repeating-linear-gradient(45deg, currentColor 0, currentColor 2px, transparent 2px, transparent 6px)',
  };
  const overLegendStyle = {
    backgroundImage:
      'repeating-linear-gradient(135deg, currentColor 0, currentColor 2px, transparent 2px, transparent 6px)',
  };
  const underLegendStyle = {
    backgroundImage:
      'radial-gradient(circle at 2px 2px, currentColor 1.3px, transparent 1.3px)',
    backgroundSize: '6px 6px',
  };

  // Fetch bins if not provided
  const { data: fetchedBins, isLoading } = useQuery({
    queryKey: ['calibration', 'reliability-bins'],
    queryFn: async () => {
      // In production: return fetch('/api/calibration/reliability-bins').then(r => r.json());
      return generateMockBins();
    },
    enabled: !propBins,
  });

  const bins = propBins || fetchedBins;
  const innerWidth = boundedWidth;
  const innerHeight = boundedHeight;

  if (isLoading) {
    return <Skeleton className="w-full" style={{ height }} />;
  }

  if (!bins || bins.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-muted-foreground"
        style={{ height }}
      >
        No calibration data available
      </div>
    );
  }

  if (!isReady || width === 0 || measuredHeight === 0) {
    return <Skeleton className="w-full" style={{ height }} />;
  }

  const maxCount = Math.max(...bins.map((b) => b.count));
  const binWidth = innerWidth / bins.length;

  // Calculate ECE
  const totalCount = bins.reduce((sum, b) => sum + b.count, 0);
  const ece =
    bins.reduce((sum, b) => sum + (b.count / totalCount) * b.calibrationError, 0) *
    100;

  return (
    <div className="space-y-4">
      <div ref={chartRef} className="w-full" style={{ height }}>
        <svg
          viewBox={`0 0 ${width} ${measuredHeight}`}
          className="w-full"
          style={{ maxHeight: height }}
          role="img"
          aria-label={`Reliability diagram showing model calibration across ${bins.length} confidence bins. Expected Calibration Error: ${ece.toFixed(2)}%. The chart compares predicted confidence (x-axis) to observed accuracy (y-axis).`}
        >
          <title>Model Calibration Reliability Diagram</title>
          <desc>
            A bar chart comparing predicted confidence to observed accuracy across {bins.length} bins.
            Perfect calibration would follow the diagonal line. ECE: {ece.toFixed(2)}%.
          </desc>
          <defs>
            <pattern
              id={patternIds.expected}
              width="6"
              height="6"
              patternUnits="userSpaceOnUse"
              className="text-muted-foreground"
            >
              <rect width="6" height="6" fill="currentColor" opacity="0.15" />
              <path d="M0 6 L6 0" stroke="currentColor" strokeWidth="1" />
            </pattern>
            <pattern
              id={patternIds.observed}
              width="6"
              height="6"
              patternUnits="userSpaceOnUse"
              className="text-primary"
            >
              <rect width="6" height="6" fill="currentColor" opacity="0.25" />
              <path d="M0 0 L0 6 M3 0 L3 6" stroke="currentColor" strokeWidth="1" />
            </pattern>
            <pattern
              id={patternIds.over}
              width="6"
              height="6"
              patternUnits="userSpaceOnUse"
              className="text-destructive"
            >
              <rect width="6" height="6" fill="currentColor" opacity="0.2" />
              <path d="M0 6 L6 0" stroke="currentColor" strokeWidth="1" />
            </pattern>
            <pattern
              id={patternIds.under}
              width="6"
              height="6"
              patternUnits="userSpaceOnUse"
              className="text-emerald-500 dark:text-emerald-400"
            >
              <rect width="6" height="6" fill="currentColor" opacity="0.2" />
              <circle cx="3" cy="3" r="1.2" fill="currentColor" />
            </pattern>
          </defs>
          <g transform={`translate(${margins.left}, ${margins.top})`}>
            {/* Diagonal reference line (perfect calibration) */}
            <line
              x1={0}
              y1={innerHeight}
              x2={innerWidth}
              y2={0}
              className="stroke-muted-foreground"
              strokeWidth={1}
              strokeDasharray="4,4"
            />

            {/* Bars */}
            {bins.map((bin) => (
              <BinBar
                key={bin.binIndex}
                bin={bin}
                maxCount={maxCount}
                height={innerHeight}
                binWidth={binWidth}
                showGap={showGap}
                isSelected={selectedBin === bin.binIndex}
                onClick={() => onBinClick?.(bin)}
                patternIds={patternIds}
              />
            ))}

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
          {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => (
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
            Predicted Confidence
          </text>

          {/* Y-axis */}
          <line x1={0} y1={0} x2={0} y2={innerHeight} className="stroke-border" strokeWidth={1} />

          {/* Y-axis labels */}
          {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => (
            <g
              key={tick}
              transform={`translate(0, ${innerHeight - tick * innerHeight})`}
            >
              <line x1={-5} x2={0} className="stroke-border" strokeWidth={1} />
              <text
                x={-10}
                textAnchor="end"
                dominantBaseline="middle"
                className="fill-muted-foreground text-xs"
              >
                {(tick * 100).toFixed(0)}%
              </text>
            </g>
          ))}

          {/* Y-axis title */}
          <text
            x={-innerHeight / 2}
            y={-35}
            transform="rotate(-90)"
            textAnchor="middle"
            className="fill-foreground text-xs font-medium"
          >
            Observed Accuracy
          </text>
        </g>
      </svg>
      </div>

      {/* Legend and stats */}
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-3 bg-primary text-primary rounded"
              style={observedLegendStyle}
              aria-hidden="true"
            />
            <span className="text-muted-foreground">Observed Accuracy (striped)</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-3 bg-muted text-muted-foreground rounded"
              style={expectedLegendStyle}
              aria-hidden="true"
            />
            <span className="text-muted-foreground">Perfect Calibration (hatched)</span>
          </div>
          {showGap && (
            <>
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 bg-destructive/40 text-destructive rounded"
                  style={overLegendStyle}
                  aria-hidden="true"
                />
                <span className="text-muted-foreground">Overconfident (striped)</span>
              </div>
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 bg-emerald-500/40 text-emerald-500 dark:bg-emerald-400/40 dark:text-emerald-400 rounded"
                  style={underLegendStyle}
                  aria-hidden="true"
                />
                <span className="text-muted-foreground">Underconfident (dotted)</span>
              </div>
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-label="About Expected Calibration Error (ECE)"
                  className="inline-flex items-center justify-center rounded-sm focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                >
                  <Info className="h-4 w-4 text-muted-foreground" />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs text-xs">
                  Expected Calibration Error (ECE) measures the average gap between
                  predicted confidence and observed accuracy, weighted by sample count.
                  Lower is better.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          <span className="font-medium">
            ECE: <span className="font-mono">{ece.toFixed(2)}%</span>
          </span>
        </div>
      </div>

      <details className="rounded-lg border bg-background/60 p-3">
        <summary className="cursor-pointer text-sm font-medium">View data table</summary>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-xs">
            <caption className="sr-only">Reliability diagram bin statistics</caption>
            <thead>
              <tr className="text-left text-muted-foreground border-b">
                <th scope="col" className="py-1 pr-2">
                  Bin range
                </th>
                <th scope="col" className="py-1 pr-2 text-right">
                  Avg confidence
                </th>
                <th scope="col" className="py-1 pr-2 text-right">
                  Avg accuracy
                </th>
                <th scope="col" className="py-1 pr-2 text-right">
                  Samples
                </th>
                <th scope="col" className="py-1 pr-2 text-right">
                  Calibration error
                </th>
              </tr>
            </thead>
            <tbody>
              {bins.map((bin) => (
                <tr key={bin.binIndex} className="border-b last:border-b-0">
                  <td className="py-1 pr-2 font-mono">
                    {(bin.binStart * 100).toFixed(0)}% - {(bin.binEnd * 100).toFixed(0)}%
                  </td>
                  <td className="py-1 pr-2 text-right font-mono">
                    {(bin.avgConfidence * 100).toFixed(1)}%
                  </td>
                  <td className="py-1 pr-2 text-right font-mono">
                    {(bin.avgAccuracy * 100).toFixed(1)}%
                  </td>
                  <td className="py-1 pr-2 text-right font-mono">
                    {bin.count.toLocaleString()}
                  </td>
                  <td className="py-1 pr-2 text-right font-mono">
                    {(bin.calibrationError * 100).toFixed(1)}%
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

export default ReliabilityDiagram;
