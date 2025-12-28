/**
 * Reliability Diagram Component
 *
 * Purpose: Visualize model calibration using a reliability diagram
 * showing predicted confidence vs observed accuracy per bin.
 */

import { useQuery } from '@tanstack/react-query';
import { Info } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Skeleton } from '@/components/ui/skeleton';

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
    const accuracy =
      midpoint < 0.5
        ? midpoint + (Math.random() - 0.5) * 0.1
        : midpoint - 0.1 + (Math.random() - 0.5) * 0.1;

    const count = Math.floor(
      1000 * Math.exp(-((midpoint - 0.7) ** 2) / 0.2) + Math.random() * 100
    );

    bins.push({
      binIndex: i,
      binStart,
      binEnd,
      count: Math.max(10, count),
      avgConfidence: midpoint + (Math.random() - 0.5) * 0.05,
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
}

function BinBar({
  bin,
  maxCount,
  height,
  binWidth,
  showGap,
  isSelected,
  onClick,
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

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <g
            className={`cursor-pointer transition-opacity ${isSelected ? 'opacity-100' : 'hover:opacity-80'}`}
            onClick={onClick}
          >
            {/* Background (expected = confidence) */}
            <rect
              x={bin.binIndex * binWidth + barPadding}
              y={height - expectedHeight}
              width={actualBarWidth}
              height={expectedHeight}
              className="fill-muted"
            />

            {/* Actual accuracy bar */}
            <rect
              x={bin.binIndex * binWidth + barPadding}
              y={height - barHeight}
              width={actualBarWidth}
              height={barHeight}
              className="fill-primary"
              style={{ opacity }}
            />

            {/* Gap indicator */}
            {showGap && Math.abs(gap) > 5 && (
              <rect
                x={bin.binIndex * binWidth + barPadding}
                y={isOverconfident ? height - barHeight : height - expectedHeight}
                width={actualBarWidth}
                height={Math.abs(gap)}
                className={isOverconfident ? 'fill-destructive/30' : 'fill-green-500/30'}
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
  const padding = { top: 20, right: 30, bottom: 40, left: 50 };
  const width = 500;
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

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

  const maxCount = Math.max(...bins.map((b) => b.count));
  const binWidth = innerWidth / bins.length;

  // Calculate ECE
  const totalCount = bins.reduce((sum, b) => sum + b.count, 0);
  const ece =
    bins.reduce((sum, b) => sum + (b.count / totalCount) * b.calibrationError, 0) *
    100;

  return (
    <div className="space-y-4">
      <svg
        viewBox={`0 0 ${width} ${height}`}
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
        <g transform={`translate(${padding.left}, ${padding.top})`}>
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

      {/* Legend and stats */}
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-primary rounded" />
            <span className="text-muted-foreground">Observed Accuracy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-muted rounded" />
            <span className="text-muted-foreground">Perfect Calibration</span>
          </div>
          {showGap && (
            <>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-destructive/30 rounded" />
                <span className="text-muted-foreground">Overconfident</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500/30 rounded" />
                <span className="text-muted-foreground">Underconfident</span>
              </div>
            </>
          )}
        </div>

        <div className="flex items-center gap-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <Info className="h-4 w-4 text-muted-foreground" />
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
    </div>
  );
}

export default ReliabilityDiagram;
