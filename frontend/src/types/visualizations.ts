/**
 * Visualization Type Definitions
 * 
 * Purpose: Complete type definitions for D3.js/Visx visualizations
 * including data schemas, chart configurations, and interaction types
 * 
 * Design decisions:
 * - Generic types for flexible data handling
 * - Accessibility-first design with ARIA requirements
 * - Responsive configuration options
 */

// ============================================================================
// Data Types
// ============================================================================

/** Time series data point */
export interface TimeSeriesDataPoint {
  /** Timestamp */
  date: Date;
  /** Value */
  value: number;
  /** Optional label */
  label?: string;
  /** Optional category */
  category?: string;
}

/** Categorical data point */
export interface CategoricalDataPoint {
  /** Category name */
  category: string;
  /** Value */
  value: number;
  /** Optional subcategory */
  subcategory?: string;
  /** Optional color override */
  color?: string;
}

/** Multi-series time series data */
export interface MultiSeriesDataPoint {
  /** Timestamp */
  date: Date;
  /** Values by series key */
  values: Record<string, number>;
}

/** Scatter plot data point */
export interface ScatterDataPoint {
  /** X value */
  x: number;
  /** Y value */
  y: number;
  /** Optional size value */
  size?: number;
  /** Optional category for color */
  category?: string;
  /** Optional label */
  label?: string;
  /** Optional ID for selection */
  id?: string;
}

/** Network node */
export interface NetworkNode {
  /** Unique identifier */
  id: string;
  /** Display label */
  label: string;
  /** Node group/category */
  group?: string;
  /** Node size */
  size?: number;
  /** Node color */
  color?: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/** Network link */
export interface NetworkLink {
  /** Source node ID */
  source: string;
  /** Target node ID */
  target: string;
  /** Link weight/strength */
  weight?: number;
  /** Link label */
  label?: string;
}

/** Hierarchical data node */
export interface HierarchyNode<T = unknown> {
  /** Node name */
  name: string;
  /** Node value */
  value?: number;
  /** Children nodes */
  children?: HierarchyNode<T>[];
  /** Additional data */
  data?: T;
}

/** Geographic data point */
export interface GeoDataPoint {
  /** Longitude */
  lng: number;
  /** Latitude */
  lat: number;
  /** Value for coloring/sizing */
  value?: number;
  /** Label */
  label?: string;
  /** Category */
  category?: string;
  /** Additional properties */
  properties?: Record<string, unknown>;
}

// ============================================================================
// Scale Types
// ============================================================================

/** Scale type enumeration */
export type ScaleType = 
  | 'linear'
  | 'log'
  | 'pow'
  | 'sqrt'
  | 'symlog'
  | 'time'
  | 'utc'
  | 'band'
  | 'point'
  | 'ordinal'
  | 'sequential'
  | 'diverging'
  | 'quantize'
  | 'quantile'
  | 'threshold';

/** Scale configuration */
export interface ScaleConfig {
  /** Scale type */
  type: ScaleType;
  /** Domain (input range) */
  domain?: [number, number] | string[] | Date[];
  /** Range (output range) */
  range?: [number, number] | string[];
  /** Clamp values to range */
  clamp?: boolean;
  /** Nice rounding */
  nice?: boolean;
  /** Zero baseline */
  zero?: boolean;
  /** Padding for band/point scales */
  padding?: number;
  /** Exponent for pow scale */
  exponent?: number;
  /** Base for log scale */
  base?: number;
}

// ============================================================================
// Axis Types
// ============================================================================

/** Axis position */
export type AxisPosition = 'top' | 'right' | 'bottom' | 'left';

/** Axis configuration */
export interface AxisConfiguration {
  /** Axis position */
  position: AxisPosition;
  /** Axis label */
  label?: string;
  /** Label offset from axis */
  labelOffset?: number;
  /** Tick count hint */
  numTicks?: number;
  /** Tick values override */
  tickValues?: (number | string | Date)[];
  /** Tick format function */
  tickFormat?: (value: unknown, index: number) => string;
  /** Tick length */
  tickLength?: number;
  /** Hide tick marks */
  hideTicks?: boolean;
  /** Hide axis line */
  hideAxisLine?: boolean;
  /** Hide zero line */
  hideZero?: boolean;
  /** Tick label rotation (degrees) */
  tickLabelRotation?: number;
  /** Tick label props */
  tickLabelProps?: {
    dx?: number;
    dy?: number;
    textAnchor?: 'start' | 'middle' | 'end';
    fontSize?: number;
  };
}

/** Grid configuration */
export interface GridConfiguration {
  /** Show horizontal grid lines */
  horizontal?: boolean;
  /** Show vertical grid lines */
  vertical?: boolean;
  /** Grid line stroke color */
  stroke?: string;
  /** Grid line stroke width */
  strokeWidth?: number;
  /** Grid line dash pattern */
  strokeDasharray?: string;
  /** Grid line opacity */
  opacity?: number;
}

// ============================================================================
// Legend Types
// ============================================================================

/** Legend position */
export type LegendPosition = 'top' | 'right' | 'bottom' | 'left' | 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';

/** Legend item */
export interface LegendItem {
  /** Item key */
  key: string;
  /** Item label */
  label: string;
  /** Item color */
  color: string;
  /** Is item active/visible */
  active?: boolean;
}

/** Legend configuration */
export interface LegendConfiguration {
  /** Show legend */
  show: boolean;
  /** Legend position */
  position: LegendPosition;
  /** Legend items */
  items: LegendItem[];
  /** Interactive (click to toggle) */
  interactive?: boolean;
  /** Legend title */
  title?: string;
  /** Direction */
  direction?: 'row' | 'column';
  /** Item spacing */
  spacing?: number;
  /** Symbol shape */
  symbolShape?: 'circle' | 'square' | 'line' | 'diamond';
  /** Symbol size */
  symbolSize?: number;
}

// ============================================================================
// Tooltip Types
// ============================================================================

/** Tooltip position strategy */
export type TooltipPositionStrategy = 'cursor' | 'element' | 'fixed';

/** Tooltip configuration */
export interface TooltipConfiguration {
  /** Show tooltip */
  enabled: boolean;
  /** Position strategy */
  positionStrategy?: TooltipPositionStrategy;
  /** Offset from trigger point */
  offset?: { x: number; y: number };
  /** Tooltip content renderer */
  content?: (data: unknown) => React.ReactNode;
  /** Snap to data points */
  snapToData?: boolean;
  /** Show crosshairs */
  showCrosshairs?: boolean;
  /** Crosshair style */
  crosshairStyle?: {
    stroke?: string;
    strokeWidth?: number;
    strokeDasharray?: string;
  };
}

// ============================================================================
// Brush/Zoom Types
// ============================================================================

/** Brush extent */
export interface BrushExtent {
  /** X range */
  x: [number, number] | null;
  /** Y range */
  y: [number, number] | null;
}

/** Brush configuration */
export interface BrushConfiguration {
  /** Enable brush */
  enabled: boolean;
  /** Brush type */
  type: '1d-x' | '1d-y' | '2d';
  /** Initial extent */
  initialExtent?: BrushExtent;
  /** Brush change handler */
  onChange?: (extent: BrushExtent) => void;
  /** Brush end handler */
  onEnd?: (extent: BrushExtent) => void;
  /** Brush fill color */
  fill?: string;
  /** Brush stroke color */
  stroke?: string;
  /** Reset on double-click */
  resetOnDoubleClick?: boolean;
}

/** Zoom configuration */
export interface ZoomConfiguration {
  /** Enable zoom */
  enabled: boolean;
  /** Zoom type */
  type: 'scale' | 'translate' | 'both';
  /** Scale extent [min, max] */
  scaleExtent?: [number, number];
  /** Translate extent [[x0, y0], [x1, y1]] */
  translateExtent?: [[number, number], [number, number]];
  /** Initial zoom transform */
  initialTransform?: {
    k: number;
    x: number;
    y: number;
  };
  /** Zoom change handler */
  onChange?: (transform: { k: number; x: number; y: number }) => void;
  /** Enable wheel zoom */
  wheelZoom?: boolean;
  /** Enable double-click zoom */
  doubleClickZoom?: boolean;
  /** Enable drag pan */
  dragPan?: boolean;
}

// ============================================================================
// Animation Types
// ============================================================================

/** Animation configuration */
export interface AnimationConfiguration {
  /** Enable animations */
  enabled: boolean;
  /** Respect prefers-reduced-motion */
  respectReducedMotion: boolean;
  /** Enter animation duration (ms) */
  enterDuration?: number;
  /** Update animation duration (ms) */
  updateDuration?: number;
  /** Exit animation duration (ms) */
  exitDuration?: number;
  /** Easing function name */
  easing?: 'linear' | 'ease' | 'ease-in' | 'ease-out' | 'ease-in-out' | 'elastic' | 'bounce';
  /** Stagger delay between elements (ms) */
  staggerDelay?: number;
}

// ============================================================================
// Accessibility Types
// ============================================================================

/** Accessibility configuration for charts */
export interface ChartAccessibilityConfig {
  /** ARIA label for the chart */
  ariaLabel: string;
  /** ARIA description */
  ariaDescription?: string;
  /** Enable keyboard navigation */
  keyboardNavigation: boolean;
  /** Show data table alternative */
  showDataTable?: boolean;
  /** Announce data points on focus */
  announceDataPoints?: boolean;
  /** Pattern fills for color-blind support */
  usePatternFills?: boolean;
  /** High contrast mode */
  highContrastMode?: boolean;
  /** Data point label format */
  dataPointLabelFormat?: (data: unknown) => string;
}

// ============================================================================
// Responsive Types
// ============================================================================

/** Breakpoint configuration */
export interface ResponsiveBreakpoint {
  /** Breakpoint name */
  name: string;
  /** Min width (px) */
  minWidth: number;
  /** Max width (px) */
  maxWidth?: number;
}

/** Responsive chart configuration */
export interface ResponsiveConfiguration {
  /** Enable responsive behavior */
  enabled: boolean;
  /** Container query vs media query */
  strategy: 'container' | 'media';
  /** Breakpoint-specific overrides */
  breakpoints?: {
    /** Breakpoint name */
    name: string;
    /** Configuration overrides */
    config: Partial<{
      margin: { top: number; right: number; bottom: number; left: number };
      showLegend: boolean;
      showXAxis: boolean;
      showYAxis: boolean;
      numTicks: number;
      aspectRatio: number;
    }>;
  }[];
  /** Maintain aspect ratio */
  aspectRatio?: number;
  /** Minimum height */
  minHeight?: number;
  /** Maximum height */
  maxHeight?: number;
}

// ============================================================================
// Chart State Types
// ============================================================================

/** Visualization interaction state */
export interface VisualizationInteractionState {
  /** Hovered data point */
  hoveredData: unknown | null;
  /** Hovered index */
  hoveredIndex: number | null;
  /** Selected data points */
  selectedData: unknown[];
  /** Brush extent */
  brushExtent: BrushExtent | null;
  /** Zoom transform */
  zoomTransform: { k: number; x: number; y: number } | null;
  /** Active tooltip position */
  tooltipPosition: { x: number; y: number } | null;
  /** Focused series key */
  focusedSeries: string | null;
}

/** Chart rendering state */
export interface ChartRenderingState {
  /** Is chart mounted */
  isMounted: boolean;
  /** Is data loading */
  isLoading: boolean;
  /** Has error */
  hasError: boolean;
  /** Error message */
  errorMessage?: string;
  /** Container dimensions */
  dimensions: { width: number; height: number };
  /** Computed scales */
  scales: {
    x: unknown;
    y: unknown;
    color?: unknown;
    size?: unknown;
  } | null;
}

// ============================================================================
// Data Transformation Types
// ============================================================================

/** Aggregation method */
export type AggregationMethod = 'sum' | 'mean' | 'median' | 'min' | 'max' | 'count' | 'first' | 'last';

/** Time interval for resampling */
export type TimeInterval = 'second' | 'minute' | 'hour' | 'day' | 'week' | 'month' | 'quarter' | 'year';

/** Data transformation pipeline step */
export interface TransformationStep {
  /** Transformation type */
  type: 'filter' | 'aggregate' | 'sort' | 'limit' | 'calculate' | 'resample' | 'fill';
  /** Transformation parameters */
  params: Record<string, unknown>;
}

/** Aggregation configuration */
export interface AggregationConfig {
  /** Group by fields */
  groupBy: string[];
  /** Aggregations to compute */
  aggregations: Array<{
    field: string;
    method: AggregationMethod;
    alias?: string;
  }>;
}

/** Time series resampling configuration */
export interface ResampleConfig {
  /** Date field */
  dateField: string;
  /** Interval */
  interval: TimeInterval;
  /** Value field */
  valueField: string;
  /** Aggregation method */
  aggregation: AggregationMethod;
  /** Fill missing values */
  fillMissing?: 'zero' | 'previous' | 'next' | 'interpolate' | null;
}

/** Moving average configuration */
export interface MovingAverageConfig {
  /** Field to compute MA on */
  field: string;
  /** Window size */
  windowSize: number;
  /** Output field name */
  outputField: string;
  /** Center the window */
  centered?: boolean;
}

// ============================================================================
// Statistical Types
// ============================================================================

/** Regression result */
export interface RegressionResult {
  /** Slope */
  slope: number;
  /** Intercept */
  intercept: number;
  /** R-squared */
  rSquared: number;
  /** Standard error */
  standardError: number;
  /** Predicted values */
  predict: (x: number) => number;
}

/** Correlation result */
export interface CorrelationResult {
  /** Pearson correlation coefficient */
  pearson: number;
  /** Spearman correlation coefficient */
  spearman?: number;
  /** P-value */
  pValue: number;
  /** 95% confidence interval */
  confidenceInterval: [number, number];
}

/** Outlier detection result */
export interface OutlierResult<T> {
  /** Outlier data points */
  outliers: T[];
  /** Outlier indices */
  indices: number[];
  /** Detection method used */
  method: 'iqr' | 'zscore' | 'mad';
  /** Threshold used */
  threshold: number;
}

/** Statistical summary */
export interface StatisticalSummary {
  /** Count */
  count: number;
  /** Mean */
  mean: number;
  /** Median */
  median: number;
  /** Standard deviation */
  stdDev: number;
  /** Variance */
  variance: number;
  /** Minimum */
  min: number;
  /** Maximum */
  max: number;
  /** Quartiles */
  quartiles: {
    q1: number;
    q2: number;
    q3: number;
  };
  /** Percentiles */
  percentiles: Record<number, number>;
}

// ============================================================================
// Export Types
// ============================================================================

/** Chart export format */
export type ExportFormat = 'svg' | 'png' | 'csv' | 'json';

/** Export configuration */
export interface ExportConfiguration {
  /** File name (without extension) */
  filename: string;
  /** Format */
  format: ExportFormat;
  /** Scale factor for image exports */
  scale?: number;
  /** Background color for image exports */
  backgroundColor?: string;
  /** Include legend */
  includeLegend?: boolean;
  /** Include title */
  includeTitle?: boolean;
  /** CSV delimiter */
  csvDelimiter?: string;
}
