/**
 * Component Type Definitions
 * 
 * Purpose: Complete TypeScript interfaces for all UI, 3D, and visualization components
 * with JSDoc documentation for each prop.
 * 
 * Organization follows atomic design:
 * - Atoms: Basic building blocks (Button, Input, etc.)
 * - Molecules: Composed elements (FormField, Card, etc.)
 * - Organisms: Complex features (DataTable, Header, etc.)
 * - 3D Components: Three.js/R3F components
 * - Visualizations: D3.js/Visx chart components
 */

import type { ReactNode, ComponentPropsWithoutRef, HTMLAttributes } from 'react';
import type { Object3D, Vector3Tuple } from 'three';

// ============================================================================
// Shared Types
// ============================================================================

/** Common size variants */
export type Size = 'sm' | 'md' | 'lg';

/** Extended size variants including extra small and extra large */
export type ExtendedSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';

/** Common variant types for UI feedback */
export type Variant = 'default' | 'primary' | 'secondary' | 'outline' | 'ghost' | 'destructive';

/** Status variants for alerts and badges */
export type StatusVariant = 'info' | 'success' | 'warning' | 'error';

/** Loading state for components */
export interface LoadingState {
  /** Whether the component is in loading state */
  isLoading: boolean;
  /** Optional loading progress (0-100) */
  progress?: number;
  /** Loading message to display */
  message?: string;
}

/** Error state for components */
export interface ErrorState {
  /** Whether there's an error */
  hasError: boolean;
  /** Error message */
  message?: string;
  /** Retry handler */
  onRetry?: () => void;
}

// ============================================================================
// Atom Components
// ============================================================================

/** Button component props */
export interface ButtonProps extends ComponentPropsWithoutRef<'button'> {
  /** Visual variant of the button */
  variant?: Variant;
  /** Size of the button */
  size?: Size;
  /** Show loading spinner and disable interactions */
  isLoading?: boolean;
  /** Icon to render on the left side */
  leftIcon?: ReactNode;
  /** Icon to render on the right side */
  rightIcon?: ReactNode;
  /** Render only an icon (no text) */
  iconOnly?: boolean;
  /** Full width button */
  fullWidth?: boolean;
}

/** Input component props */
export interface InputProps extends Omit<ComponentPropsWithoutRef<'input'>, 'size'> {
  /** Error message to display */
  error?: string;
  /** Success state */
  success?: boolean;
  /** Left addon (icon or text) */
  leftAddon?: ReactNode;
  /** Right addon (icon or text) */
  rightAddon?: ReactNode;
  /** Size variant */
  inputSize?: Size;
}

/** Select option type */
export interface SelectOption<T = string> {
  label: string;
  value: T;
  disabled?: boolean;
  description?: string;
}

/** Select component props */
export interface SelectProps<T = string> {
  /** Array of options */
  options: SelectOption<T>[];
  /** Currently selected value(s) */
  value?: T | T[];
  /** Change handler */
  onChange?: (value: T | T[]) => void;
  /** Allow multiple selections */
  multiple?: boolean;
  /** Enable search filtering */
  searchable?: boolean;
  /** Allow creating new options */
  creatable?: boolean;
  /** Async option loading */
  loadOptions?: (query: string) => Promise<SelectOption<T>[]>;
  /** Placeholder text */
  placeholder?: string;
  /** Disabled state */
  disabled?: boolean;
  /** Error message */
  error?: string;
}

/** Checkbox props */
export interface CheckboxProps extends Omit<ComponentPropsWithoutRef<'button'>, 'onChange'> {
  /** Checked state */
  checked?: boolean;
  /** Indeterminate state */
  indeterminate?: boolean;
  /** Change handler */
  onCheckedChange?: (checked: boolean) => void;
  /** Label text */
  label?: string;
  /** Description text */
  description?: string;
}

/** Radio group props */
export interface RadioGroupProps<T = string> {
  /** Radio options */
  options: SelectOption<T>[];
  /** Selected value */
  value?: T;
  /** Change handler */
  onChange?: (value: T) => void;
  /** Orientation */
  orientation?: 'horizontal' | 'vertical';
  /** Disabled state */
  disabled?: boolean;
}

/** Switch props */
export interface SwitchProps extends Omit<ComponentPropsWithoutRef<'button'>, 'onChange'> {
  /** Checked state */
  checked?: boolean;
  /** Change handler */
  onCheckedChange?: (checked: boolean) => void;
  /** Label text */
  label?: string;
  /** Size variant */
  size?: Size;
}

/** Badge props */
export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  /** Visual variant */
  variant?: StatusVariant | 'default';
  /** Size */
  size?: Size;
  /** Dot indicator style */
  dot?: boolean;
}

/** Avatar props */
export interface AvatarProps {
  /** Image source URL */
  src?: string;
  /** Alt text */
  alt?: string;
  /** Fallback initials or content */
  fallback?: ReactNode;
  /** Size */
  size?: ExtendedSize;
  /** Status indicator */
  status?: 'online' | 'offline' | 'busy' | 'away';
}

/** Tooltip props */
export interface TooltipProps {
  /** Tooltip content */
  content: ReactNode;
  /** Trigger element */
  children: ReactNode;
  /** Placement */
  side?: 'top' | 'right' | 'bottom' | 'left';
  /** Alignment */
  align?: 'start' | 'center' | 'end';
  /** Delay before showing (ms) */
  delayDuration?: number;
}

/** Skeleton loader props */
export interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  /** Width */
  width?: string | number;
  /** Height */
  height?: string | number;
  /** Circular shape */
  circle?: boolean;
  /** Number of lines (for text skeletons) */
  lines?: number;
}

// ============================================================================
// Molecule Components
// ============================================================================

/** Form field props (label + input + error composition) */
export interface FormFieldProps {
  /** Field label */
  label: string;
  /** Field name for form state */
  name: string;
  /** Helper/description text */
  description?: string;
  /** Error message */
  error?: string;
  /** Required indicator */
  required?: boolean;
  /** Form input element */
  children: ReactNode;
}

/** Card component props */
export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Card header content */
  header?: ReactNode;
  /** Card footer content */
  footer?: ReactNode;
  /** Padding size */
  padding?: Size;
  /** Hover effect */
  hoverable?: boolean;
  /** Selected state */
  selected?: boolean;
  /** Click handler */
  onClick?: () => void;
}

/** Alert props */
export interface AlertProps extends HTMLAttributes<HTMLDivElement> {
  /** Alert variant */
  variant?: StatusVariant;
  /** Alert title */
  title?: string;
  /** Dismissible */
  dismissible?: boolean;
  /** Dismiss handler */
  onDismiss?: () => void;
  /** Icon override */
  icon?: ReactNode;
}

/** Toast notification data */
export interface ToastData {
  id: string;
  title: string;
  description?: string;
  variant?: StatusVariant;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

/** Modal/Dialog props */
export interface ModalProps {
  /** Open state */
  open: boolean;
  /** Close handler */
  onClose: () => void;
  /** Modal title */
  title?: string;
  /** Modal description */
  description?: string;
  /** Modal content */
  children: ReactNode;
  /** Modal size */
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  /** Close on overlay click */
  closeOnOverlayClick?: boolean;
  /** Close on escape key */
  closeOnEscape?: boolean;
}

/** Tabs component props */
export interface TabsProps<T extends string = string> {
  /** Tab items */
  items: Array<{
    id: T;
    label: string;
    icon?: ReactNode;
    disabled?: boolean;
    content: ReactNode;
  }>;
  /** Active tab */
  activeTab?: T;
  /** Tab change handler */
  onTabChange?: (tab: T) => void;
  /** Visual variant */
  variant?: 'default' | 'pills' | 'underline';
  /** Orientation */
  orientation?: 'horizontal' | 'vertical';
}

/** Accordion item */
export interface AccordionItem {
  id: string;
  title: ReactNode;
  content: ReactNode;
  disabled?: boolean;
}

/** Accordion props */
export interface AccordionProps {
  /** Accordion items */
  items: AccordionItem[];
  /** Allow multiple open */
  multiple?: boolean;
  /** Default open items */
  defaultOpen?: string[];
  /** Controlled open state */
  openItems?: string[];
  /** Change handler */
  onOpenChange?: (openItems: string[]) => void;
}

/** Pagination props */
export interface PaginationProps {
  /** Current page (1-indexed) */
  currentPage: number;
  /** Total pages */
  totalPages: number;
  /** Page change handler */
  onPageChange: (page: number) => void;
  /** Siblings to show on each side */
  siblings?: number;
  /** Show first/last buttons */
  showEdges?: boolean;
  /** Items per page options */
  pageSizeOptions?: number[];
  /** Current page size */
  pageSize?: number;
  /** Page size change handler */
  onPageSizeChange?: (size: number) => void;
}

/** Search input props */
export interface SearchInputProps extends Omit<InputProps, 'onChange'> {
  /** Search query */
  value: string;
  /** Change handler */
  onChange: (value: string) => void;
  /** Debounce delay (ms) */
  debounceMs?: number;
  /** Clear button */
  clearable?: boolean;
}

/** Date picker props */
export interface DatePickerProps {
  /** Selected date */
  value?: Date;
  /** Change handler */
  onChange?: (date: Date | undefined) => void;
  /** Minimum selectable date */
  minDate?: Date;
  /** Maximum selectable date */
  maxDate?: Date;
  /** Date format */
  format?: string;
  /** Placeholder text */
  placeholder?: string;
  /** Disabled state */
  disabled?: boolean;
  /** Error message */
  error?: string;
}

/** File upload props */
export interface FileUploadProps {
  /** Accepted MIME types */
  accept?: string[];
  /** Allow multiple files */
  multiple?: boolean;
  /** Maximum file size (bytes) */
  maxSize?: number;
  /** Maximum number of files */
  maxFiles?: number;
  /** Uploaded files */
  files?: File[];
  /** Change handler */
  onChange?: (files: File[]) => void;
  /** Upload progress (0-100) */
  progress?: number;
  /** Drag and drop enabled */
  dragAndDrop?: boolean;
  /** Error message */
  error?: string;
}

// ============================================================================
// Organism Components
// ============================================================================

/** Data table column definition */
export interface DataTableColumn<T> {
  /** Unique column key */
  key: string;
  /** Column header */
  header: ReactNode;
  /** Cell renderer */
  cell: (row: T, index: number) => ReactNode;
  /** Sortable */
  sortable?: boolean;
  /** Filterable */
  filterable?: boolean;
  /** Column width */
  width?: string | number;
  /** Visibility */
  visible?: boolean;
  /** Alignment */
  align?: 'left' | 'center' | 'right';
}

/** Data table props */
export interface DataTableProps<T> {
  /** Table data */
  data: T[];
  /** Column definitions */
  columns: DataTableColumn<T>[];
  /** Unique row key getter */
  getRowKey: (row: T) => string;
  /** Row selection */
  selectable?: boolean;
  /** Selected row keys */
  selectedKeys?: string[];
  /** Selection change handler */
  onSelectionChange?: (keys: string[]) => void;
  /** Sorting state */
  sorting?: { key: string; direction: 'asc' | 'desc' } | null;
  /** Sort change handler */
  onSortChange?: (sorting: { key: string; direction: 'asc' | 'desc' } | null) => void;
  /** Loading state */
  isLoading?: boolean;
  /** Empty state message */
  emptyMessage?: ReactNode;
  /** Pagination */
  pagination?: PaginationProps;
  /** Row click handler */
  onRowClick?: (row: T) => void;
  /** Sticky header */
  stickyHeader?: boolean;
  /** Row actions */
  rowActions?: (row: T) => ReactNode;
}

/** Navigation item */
export interface NavItem {
  id: string;
  label: string;
  href?: string;
  icon?: ReactNode;
  badge?: ReactNode;
  children?: NavItem[];
  onClick?: () => void;
}

/** Header props */
export interface HeaderProps {
  /** Logo/brand element */
  logo?: ReactNode;
  /** Navigation items */
  navItems?: NavItem[];
  /** User menu */
  user?: {
    name: string;
    email: string;
    avatar?: string;
    menuItems?: Array<{ label: string; onClick: () => void; icon?: ReactNode }>;
  };
  /** Notification bell */
  notifications?: {
    count: number;
    onClick: () => void;
  };
  /** Theme toggle */
  showThemeToggle?: boolean;
}

/** Sidebar props */
export interface SidebarProps {
  /** Navigation items */
  items: NavItem[];
  /** Collapsed state */
  collapsed?: boolean;
  /** Collapse toggle handler */
  onToggleCollapse?: () => void;
  /** Collapsible */
  collapsible?: boolean;
  /** Footer content */
  footer?: ReactNode;
  /** Active item ID */
  activeId?: string;
}

/** Command palette props */
export interface CommandPaletteProps {
  /** Open state */
  open: boolean;
  /** Close handler */
  onClose: () => void;
  /** Commands */
  commands: Array<{
    id: string;
    label: string;
    description?: string;
    icon?: ReactNode;
    shortcut?: string;
    onSelect: () => void;
    group?: string;
  }>;
  /** Placeholder text */
  placeholder?: string;
}

/** Filter panel props */
export interface FilterPanelProps<T extends Record<string, unknown>> {
  /** Filter schema */
  filters: Array<{
    key: keyof T;
    label: string;
    type: 'text' | 'select' | 'date' | 'range' | 'boolean';
    options?: SelectOption[];
  }>;
  /** Current filter values */
  values: Partial<T>;
  /** Change handler */
  onChange: (values: Partial<T>) => void;
  /** Clear all filters */
  onClear: () => void;
}

// ============================================================================
// 3D Component Types
// ============================================================================

/** Scene component props */
export interface SceneProps {
  /** Camera configuration */
  camera?: {
    position?: Vector3Tuple;
    fov?: number;
    near?: number;
    far?: number;
  };
  /** Ambient light intensity */
  ambientLight?: number;
  /** Environment map */
  environment?: 'sunset' | 'dawn' | 'night' | 'warehouse' | 'forest' | 'apartment' | 'studio' | 'city' | 'park' | 'lobby';
  /** Background color */
  backgroundColor?: string;
  /** Enable shadows */
  shadows?: boolean;
  /** Post-processing effects */
  effects?: {
    bloom?: { intensity?: number; threshold?: number };
    ssao?: { intensity?: number; radius?: number };
    dof?: { focusDistance?: number; focalLength?: number; bokehScale?: number };
  };
  /** Enable controls */
  controls?: 'orbit' | 'fly' | 'pointer-lock' | 'none';
  /** Scene content */
  children: ReactNode;
  /** Performance stats */
  showStats?: boolean;
}

/** 3D Model component props */
export interface ModelProps {
  /** Path to GLTF/GLB model */
  url: string;
  /** Model position */
  position?: Vector3Tuple;
  /** Model rotation (Euler) */
  rotation?: Vector3Tuple;
  /** Model scale */
  scale?: number | Vector3Tuple;
  /** Play animation by name */
  animation?: string;
  /** Animation loop */
  loop?: boolean;
  /** Animation time scale */
  timeScale?: number;
  /** Enable shadows (cast/receive) */
  shadows?: boolean;
  /** Click handler */
  onClick?: (event: ThreeEvent) => void;
  /** Hover handlers */
  onPointerOver?: (event: ThreeEvent) => void;
  onPointerOut?: (event: ThreeEvent) => void;
  /** Material overrides */
  materials?: Record<string, { color?: string; metalness?: number; roughness?: number }>;
  /** Visibility */
  visible?: boolean;
}

/** Three.js event type (simplified) */
export interface ThreeEvent {
  object: Object3D;
  point: { x: number; y: number; z: number };
  distance: number;
  stopPropagation: () => void;
}

/** Controls component props */
export interface ControlsProps {
  /** Control type */
  type: 'orbit' | 'fly' | 'pointer-lock';
  /** Enable damping */
  enableDamping?: boolean;
  /** Damping factor */
  dampingFactor?: number;
  /** Min/max distance for orbit controls */
  minDistance?: number;
  maxDistance?: number;
  /** Min/max polar angle */
  minPolarAngle?: number;
  maxPolarAngle?: number;
  /** Enable pan */
  enablePan?: boolean;
  /** Enable zoom */
  enableZoom?: boolean;
  /** Enable rotate */
  enableRotate?: boolean;
  /** Target point to orbit around */
  target?: Vector3Tuple;
}

/** HTML overlay in 3D space */
export interface OverlayProps {
  /** Position in 3D space */
  position: Vector3Tuple;
  /** HTML content */
  children: ReactNode;
  /** Distance-based scaling */
  distanceFactor?: number;
  /** Occlusion (hide behind objects) */
  occlude?: boolean | Object3D[];
  /** Z-index offset */
  zIndexRange?: [number, number];
}

// ============================================================================
// Visualization Component Types
// ============================================================================

/** Base chart configuration */
export interface ChartConfig {
  /** Chart width (responsive if not set) */
  width?: number;
  /** Chart height */
  height?: number;
  /** Chart margins */
  margin?: { top: number; right: number; bottom: number; left: number };
  /** Color palette */
  colors?: string[];
  /** Animation duration (ms) */
  animationDuration?: number;
  /** Enable animations */
  animate?: boolean;
  /** Respect prefers-reduced-motion */
  respectReducedMotion?: boolean;
}

/** Axis configuration */
export interface AxisConfig {
  /** Axis label */
  label?: string;
  /** Tick format function */
  tickFormat?: (value: unknown) => string;
  /** Number of ticks */
  tickCount?: number;
  /** Show grid lines */
  showGrid?: boolean;
  /** Axis scale type */
  scaleType?: 'linear' | 'log' | 'time' | 'band' | 'point';
}

/** Tooltip data for charts */
export interface ChartTooltipData<T = unknown> {
  /** X coordinate */
  x: number;
  /** Y coordinate */
  y: number;
  /** Data associated with tooltip */
  data: T;
  /** Formatted label */
  label: string;
  /** Formatted value */
  value: string;
  /** Series color */
  color?: string;
}

/** Base visualization props */
export interface BaseVisualizationProps<T> extends ChartConfig {
  /** Chart data */
  data: T[];
  /** X-axis accessor */
  xAccessor: (d: T) => unknown;
  /** Y-axis accessor */
  yAccessor: (d: T) => number;
  /** X-axis configuration */
  xAxis?: AxisConfig;
  /** Y-axis configuration */
  yAxis?: AxisConfig;
  /** Tooltip configuration */
  tooltip?: {
    enabled?: boolean;
    format?: (d: T) => string;
  };
  /** Legend configuration */
  legend?: {
    enabled?: boolean;
    position?: 'top' | 'right' | 'bottom' | 'left';
  };
  /** Loading state */
  isLoading?: boolean;
  /** Error state */
  error?: string;
  /** Empty state message */
  emptyMessage?: string;
  /** Accessibility label */
  ariaLabel: string;
  /** Show as data table (accessibility) */
  showDataTable?: boolean;
}

/** Line chart specific props */
export interface LineChartProps<T> extends BaseVisualizationProps<T> {
  /** Series configuration for multi-line charts */
  series?: Array<{
    key: string;
    yAccessor: (d: T) => number;
    color?: string;
    label?: string;
    strokeWidth?: number;
    strokeDasharray?: string;
  }>;
  /** Show area fill */
  showArea?: boolean;
  /** Show data points */
  showPoints?: boolean;
  /** Point radius */
  pointRadius?: number;
  /** Curve type */
  curve?: 'linear' | 'monotone' | 'step' | 'basis' | 'cardinal';
  /** Enable brush selection */
  enableBrush?: boolean;
  /** Brush change handler */
  onBrushChange?: (range: [Date, Date] | [number, number] | null) => void;
}

/** Bar chart specific props */
export interface BarChartProps<T> extends BaseVisualizationProps<T> {
  /** Orientation */
  orientation?: 'vertical' | 'horizontal';
  /** Bar padding (0-1) */
  barPadding?: number;
  /** Group bars for multi-series */
  groupMode?: 'grouped' | 'stacked';
  /** Series for multi-bar charts */
  series?: Array<{
    key: string;
    yAccessor: (d: T) => number;
    color?: string;
    label?: string;
  }>;
  /** Show value labels */
  showValues?: boolean;
  /** Bar click handler */
  onBarClick?: (d: T) => void;
}

/** Scatter plot specific props */
export interface ScatterPlotProps<T> extends BaseVisualizationProps<T> {
  /** Point size accessor */
  sizeAccessor?: (d: T) => number;
  /** Color accessor */
  colorAccessor?: (d: T) => string;
  /** Min/max point size */
  sizeRange?: [number, number];
  /** Enable brush selection */
  enableBrush?: boolean;
  /** Selection handler */
  onSelection?: (selected: T[]) => void;
  /** Regression line */
  showRegression?: boolean;
}

/** Pie/Donut chart props */
export interface PieChartProps<T> {
  /** Chart data */
  data: T[];
  /** Value accessor */
  valueAccessor: (d: T) => number;
  /** Label accessor */
  labelAccessor: (d: T) => string;
  /** Chart dimensions */
  width?: number;
  height?: number;
  /** Inner radius (0 for pie, >0 for donut) */
  innerRadius?: number;
  /** Color palette */
  colors?: string[];
  /** Show labels */
  showLabels?: boolean;
  /** Label type */
  labelType?: 'percent' | 'value' | 'label';
  /** Sort by value */
  sortByValue?: boolean;
  /** Click handler */
  onSliceClick?: (d: T) => void;
  /** Accessibility label */
  ariaLabel: string;
}

/** Heatmap props */
export interface HeatmapProps<T> {
  /** Chart data */
  data: T[];
  /** X accessor */
  xAccessor: (d: T) => string;
  /** Y accessor */
  yAccessor: (d: T) => string;
  /** Value accessor */
  valueAccessor: (d: T) => number;
  /** Chart dimensions */
  width?: number;
  height?: number;
  /** Color scale */
  colorScale?: 'sequential' | 'diverging';
  /** Color range */
  colorRange?: [string, string] | [string, string, string];
  /** Cell click handler */
  onCellClick?: (d: T) => void;
  /** Accessibility label */
  ariaLabel: string;
}

/** Network graph props */
export interface NetworkGraphProps<N, L> {
  /** Node data */
  nodes: N[];
  /** Link data */
  links: L[];
  /** Node ID accessor */
  nodeId: (d: N) => string;
  /** Link source accessor */
  linkSource: (d: L) => string;
  /** Link target accessor */
  linkTarget: (d: L) => string;
  /** Node label accessor */
  nodeLabel?: (d: N) => string;
  /** Node size accessor */
  nodeSize?: (d: N) => number;
  /** Node color accessor */
  nodeColor?: (d: N) => string;
  /** Link width accessor */
  linkWidth?: (d: L) => number;
  /** Chart dimensions */
  width?: number;
  height?: number;
  /** Enable zoom/pan */
  enableZoom?: boolean;
  /** Node click handler */
  onNodeClick?: (d: N) => void;
  /** Accessibility label */
  ariaLabel: string;
}

/** KPI card props */
export interface KPICardProps {
  /** Card title */
  title: string;
  /** Main value */
  value: number | string;
  /** Value format */
  format?: 'number' | 'currency' | 'percent';
  /** Change indicator */
  change?: {
    value: number;
    direction: 'up' | 'down' | 'neutral';
    period?: string;
  };
  /** Sparkline data */
  sparkline?: number[];
  /** Icon */
  icon?: ReactNode;
  /** Loading state */
  isLoading?: boolean;
  /** Click handler for drill-down */
  onClick?: () => void;
}

/** Geographic map props */
export interface GeoMapProps<T> {
  /** Geographic data (GeoJSON) */
  geoData: GeoJSON.FeatureCollection;
  /** Data points */
  data: T[];
  /** Location accessor (returns [lng, lat]) */
  locationAccessor: (d: T) => [number, number];
  /** Value accessor for coloring */
  valueAccessor?: (d: T) => number;
  /** Map center [lng, lat] */
  center?: [number, number];
  /** Zoom level */
  zoom?: number;
  /** Chart dimensions */
  width?: number;
  height?: number;
  /** Color scale */
  colorScale?: string[];
  /** Point click handler */
  onPointClick?: (d: T) => void;
  /** Accessibility label */
  ariaLabel: string;
}
