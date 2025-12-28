/**
 * Three.js Scene Type Definitions
 * 
 * Purpose: Type definitions for 3D scenes, configurations, and assets
 * Used by React Three Fiber components throughout the application
 * 
 * Design decisions:
 * - Strict typing for all Three.js-related props
 * - Scene configuration schemas for different use cases
 * - Performance budget types for monitoring
 */

import type { Vector3Tuple } from 'three';

// ============================================================================
// Scene Configuration Types
// ============================================================================

/** Scene purpose classification */
export type ScenePurpose =
  | 'product-showcase'    // Static product visualization
  | 'data-visualization'  // 3D data representation
  | 'interactive'         // User-manipulable experience
  | 'background'          // Ambient/decorative
  | 'configurator';       // Product customization

/** Interaction model for scenes */
export type InteractionModel =
  | 'view-only'           // No interaction
  | 'rotate-zoom'         // Orbit controls
  | 'manipulate'          // Object manipulation
  | 'navigate'            // First-person/fly navigation
  | 'physics';            // Physics-based interaction

/** Scene configuration schema */
export interface SceneConfig {
  /** Unique scene identifier */
  id: string;
  /** Scene name for display */
  name: string;
  /** Scene purpose category */
  purpose: ScenePurpose;
  /** Interaction model */
  interaction: InteractionModel;
  /** Camera configuration */
  camera: CameraConfig;
  /** Lighting configuration */
  lighting: LightingConfig;
  /** Environment configuration */
  environment: EnvironmentConfig;
  /** Post-processing effects */
  effects?: EffectsConfig;
  /** Performance targets */
  performance: PerformanceTargets;
  /** Accessibility settings */
  accessibility: AccessibilityConfig;
  /** Assets to load */
  assets: AssetManifestEntry[];
}

/** Camera configuration */
export interface CameraConfig {
  /** Camera type */
  type: 'perspective' | 'orthographic';
  /** Initial position */
  position: Vector3Tuple;
  /** Look-at target */
  target: Vector3Tuple;
  /** Field of view (perspective only) */
  fov?: number;
  /** Near clipping plane */
  near: number;
  /** Far clipping plane */
  far: number;
  /** Orthographic frustum size */
  frustumSize?: number;
}

/** Lighting configuration */
export interface LightingConfig {
  /** Ambient light settings */
  ambient?: {
    color: string;
    intensity: number;
  };
  /** Directional lights */
  directional?: Array<{
    position: Vector3Tuple;
    color: string;
    intensity: number;
    castShadow?: boolean;
    shadowMapSize?: number;
  }>;
  /** Point lights */
  point?: Array<{
    position: Vector3Tuple;
    color: string;
    intensity: number;
    distance?: number;
    decay?: number;
  }>;
  /** Spot lights */
  spot?: Array<{
    position: Vector3Tuple;
    target: Vector3Tuple;
    color: string;
    intensity: number;
    angle?: number;
    penumbra?: number;
    castShadow?: boolean;
  }>;
  /** Use environment map for lighting */
  useEnvironment?: boolean;
}

/** Environment configuration */
export interface EnvironmentConfig {
  /** Environment preset */
  preset?: 'sunset' | 'dawn' | 'night' | 'warehouse' | 'forest' | 'apartment' | 'studio' | 'city' | 'park' | 'lobby';
  /** Custom HDR environment map URL */
  hdrUrl?: string;
  /** Background type */
  background?: 'environment' | 'color' | 'transparent';
  /** Background color (if type is 'color') */
  backgroundColor?: string;
  /** Ground plane */
  ground?: {
    enabled: boolean;
    height?: number;
    radius?: number;
    resolution?: number;
  };
}

/** Post-processing effects configuration */
export interface EffectsConfig {
  /** Bloom effect */
  bloom?: {
    enabled: boolean;
    intensity: number;
    threshold: number;
    radius: number;
  };
  /** Screen Space Ambient Occlusion */
  ssao?: {
    enabled: boolean;
    samples: number;
    radius: number;
    intensity: number;
  };
  /** Depth of Field */
  depthOfField?: {
    enabled: boolean;
    focusDistance: number;
    focalLength: number;
    bokehScale: number;
  };
  /** Vignette */
  vignette?: {
    enabled: boolean;
    offset: number;
    darkness: number;
  };
  /** Color grading (LUT) */
  colorGrading?: {
    enabled: boolean;
    lutUrl: string;
  };
  /** Anti-aliasing */
  antiAliasing?: 'fxaa' | 'smaa' | 'msaa' | 'none';
}

/** Performance targets for scenes */
export interface PerformanceTargets {
  /** Target frames per second */
  targetFPS: {
    desktop: number;
    mobile: number;
  };
  /** Maximum draw calls */
  maxDrawCalls: number;
  /** Maximum triangles */
  maxTriangles: number;
  /** Maximum texture memory (MB) */
  maxTextureMemory: number;
  /** WebGL version requirement */
  webglVersion: 1 | 2;
  /** Fallback behavior if performance target not met */
  fallback: 'lower-quality' | 'static-image' | 'hide';
}

/** Accessibility configuration for 3D content */
export interface AccessibilityConfig {
  /** Scene description for screen readers */
  description: string;
  /** Alternative 2D representation available */
  has2DFallback: boolean;
  /** Keyboard navigation enabled */
  keyboardNav: boolean;
  /** Reduced motion mode behavior */
  reducedMotion: 'disable-animations' | 'reduce-speed' | 'ignore';
  /** Focus indicators visible */
  focusIndicators: boolean;
  /** Audio descriptions available */
  audioDescriptions?: boolean;
}

// ============================================================================
// Asset Management Types
// ============================================================================

/** Asset types supported */
export type AssetType = 
  | 'gltf'         // GLTF/GLB models
  | 'texture'      // Texture images
  | 'environment'  // HDR environment maps
  | 'audio'        // Audio files
  | 'animation';   // Separate animation files

/** Asset manifest entry */
export interface AssetManifestEntry {
  /** Unique asset identifier */
  id: string;
  /** Asset type */
  type: AssetType;
  /** Asset URL/path */
  url: string;
  /** Preload priority (lower = higher priority) */
  priority: number;
  /** Asset size hint (bytes) */
  sizeHint?: number;
  /** Compressed format available */
  compressed?: {
    draco?: boolean;
    ktx2?: boolean;
    basis?: boolean;
  };
  /** LOD variants */
  lod?: Array<{
    level: number;
    url: string;
    distanceThreshold: number;
  }>;
  /** Fallback asset if loading fails */
  fallbackUrl?: string;
}

/** Asset loading state */
export interface AssetLoadingState {
  /** Asset ID */
  id: string;
  /** Loading status */
  status: 'pending' | 'loading' | 'loaded' | 'error';
  /** Loading progress (0-1) */
  progress: number;
  /** Error message if failed */
  error?: string;
}

/** Asset manifest for a scene */
export interface AssetManifest {
  /** Scene ID this manifest belongs to */
  sceneId: string;
  /** Version for cache invalidation */
  version: string;
  /** Last updated timestamp */
  updatedAt: string;
  /** Assets in this manifest */
  assets: AssetManifestEntry[];
  /** Total size of all assets (bytes) */
  totalSize: number;
}

// ============================================================================
// Animation Types
// ============================================================================

/** Animation clip metadata */
export interface AnimationClipMeta {
  /** Animation name */
  name: string;
  /** Duration in seconds */
  duration: number;
  /** Whether it should loop */
  loop: boolean;
  /** Loop mode */
  loopMode?: 'repeat' | 'pingpong' | 'once';
  /** Blend weight */
  weight?: number;
  /** Playback speed */
  timeScale?: number;
}

/** Animation state */
export interface AnimationState {
  /** Currently playing animation */
  currentAnimation?: string;
  /** Is playing */
  isPlaying: boolean;
  /** Current time in animation */
  currentTime: number;
  /** Playback progress (0-1) */
  progress: number;
  /** Available animations */
  availableAnimations: AnimationClipMeta[];
}

// ============================================================================
// Interaction Types
// ============================================================================

/** Object selection state */
export interface SelectionState {
  /** Selected object IDs */
  selectedIds: string[];
  /** Hovered object ID */
  hoveredId?: string;
  /** Selection mode */
  mode: 'single' | 'multi';
  /** Transform mode for selected objects */
  transformMode?: 'translate' | 'rotate' | 'scale';
}

/** Camera control state */
export interface CameraControlState {
  /** Current camera position */
  position: Vector3Tuple;
  /** Current camera target */
  target: Vector3Tuple;
  /** Current zoom level */
  zoom: number;
  /** Is camera animating */
  isAnimating: boolean;
  /** Camera preset name if using one */
  preset?: string;
}

/** Object transform */
export interface ObjectTransform {
  /** Position */
  position: Vector3Tuple;
  /** Rotation (euler angles in radians) */
  rotation: Vector3Tuple;
  /** Scale */
  scale: Vector3Tuple;
}

// ============================================================================
// Performance Monitoring Types
// ============================================================================

/** Performance metrics snapshot */
export interface PerformanceMetrics {
  /** Current FPS */
  fps: number;
  /** Frame time in ms */
  frameTime: number;
  /** Number of draw calls */
  drawCalls: number;
  /** Number of triangles rendered */
  triangles: number;
  /** Number of points rendered */
  points: number;
  /** Number of lines rendered */
  lines: number;
  /** Texture memory usage (bytes) */
  textureMemory: number;
  /** Geometry count */
  geometries: number;
  /** Texture count */
  textures: number;
  /** Program (shader) count */
  programs: number;
  /** Timestamp */
  timestamp: number;
}

/** Performance budget violation */
export interface PerformanceBudgetViolation {
  /** Metric that violated budget */
  metric: keyof PerformanceMetrics;
  /** Current value */
  current: number;
  /** Budget limit */
  limit: number;
  /** Severity */
  severity: 'warning' | 'critical';
  /** Suggested action */
  suggestion: string;
}

// ============================================================================
// Scene State Types
// ============================================================================

/** Complete scene state (for Zustand store) */
export interface SceneState {
  /** Scene configuration */
  config: SceneConfig | null;
  /** Asset loading states */
  assetStates: Record<string, AssetLoadingState>;
  /** Overall loading progress (0-1) */
  loadingProgress: number;
  /** Scene is ready for interaction */
  isReady: boolean;
  /** Current error if any */
  error?: string;
  /** Animation state */
  animation: AnimationState;
  /** Selection state */
  selection: SelectionState;
  /** Camera control state */
  camera: CameraControlState;
  /** Performance metrics */
  performance: PerformanceMetrics | null;
  /** Performance violations */
  violations: PerformanceBudgetViolation[];
}

// ============================================================================
// Scene Actions Types
// ============================================================================

/** Scene store actions */
export interface SceneActions {
  /** Load a scene configuration */
  loadScene: (config: SceneConfig) => Promise<void>;
  /** Unload current scene */
  unloadScene: () => void;
  /** Update asset loading state */
  updateAssetState: (id: string, state: Partial<AssetLoadingState>) => void;
  /** Play animation */
  playAnimation: (name: string, options?: Partial<AnimationClipMeta>) => void;
  /** Stop animation */
  stopAnimation: () => void;
  /** Select objects */
  select: (ids: string[]) => void;
  /** Clear selection */
  clearSelection: () => void;
  /** Set transform mode */
  setTransformMode: (mode: 'translate' | 'rotate' | 'scale' | undefined) => void;
  /** Update camera */
  updateCamera: (state: Partial<CameraControlState>) => void;
  /** Animate camera to position */
  animateCameraTo: (position: Vector3Tuple, target: Vector3Tuple, duration?: number) => void;
  /** Update performance metrics */
  updatePerformance: (metrics: PerformanceMetrics) => void;
  /** Reset scene state */
  reset: () => void;
}
