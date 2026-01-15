/**
 * 3D Embedding Space Visualization
 *
 * Purpose: Interactive 3D visualization of preference embeddings
 * using React Three Fiber for exploring annotation clusters.
 *
 * Features:
 * - Instanced mesh rendering for 50K+ points
 * - Color-coded attributes (confidence, agreement, recency)
 * - Orbit controls for camera manipulation
 * - Lasso selection for bulk operations
 * - Level-of-detail for performance
 */

import * as React from 'react';
import { Canvas, type ThreeEvent, useThree } from '@react-three/fiber';
import {
  OrbitControls,
  PerspectiveCamera,
  Html,
  Bounds,
} from '@react-three/drei';
import * as THREE from 'three';
import { BarChart2 } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { isWebGLSupported } from '@/lib/three/canvas-config';
import {
  A11yCanvas,
  A11yProvider,
  KeyboardControls,
  SceneFallback,
  useA11y,
} from '@/lib/three/a11y/A11yScene';

const MAX_A11Y_POINTS = 200;

// Types
interface EmbeddingPoint {
  id: string;
  position: [number, number, number];
  confidence: number;
  agreement: number;
  recency: number;
  taskType: 'chosen' | 'rejected' | 'tie';
  taskId: string;
  prompt?: string;
}

interface EmbeddingSpaceProps {
  onPointClick?: (point: EmbeddingPoint) => void;
  onSelectionChange?: (points: EmbeddingPoint[]) => void;
  maxPoints?: number;
  colorBy?: 'confidence' | 'agreement' | 'recency' | 'taskType';
  /** Height of the canvas - use responsive classes like 'h-96 md:h-[600px]' */
  height?: string;
}

// Color utilities
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  let r: number, g: number, b: number;

  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }

  return [r, g, b];
}

function getPointColor(
  point: EmbeddingPoint,
  colorBy: EmbeddingSpaceProps['colorBy']
): THREE.Color {
  switch (colorBy) {
    case 'confidence': {
      // Hue: confidence (red=low, green=high)
      const hue = point.confidence * 0.33; // 0 to 120 degrees
      const [r, g, b] = hslToRgb(hue, 0.8, 0.5);
      return new THREE.Color(r, g, b);
    }
    case 'agreement': {
      // Blue to green gradient
      const hue = 0.33 + point.agreement * 0.17; // 120 to 180 degrees
      const [r, g, b] = hslToRgb(hue, 0.7, 0.5);
      return new THREE.Color(r, g, b);
    }
    case 'recency': {
      // Purple (old) to yellow (new)
      const hue = 0.75 - point.recency * 0.6;
      const [r, g, b] = hslToRgb(hue, 0.6, 0.5);
      return new THREE.Color(r, g, b);
    }
    case 'taskType':
    default: {
      switch (point.taskType) {
        case 'chosen':
          return new THREE.Color(0x22c55e); // green
        case 'rejected':
          return new THREE.Color(0xef4444); // red
        case 'tie':
          return new THREE.Color(0xeab308); // yellow
        default:
          return new THREE.Color(0x6b7280); // gray
      }
    }
  }
}

// Instanced points component
interface InstancedPointsProps {
  points: EmbeddingPoint[];
  colorBy: EmbeddingSpaceProps['colorBy'];
  selectedIds: Set<string>;
  onPointClick: (point: EmbeddingPoint) => void;
  setHoveredId: (id: string | null) => void;
  focusedId?: string | null;
}

function InstancedPoints({
  points,
  colorBy,
  selectedIds,
  onPointClick,
  setHoveredId,
  focusedId,
}: InstancedPointsProps) {
  const meshRef = React.useRef<THREE.InstancedMesh>(null);
  const lastHoverIndexRef = React.useRef<number | null>(null);
  const tempObjectRef = React.useRef(new THREE.Object3D());
  const focusedIndexRef = React.useRef<number | null>(null);

  const getBaseScale = React.useCallback(
    (point: EmbeddingPoint) => (selectedIds.has(point.id) ? 1.3 : 1.0),
    [selectedIds]
  );

  const getBaseColor = React.useCallback(
    (point: EmbeddingPoint) => {
      const baseColor = getPointColor(point, colorBy);
      if (selectedIds.has(point.id)) {
        return new THREE.Color(1, 1, 1);
      }
      return baseColor;
    },
    [colorBy, selectedIds]
  );

  const indexById = React.useMemo(
    () => new Map(points.map((point, index) => [point.id, index])),
    [points]
  );

  const updateInstance = React.useCallback(
    (index: number, point: EmbeddingPoint, scale: number, color: THREE.Color) => {
      if (!meshRef.current) return;
      const tempObject = tempObjectRef.current;
      tempObject.position.set(...point.position);
      tempObject.scale.setScalar(scale);
      tempObject.updateMatrix();
      meshRef.current.setMatrixAt(index, tempObject.matrix);
      meshRef.current.setColorAt(index, color);
      meshRef.current.instanceMatrix.needsUpdate = true;
      if (meshRef.current.instanceColor) {
        meshRef.current.instanceColor.needsUpdate = true;
      }
    },
    []
  );

  const applyVisuals = React.useCallback(
    (index: number) => {
      const point = points[index];
      if (!point) return;
      const isHovered = index === lastHoverIndexRef.current;
      const isFocused = index === focusedIndexRef.current;
      const scaleMultiplier = isHovered ? 1.5 : isFocused ? 1.3 : 1;
      const colorMultiplier = isHovered ? 1.5 : isFocused ? 1.3 : 1;
      const color = getBaseColor(point).multiplyScalar(colorMultiplier);
      updateInstance(index, point, getBaseScale(point) * scaleMultiplier, color);
    },
    [points, getBaseColor, getBaseScale, updateInstance]
  );

  // Update instance matrices and colors
  React.useEffect(() => {
    if (!meshRef.current) return;

    const tempObject = new THREE.Object3D();
    const tempColor = new THREE.Color();

    points.forEach((point, i) => {
      tempObject.position.set(...point.position);
      const isSelected = selectedIds.has(point.id);
      tempObject.scale.setScalar(isSelected ? 1.3 : 1.0);

      tempObject.updateMatrix();
      meshRef.current!.setMatrixAt(i, tempObject.matrix);

      const baseColor = getPointColor(point, colorBy);
      if (isSelected) {
        tempColor.setRGB(1, 1, 1);
      } else {
        tempColor.copy(baseColor);
      }
      meshRef.current!.setColorAt(i, tempColor);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
    if (lastHoverIndexRef.current !== null) {
      applyVisuals(lastHoverIndexRef.current);
    }
    if (focusedIndexRef.current !== null) {
      applyVisuals(focusedIndexRef.current);
    }
  }, [points, colorBy, selectedIds, applyVisuals]);

  const handlePointerMove = React.useCallback(
    (event: ThreeEvent<PointerEvent>) => {
      if (!meshRef.current) return;
      const instanceId = event.instanceId;
      if (instanceId === undefined) {
        if (lastHoverIndexRef.current !== null) {
          const previousHover = lastHoverIndexRef.current;
          lastHoverIndexRef.current = null;
          applyVisuals(previousHover);
        }
        setHoveredId(null);
        return;
      }

      if (instanceId === lastHoverIndexRef.current) return;

      const previousHover = lastHoverIndexRef.current;
      lastHoverIndexRef.current = instanceId;
      if (previousHover !== null) {
        applyVisuals(previousHover);
      }
      const point = points[instanceId];
      if (!point) return;
      applyVisuals(instanceId);
      setHoveredId(point.id);
    },
    [points, applyVisuals, setHoveredId]
  );

  const handlePointerOut = React.useCallback(() => {
    if (lastHoverIndexRef.current !== null) {
      const previousHover = lastHoverIndexRef.current;
      lastHoverIndexRef.current = null;
      applyVisuals(previousHover);
    }
    setHoveredId(null);
  }, [applyVisuals, setHoveredId]);

  React.useEffect(() => {
    if (!focusedId) {
      if (focusedIndexRef.current !== null) {
        const previousIndex = focusedIndexRef.current;
        focusedIndexRef.current = null;
        applyVisuals(previousIndex);
      }
      return;
    }

    const nextIndex = indexById.get(focusedId) ?? null;
    if (nextIndex === focusedIndexRef.current) return;
    if (focusedIndexRef.current !== null) {
      applyVisuals(focusedIndexRef.current);
    }
    focusedIndexRef.current = nextIndex;
    if (nextIndex !== null) {
      applyVisuals(nextIndex);
    }
  }, [focusedId, indexById, applyVisuals]);

  const handleClick = React.useCallback(
    (event: ThreeEvent<MouseEvent>) => {
      const instanceId = event.instanceId;
      if (instanceId === undefined) return;
      const point = points[instanceId];
      if (point) {
        onPointClick(point);
      }
    },
    [points, onPointClick]
  );

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, points.length]}
      onClick={handleClick}
      onPointerMove={handlePointerMove}
      onPointerOut={handlePointerOut}
    >
      <sphereGeometry args={[0.05, 16, 16]} />
      <meshStandardMaterial vertexColors />
    </instancedMesh>
  );
}

// Tooltip component
interface TooltipProps {
  point: EmbeddingPoint | null;
  position: THREE.Vector3 | null;
}

function Tooltip({ point, position }: TooltipProps) {
  if (!point || !position) return null;

  return (
    <Html position={[position.x, position.y + 0.2, position.z]} center>
      <div className="bg-popover text-popover-foreground rounded-lg shadow-lg border p-3 min-w-[200px] pointer-events-none">
        <div className="font-medium text-sm mb-2 truncate max-w-[250px]">
          {point.prompt || `Task ${point.taskId.slice(0, 8)}...`}
        </div>
        <div className="grid grid-cols-2 gap-1 text-xs text-muted-foreground">
          <span>Confidence:</span>
          <span className="font-mono">{(point.confidence * 100).toFixed(1)}%</span>
          <span>Agreement:</span>
          <span className="font-mono">{(point.agreement * 100).toFixed(1)}%</span>
          <span>Type:</span>
          <span className="capitalize">{point.taskType}</span>
        </div>
      </div>
    </Html>
  );
}

// Selection box for lasso selection
interface SelectionBoxProps {
  onSelect: (bounds: THREE.Box3) => void;
}

function SelectionBox({ onSelect }: SelectionBoxProps) {
  const [isSelecting, setIsSelecting] = React.useState(false);
  const [startPoint, setStartPoint] = React.useState<THREE.Vector3 | null>(null);
  const { camera, gl } = useThree();

  React.useEffect(() => {
    const canvas = gl.domElement;

    const handleMouseDown = (e: MouseEvent) => {
      if (e.shiftKey) {
        setIsSelecting(true);
        const rect = canvas.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        const vector = new THREE.Vector3(x, y, 0.5).unproject(camera);
        setStartPoint(vector);
      }
    };

    const handleMouseUp = (e: MouseEvent) => {
      if (isSelecting && startPoint) {
        const rect = canvas.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        const endPoint = new THREE.Vector3(x, y, 0.5).unproject(camera);

        const box = new THREE.Box3();
        box.expandByPoint(startPoint);
        box.expandByPoint(endPoint);
        onSelect(box);
      }
      setIsSelecting(false);
      setStartPoint(null);
    };

    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mouseup', handleMouseUp);

    return () => {
      canvas.removeEventListener('mousedown', handleMouseDown);
      canvas.removeEventListener('mouseup', handleMouseUp);
    };
  }, [camera, gl, isSelecting, startPoint, onSelect]);

  return null;
}

interface EmbeddingA11yRegistryProps {
  points: EmbeddingPoint[];
  onFocusIdChange: (id: string | null) => void;
  onActivate: (point: EmbeddingPoint) => void;
}

function EmbeddingA11yRegistry({
  points,
  onFocusIdChange,
  onActivate,
}: EmbeddingA11yRegistryProps) {
  const {
    registerInteractiveObject,
    unregisterInteractiveObject,
    focusedObjectId,
    announce,
    setFocusedObjectId,
  } = useA11y();

  const limitedPoints = React.useMemo(
    () => points.slice(0, MAX_A11Y_POINTS),
    [points]
  );
  const pointById = React.useMemo(
    () => new Map(limitedPoints.map((point) => [point.id, point])),
    [limitedPoints]
  );
  const indexById = React.useMemo(
    () => new Map(limitedPoints.map((point, index) => [point.id, index])),
    [limitedPoints]
  );

  React.useEffect(() => {
    limitedPoints.forEach((point) => registerInteractiveObject(point.id));
    return () => {
      limitedPoints.forEach((point) => unregisterInteractiveObject(point.id));
    };
  }, [limitedPoints, registerInteractiveObject, unregisterInteractiveObject]);

  React.useEffect(() => {
    if (!focusedObjectId) {
      onFocusIdChange(null);
      return;
    }
    const point = pointById.get(focusedObjectId);
    if (!point) {
      onFocusIdChange(null);
      if (limitedPoints[0]) {
        setFocusedObjectId(limitedPoints[0].id);
      }
      return;
    }

    onFocusIdChange(point.id);
    const index = (indexById.get(point.id) ?? 0) + 1;
    announce(
      `Focused point ${index} of ${limitedPoints.length}. Confidence ${(
        point.confidence * 100
      ).toFixed(1)} percent. Agreement ${(point.agreement * 100).toFixed(1)} percent.`
    );
  }, [
    focusedObjectId,
    pointById,
    indexById,
    limitedPoints,
    announce,
    onFocusIdChange,
    setFocusedObjectId,
  ]);

  const handleActivate = React.useCallback(
    (id: string) => {
      const point = pointById.get(id);
      if (point) {
        onActivate(point);
      }
    },
    [pointById, onActivate]
  );

  return <KeyboardControls onActivate={handleActivate} />;
}

// Legend component
interface LegendProps {
  colorBy: EmbeddingSpaceProps['colorBy'];
}

function Legend({ colorBy }: LegendProps) {
  const legends: Record<string, { label: string; items: { color: string; label: string }[] }> = {
    taskType: {
      label: 'Response Type',
      items: [
        { color: '#22c55e', label: 'Chosen' },
        { color: '#ef4444', label: 'Rejected' },
        { color: '#eab308', label: 'Tie' },
      ],
    },
    confidence: {
      label: 'Confidence',
      items: [
        { color: '#ef4444', label: 'Low' },
        { color: '#eab308', label: 'Medium' },
        { color: '#22c55e', label: 'High' },
      ],
    },
    agreement: {
      label: 'Agreement',
      items: [
        { color: '#3b82f6', label: 'Low' },
        { color: '#06b6d4', label: 'Medium' },
        { color: '#22c55e', label: 'High' },
      ],
    },
    recency: {
      label: 'Recency',
      items: [
        { color: '#8b5cf6', label: 'Old' },
        { color: '#f59e0b', label: 'Recent' },
      ],
    },
  };

  const legend = legends[colorBy || 'taskType'];

  return (
    <div className="absolute bottom-4 left-4 bg-background/80 backdrop-blur-sm rounded-lg border p-3">
      <div className="text-xs font-medium mb-2">{legend?.label}</div>
      <div className="space-y-1">
        {legend?.items.map((item) => (
          <div key={item.label} className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-xs text-muted-foreground">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Stats overlay
interface StatsProps {
  pointCount: number;
  selectedCount: number;
}

function Stats({ pointCount, selectedCount }: StatsProps) {
  return (
    <div className="absolute top-4 left-4 bg-background/80 backdrop-blur-sm rounded-lg border p-3">
      <div className="text-xs space-y-1">
        <div>
          <span className="text-muted-foreground">Points:</span>{' '}
          <span className="font-mono">{pointCount.toLocaleString()}</span>
        </div>
        {selectedCount > 0 && (
          <div>
            <span className="text-muted-foreground">Selected:</span>{' '}
            <span className="font-mono">{selectedCount.toLocaleString()}</span>
          </div>
        )}
      </div>
    </div>
  );
}

// Empty state
function EmptyState() {
  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <div className="text-center flex flex-col items-center">
        <BarChart2 className="w-12 h-12 mb-4 text-muted-foreground" />
        <h3 className="text-lg font-medium mb-2">No Embeddings Yet</h3>
        <p className="text-sm text-muted-foreground max-w-[300px]">
          Start annotating tasks to populate the embedding space visualization.
        </p>
      </div>
    </div>
  );
}

// Loading state
function LoadingState() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-background/50">
      <div className="text-center">
        <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mb-4 mx-auto" />
        <p className="text-sm text-muted-foreground">Loading embeddings...</p>
      </div>
    </div>
  );
}

function createSeededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Generate mock embeddings for demo
function generateMockEmbeddings(count: number): EmbeddingPoint[] {
  const points: EmbeddingPoint[] = [];
  const types: EmbeddingPoint['taskType'][] = ['chosen', 'rejected', 'tie'];
  const rng = createSeededRandom(42);

  // Generate clusters
  const clusterCenters = [
    [2, 0, 0],
    [-2, 0, 0],
    [0, 2, 0],
    [0, -2, 0],
    [0, 0, 2],
  ];

  for (let i = 0; i < count; i++) {
    const cluster = clusterCenters[i % clusterCenters.length];
    const spread = 1.5;

    points.push({
      id: `point-${i}`,
      position: [
        (cluster?.[0] ?? 0) + (rng() - 0.5) * spread,
        (cluster?.[1] ?? 0) + (rng() - 0.5) * spread,
        (cluster?.[2] ?? 0) + (rng() - 0.5) * spread,
      ],
      confidence: rng(),
      agreement: rng(),
      recency: rng(),
      taskType: (types[Math.floor(rng() * types.length)] ?? 'chosen'),
      taskId: `task-${i}`,
      prompt: `Sample prompt for task ${i}: How would you rate this response?`,
    });
  }

  return points;
}

// Main component
export function EmbeddingSpace({
  onPointClick,
  onSelectionChange,
  maxPoints = 10000,
  colorBy = 'taskType',
  height = 'h-80 sm:h-96 md:h-[500px] lg:h-[600px]',
}: EmbeddingSpaceProps) {
  const [selectedIds, setSelectedIds] = React.useState<Set<string>>(new Set());
  const [hoveredId, setHoveredId] = React.useState<string | null>(null);
  const [a11yFocusId, setA11yFocusId] = React.useState<string | null>(null);

  // Reusable vectors to avoid garbage creation
  const tempVectorRef = React.useRef(new THREE.Vector3());
  const hoveredPositionRef = React.useRef(new THREE.Vector3());

  // Fetch embeddings
  const { data: points, isLoading, error } = useQuery({
    queryKey: ['embeddings', 'projections', maxPoints],
    queryFn: async () => {
      // In production, this would fetch from the API:
      // const response = await fetch(`/api/embeddings/projections?limit=${maxPoints}`);
      // return response.json();

      // For now, generate mock data
      return generateMockEmbeddings(Math.min(maxPoints, 1000));
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  const pointById = React.useMemo(
    () => new Map((points || []).map((point) => [point.id, point])),
    [points]
  );

  const hoveredPoint = React.useMemo(() => {
    if (!hoveredId) return null;
    return pointById.get(hoveredId) ?? null;
  }, [hoveredId, pointById]);

  // Calculate hovered position without creating garbage
  const hoveredPosition = React.useMemo(() => {
    if (!hoveredPoint) return null;
    return hoveredPositionRef.current.set(
      hoveredPoint.position[0],
      hoveredPoint.position[1],
      hoveredPoint.position[2]
    );
  }, [hoveredPoint]);

  // Handle point click
  const handlePointClick = React.useCallback(
    (point: EmbeddingPoint) => {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        if (next.has(point.id)) {
          next.delete(point.id);
        } else {
          next.add(point.id);
        }
        return next;
      });
      onPointClick?.(point);
    },
    [onPointClick]
  );

  // Handle box selection
  const handleBoxSelect = React.useCallback(
    (box: THREE.Box3) => {
      if (!points) return;

      // Reuse temp vector to avoid garbage creation
      const tempVector = tempVectorRef.current;
      const selected = points.filter((p) => {
        tempVector.set(p.position[0], p.position[1], p.position[2]);
        return box.containsPoint(tempVector);
      });

      const newSelection = new Set(selected.map((p) => p.id));
      setSelectedIds(newSelection);
      onSelectionChange?.(selected);
    },
    [points, onSelectionChange]
  );

  // Clear selection
  const clearSelection = React.useCallback(() => {
    setSelectedIds(new Set());
    onSelectionChange?.([]);
  }, [onSelectionChange]);

  if (isLoading) {
    return (
      <div className={`relative w-full ${height} bg-muted/20 rounded-lg overflow-hidden`}>
        <LoadingState />
      </div>
    );
  }

  if (error || !points || points.length === 0) {
    return (
      <div className={`relative w-full ${height} bg-muted/20 rounded-lg overflow-hidden`}>
        <EmptyState />
      </div>
    );
  }

  if (!isWebGLSupported()) {
    return (
      <div className={`relative w-full ${height} bg-muted/20 rounded-lg overflow-hidden`}>
        <SceneFallback
          description={`Embedding space visualization showing ${points.length} data points.`}
          dataTable={
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', padding: 6, borderBottom: '1px solid #333' }}>
                    ID
                  </th>
                  <th style={{ textAlign: 'left', padding: 6, borderBottom: '1px solid #333' }}>
                    Confidence
                  </th>
                  <th style={{ textAlign: 'left', padding: 6, borderBottom: '1px solid #333' }}>
                    Type
                  </th>
                </tr>
              </thead>
              <tbody>
                {points.slice(0, 10).map((point) => (
                  <tr key={point.id}>
                    <td style={{ padding: 6 }}>{point.id.slice(0, 8)}</td>
                    <td style={{ padding: 6 }}>{(point.confidence * 100).toFixed(1)}%</td>
                    <td style={{ padding: 6 }}>{point.taskType}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        />
      </div>
    );
  }

  const a11yPointCount = Math.min(points.length, MAX_A11Y_POINTS);
  const sceneDescription = `Embedding space visualization showing ${points.length} data points. ${a11yPointCount} points are available for keyboard navigation.`;

  return (
    <div className={`relative w-full ${height} bg-muted/20 rounded-lg overflow-hidden`}>
      <A11yProvider>
        <Canvas>
          <A11yCanvas sceneDescription={sceneDescription}>
            <PerspectiveCamera makeDefault position={[5, 5, 5]} fov={50} />
            <OrbitControls
              enablePan
              enableZoom
              enableRotate
              dampingFactor={0.05}
              rotateSpeed={0.5}
            />

            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} intensity={1} />

            <Bounds fit clip observe margin={1.2}>
              <InstancedPoints
                points={points}
                colorBy={colorBy}
                selectedIds={selectedIds}
                onPointClick={handlePointClick}
                setHoveredId={setHoveredId}
                focusedId={a11yFocusId}
              />
            </Bounds>

            <SelectionBox onSelect={handleBoxSelect} />
            <Tooltip point={hoveredPoint} position={hoveredPosition} />

            <EmbeddingA11yRegistry
              points={points}
              onFocusIdChange={setA11yFocusId}
              onActivate={handlePointClick}
            />

            {/* Grid helper */}
            <gridHelper args={[10, 10, 0x444444, 0x222222]} />

            {/* Axes helper */}
            <axesHelper args={[2]} />
          </A11yCanvas>
        </Canvas>
      </A11yProvider>

      <Legend colorBy={colorBy} />
      <Stats pointCount={points.length} selectedCount={selectedIds.size} />

      {/* Controls overlay */}
      <div className="absolute top-4 right-4 space-y-2">
        {selectedIds.size > 0 && (
          <button
            onClick={clearSelection}
            className="bg-background/80 backdrop-blur-sm rounded-lg border px-3 py-1.5 text-xs hover:bg-muted transition-colors"
          >
            Clear Selection
          </button>
        )}
        <div className="bg-background/80 backdrop-blur-sm rounded-lg border px-3 py-1.5 text-xs text-muted-foreground">
          <div>Shift+Drag: Box Select</div>
          <div>Click/Enter: Toggle Select</div>
          <div>Tab: Focus Points (first {a11yPointCount})</div>
          <div>Arrow Keys: Rotate</div>
          <div>Scroll/+/-: Zoom</div>
        </div>
      </div>
    </div>
  );
}

export default EmbeddingSpace;
