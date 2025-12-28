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
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import {
  OrbitControls,
  PerspectiveCamera,
  Html,
  Bounds,
} from '@react-three/drei';
import * as THREE from 'three';
import { useQuery } from '@tanstack/react-query';

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
  hoveredId: string | null;
  setHoveredId: (id: string | null) => void;
}

function InstancedPoints({
  points,
  colorBy,
  selectedIds,
  onPointClick,
  hoveredId,
  setHoveredId,
}: InstancedPointsProps) {
  const meshRef = React.useRef<THREE.InstancedMesh>(null);
  const { camera, raycaster, pointer } = useThree();

  // Update instance matrices and colors
  React.useEffect(() => {
    if (!meshRef.current) return;

    const tempObject = new THREE.Object3D();
    const tempColor = new THREE.Color();

    points.forEach((point, i) => {
      // Position
      tempObject.position.set(...point.position);

      // Scale based on selection/hover state
      const isSelected = selectedIds.has(point.id);
      const isHovered = hoveredId === point.id;
      const scale = isHovered ? 1.5 : isSelected ? 1.3 : 1.0;
      tempObject.scale.setScalar(scale);

      tempObject.updateMatrix();
      meshRef.current!.setMatrixAt(i, tempObject.matrix);

      // Color
      const baseColor = getPointColor(point, colorBy);
      if (isSelected) {
        tempColor.setRGB(1, 1, 1); // White for selected
      } else if (isHovered) {
        tempColor.copy(baseColor).multiplyScalar(1.5);
      } else {
        tempColor.copy(baseColor);
      }
      meshRef.current!.setColorAt(i, tempColor);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [points, colorBy, selectedIds, hoveredId]);

  // Raycasting for hover/click detection
  useFrame(() => {
    if (!meshRef.current) return;

    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObject(meshRef.current);

    if (intersects.length > 0) {
      const instanceId = intersects[0]?.instanceId;
      if (instanceId !== undefined && points[instanceId]) {
        setHoveredId(points[instanceId].id);
      }
    } else {
      setHoveredId(null);
    }
  });

  const handleClick = React.useCallback(() => {
    if (hoveredId) {
      const point = points.find((p) => p.id === hoveredId);
      if (point) {
        onPointClick(point);
      }
    }
  }, [hoveredId, points, onPointClick]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, points.length]}
      onClick={handleClick}
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

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleMouseDown = (e: any) => {
      if (e.shiftKey) {
        setIsSelecting(true);
        const rect = canvas.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        const vector = new THREE.Vector3(x, y, 0.5).unproject(camera);
        setStartPoint(vector);
      }
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleMouseUp = (e: any) => {
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
      <div className="text-center">
        <div className="text-4xl mb-4">📊</div>
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

// Generate mock embeddings for demo
function generateMockEmbeddings(count: number): EmbeddingPoint[] {
  const points: EmbeddingPoint[] = [];
  const types: EmbeddingPoint['taskType'][] = ['chosen', 'rejected', 'tie'];

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
        (cluster?.[0] ?? 0) + (Math.random() - 0.5) * spread,
        (cluster?.[1] ?? 0) + (Math.random() - 0.5) * spread,
        (cluster?.[2] ?? 0) + (Math.random() - 0.5) * spread,
      ],
      confidence: Math.random(),
      agreement: Math.random(),
      recency: Math.random(),
      taskType: (types[Math.floor(Math.random() * types.length)] ?? 'chosen'),
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
}: EmbeddingSpaceProps) {
  const [selectedIds, setSelectedIds] = React.useState<Set<string>>(new Set());
  const [hoveredId, setHoveredId] = React.useState<string | null>(null);
  const [hoveredPoint, setHoveredPoint] = React.useState<EmbeddingPoint | null>(null);

  // Reusable vectors to avoid garbage creation
  const tempVectorRef = React.useRef(new THREE.Vector3());
  const hoveredPositionRef = React.useRef(new THREE.Vector3());

  // Calculate hovered position without creating garbage (move hook before early returns)
  const hoveredPosition = React.useMemo(() => {
    if (!hoveredPoint) return null;
    return hoveredPositionRef.current.set(
      hoveredPoint.position[0],
      hoveredPoint.position[1],
      hoveredPoint.position[2]
    );
  }, [hoveredPoint]);

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

  // Update hovered point
  React.useEffect(() => {
    if (hoveredId && points) {
      const point = points.find((p) => p.id === hoveredId);
      setHoveredPoint(point || null);
    } else {
      setHoveredPoint(null);
    }
  }, [hoveredId, points]);

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
      <div className="relative w-full h-[600px] bg-muted/20 rounded-lg overflow-hidden">
        <LoadingState />
      </div>
    );
  }

  if (error || !points || points.length === 0) {
    return (
      <div className="relative w-full h-[600px] bg-muted/20 rounded-lg overflow-hidden">
        <EmptyState />
      </div>
    );
  }


  return (
    <div
      className="relative w-full h-[600px] bg-muted/20 rounded-lg overflow-hidden"
      role="application"
      aria-label={`3D embedding space visualization showing ${points?.length || 0} data points. Use mouse to rotate, scroll to zoom, shift+drag to select.`}
    >
      <Canvas
        aria-hidden="true"
        tabIndex={-1}
      >
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
            hoveredId={hoveredId}
            setHoveredId={setHoveredId}
          />
        </Bounds>

        <SelectionBox onSelect={handleBoxSelect} />
        <Tooltip point={hoveredPoint} position={hoveredPosition} />

        {/* Grid helper */}
        <gridHelper args={[10, 10, 0x444444, 0x222222]} />

        {/* Axes helper */}
        <axesHelper args={[2]} />
      </Canvas>

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
          <div>Click: Toggle Select</div>
          <div>Scroll: Zoom</div>
        </div>
      </div>
    </div>
  );
}

export default EmbeddingSpace;
