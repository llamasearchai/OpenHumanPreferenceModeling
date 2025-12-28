/**
 * Preference Landscape 3D Scene
 *
 * Purpose: Visualize high-dimensional preference data projected to 3D space
 * Features t-SNE/UMAP point cloud with cluster coloring and confidence-based sizing
 */

import React, { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html, PerspectiveCamera, Stars } from '@react-three/drei';
import * as THREE from 'three';

import { getCanvasPropsForPreset, isWebGLSupported } from '@/lib/three/canvas-config';
import { usePerformanceMonitor } from '@/lib/three/performance-monitor';
import { useSelection, useKeyboardNavigation } from '@/lib/three/interactions';
import { PreferencePointMaterial, createPreferencePointGeometry, defaultClusterColors } from '@/lib/three/materials/PreferencePointMaterial';
import { EffectsPipeline, effectPresets } from '@/lib/three/effects/EffectsPipeline';
import { A11yProvider, A11yCanvas, KeyboardControls, SceneFallback } from '@/lib/three/a11y/A11yScene';

// ============================================================================
// Types
// ============================================================================

export interface PreferenceDataPoint {
  id: string;
  position: [number, number, number];
  confidence: number;
  clusterId: number;
  label?: string;
  metadata?: Record<string, unknown>;
}

export interface PreferenceLandscapeProps {
  /** Data points to visualize */
  dataPoints: PreferenceDataPoint[];
  /** Cluster color palette */
  clusterColors?: string[];
  /** Callback when point is selected */
  onPointSelect?: (point: PreferenceDataPoint | null) => void;
  /** Callback when point is hovered */
  onPointHover?: (point: PreferenceDataPoint | null) => void;
  /** Enable effects */
  enableEffects?: boolean;
  /** Show stars background */
  showStars?: boolean;
  /** Show grid */
  showGrid?: boolean;
  /** Initial camera position */
  cameraPosition?: [number, number, number];
  /** Auto-rotate speed (0 to disable) */
  autoRotateSpeed?: number;
  /** Performance quality */
  quality?: 'high' | 'medium' | 'low';
  /** Show performance stats */
  showStats?: boolean;
  /** Custom className */
  className?: string;
}

// ============================================================================
// Point Cloud Component
// ============================================================================

interface PointCloudProps {
  dataPoints: PreferenceDataPoint[];
  clusterColors: string[];
  selectedId: string | null;
  hoveredId: string | null;
  onPointClick: (point: PreferenceDataPoint) => void;
  onPointHover: (point: PreferenceDataPoint | null) => void;
  quality: 'high' | 'medium' | 'low';
}

function PointCloud({
  dataPoints,
  clusterColors,
  selectedId,
  hoveredId,
  onPointClick,
  onPointHover,
  quality,
}: PointCloudProps): React.ReactElement | null {
  const pointsRef = useRef<THREE.Points>(null);
  const materialRef = useRef<PreferencePointMaterial>(null);
  const { camera, raycaster, gl } = useThree();

  // Create geometry
  const geometry = useMemo(() => {
    const points = dataPoints.map((p) => ({
      position: p.position,
      confidence: p.confidence,
      clusterId: p.clusterId,
    }));
    return createPreferencePointGeometry(points, clusterColors);
  }, [dataPoints, clusterColors]);

  // Create material
  const material = useMemo(() => {
    const sizeConfig = {
      high: { min: 3, max: 10 },
      medium: { min: 2, max: 8 },
      low: { min: 2, max: 6 },
    };
    return new PreferencePointMaterial({
      pointSizeMin: sizeConfig[quality].min,
      pointSizeMax: sizeConfig[quality].max,
      showGlow: quality !== 'low',
    });
  }, [quality]);

  // Update time uniform for animations
  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.updateTime(state.clock.elapsedTime);
    }
  });

  // Handle pointer events
  const handlePointerMove = useCallback(
    (event: THREE.Event) => {
      if (!pointsRef.current) return;

      const pointer = new THREE.Vector2();
      const rect = gl.domElement.getBoundingClientRect();
      const e = event as unknown as { clientX: number; clientY: number };
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(pointer, camera);
      raycaster.params.Points = { threshold: 0.5 };

      const intersects = raycaster.intersectObject(pointsRef.current);
      if (intersects.length > 0 && intersects[0].index !== undefined) {
        const point = dataPoints[intersects[0].index];
        onPointHover(point);
      } else {
        onPointHover(null);
      }
    },
    [dataPoints, camera, raycaster, gl, onPointHover]
  );

  const handleClick = useCallback(
    (event: THREE.Event) => {
      if (!pointsRef.current) return;

      const pointer = new THREE.Vector2();
      const rect = gl.domElement.getBoundingClientRect();
      const e = event as unknown as { clientX: number; clientY: number };
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(pointer, camera);
      raycaster.params.Points = { threshold: 0.5 };

      const intersects = raycaster.intersectObject(pointsRef.current);
      if (intersects.length > 0 && intersects[0].index !== undefined) {
        const point = dataPoints[intersects[0].index];
        onPointClick(point);
      }
    },
    [dataPoints, camera, raycaster, gl, onPointClick]
  );

  if (dataPoints.length === 0) return null;

  return (
    <points
      ref={pointsRef}
      geometry={geometry}
      material={material}
      onPointerMove={handlePointerMove}
      onClick={handleClick}
    >
      <primitive ref={materialRef} object={material} attach="material" />
    </points>
  );
}

// ============================================================================
// Tooltip Component
// ============================================================================

interface TooltipProps {
  point: PreferenceDataPoint | null;
}

function Tooltip({ point }: TooltipProps): React.ReactElement | null {
  if (!point) return null;

  return (
    <Html position={point.position} center distanceFactor={10}>
      <div
        style={{
          background: 'rgba(0, 0, 0, 0.85)',
          color: 'white',
          padding: '8px 12px',
          borderRadius: 6,
          fontSize: 12,
          whiteSpace: 'nowrap',
          pointerEvents: 'none',
          transform: 'translateY(-100%)',
          marginBottom: 8,
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 4 }}>
          {point.label || `Point ${point.id.slice(0, 8)}`}
        </div>
        <div style={{ opacity: 0.8 }}>
          Cluster: {point.clusterId} | Confidence: {point.confidence}/5
        </div>
        {point.metadata && (
          <div style={{ marginTop: 4, fontSize: 11, opacity: 0.6 }}>
            {Object.entries(point.metadata)
              .slice(0, 3)
              .map(([key, value]) => (
                <div key={key}>
                  {key}: {String(value)}
                </div>
              ))}
          </div>
        )}
      </div>
    </Html>
  );
}

// ============================================================================
// Scene Content
// ============================================================================

interface SceneContentProps extends Omit<PreferenceLandscapeProps, 'className'> {
  selectedPoint: PreferenceDataPoint | null;
  setSelectedPoint: (point: PreferenceDataPoint | null) => void;
}

function SceneContent({
  dataPoints,
  clusterColors = defaultClusterColors,
  onPointSelect,
  onPointHover,
  enableEffects = true,
  showStars = true,
  showGrid = true,
  autoRotateSpeed = 0,
  quality = 'high',
  selectedPoint,
  setSelectedPoint,
}: SceneContentProps): React.ReactElement {
  const [hoveredPoint, setHoveredPoint] = useState<PreferenceDataPoint | null>(null);
  const controlsRef = useRef<any>(null);

  // Performance monitoring
  const { metrics, quality: autoQuality } = usePerformanceMonitor({
    targetFps: quality === 'low' ? 30 : 60,
    autoAdjust: true,
  });

  // Handle point selection
  const handlePointClick = useCallback(
    (point: PreferenceDataPoint) => {
      setSelectedPoint(selectedPoint?.id === point.id ? null : point);
      onPointSelect?.(selectedPoint?.id === point.id ? null : point);
    },
    [selectedPoint, setSelectedPoint, onPointSelect]
  );

  // Handle point hover
  const handlePointHover = useCallback(
    (point: PreferenceDataPoint | null) => {
      setHoveredPoint(point);
      onPointHover?.(point);
    },
    [onPointHover]
  );

  // Keyboard navigation
  useKeyboardNavigation({ enabled: true });

  const effectiveQuality = autoQuality || quality;

  return (
    <>
      {/* Camera */}
      <PerspectiveCamera makeDefault position={[0, 50, 100]} fov={60} />

      {/* Controls */}
      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.05}
        minDistance={10}
        maxDistance={500}
        autoRotate={autoRotateSpeed > 0}
        autoRotateSpeed={autoRotateSpeed}
      />

      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[100, 100, 100]} intensity={1} />
      <pointLight position={[-100, -100, -100]} intensity={0.5} />

      {/* Background */}
      {showStars && <Stars radius={300} depth={50} count={5000} factor={4} fade speed={1} />}

      {/* Grid */}
      {showGrid && (
        <gridHelper args={[200, 50, '#1e293b', '#0f172a']} position={[0, -20, 0]} />
      )}

      {/* Point Cloud */}
      <PointCloud
        dataPoints={dataPoints}
        clusterColors={clusterColors}
        selectedId={selectedPoint?.id || null}
        hoveredId={hoveredPoint?.id || null}
        onPointClick={handlePointClick}
        onPointHover={handlePointHover}
        quality={effectiveQuality}
      />

      {/* Tooltip */}
      <Tooltip point={hoveredPoint} />

      {/* Effects */}
      {enableEffects && (
        <EffectsPipeline
          config={effectPresets.dataVisualization}
          quality={effectiveQuality}
        />
      )}
    </>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function PreferenceLandscape({
  dataPoints,
  clusterColors = defaultClusterColors,
  onPointSelect,
  onPointHover,
  enableEffects = true,
  showStars = true,
  showGrid = true,
  cameraPosition = [0, 50, 100],
  autoRotateSpeed = 0,
  quality = 'high',
  showStats = false,
  className,
}: PreferenceLandscapeProps): React.ReactElement {
  const [selectedPoint, setSelectedPoint] = useState<PreferenceDataPoint | null>(null);

  // Check WebGL support
  if (!isWebGLSupported()) {
    return (
      <SceneFallback
        description={`3D scatter plot showing ${dataPoints.length} preference data points across ${new Set(dataPoints.map((p) => p.clusterId)).size} clusters`}
        dataTable={
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ padding: 8, textAlign: 'left', borderBottom: '1px solid #333' }}>ID</th>
                <th style={{ padding: 8, textAlign: 'left', borderBottom: '1px solid #333' }}>Cluster</th>
                <th style={{ padding: 8, textAlign: 'left', borderBottom: '1px solid #333' }}>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {dataPoints.slice(0, 10).map((point) => (
                <tr key={point.id}>
                  <td style={{ padding: 8 }}>{point.id.slice(0, 8)}</td>
                  <td style={{ padding: 8 }}>{point.clusterId}</td>
                  <td style={{ padding: 8 }}>{point.confidence}/5</td>
                </tr>
              ))}
              {dataPoints.length > 10 && (
                <tr>
                  <td colSpan={3} style={{ padding: 8, fontStyle: 'italic' }}>
                    ...and {dataPoints.length - 10} more points
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        }
      />
    );
  }

  const canvasProps = getCanvasPropsForPreset('preference-landscape');

  return (
    <div className={className} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        {...canvasProps}
        camera={{ position: cameraPosition, fov: 60 }}
        style={{ background: 'linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%)' }}
      >
        <A11yProvider>
          <A11yCanvas sceneDescription={`3D preference landscape with ${dataPoints.length} data points`}>
            <KeyboardControls
              onActivate={(id) => {
                const point = dataPoints.find((p) => p.id === id);
                if (point) {
                  setSelectedPoint(point);
                  onPointSelect?.(point);
                }
              }}
            />
            <SceneContent
              dataPoints={dataPoints}
              clusterColors={clusterColors}
              onPointSelect={onPointSelect}
              onPointHover={onPointHover}
              enableEffects={enableEffects}
              showStars={showStars}
              showGrid={showGrid}
              autoRotateSpeed={autoRotateSpeed}
              quality={quality}
              selectedPoint={selectedPoint}
              setSelectedPoint={setSelectedPoint}
            />
          </A11yCanvas>
        </A11yProvider>
      </Canvas>

      {/* Legend */}
      <div
        style={{
          position: 'absolute',
          bottom: 16,
          left: 16,
          background: 'rgba(0, 0, 0, 0.7)',
          padding: 12,
          borderRadius: 8,
          color: 'white',
          fontSize: 12,
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 8 }}>Clusters</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {clusterColors.slice(0, new Set(dataPoints.map((p) => p.clusterId)).size).map((color, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  background: color,
                }}
              />
              <span>Cluster {i}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Selected point info */}
      {selectedPoint && (
        <div
          style={{
            position: 'absolute',
            top: 16,
            right: 16,
            background: 'rgba(0, 0, 0, 0.85)',
            padding: 16,
            borderRadius: 8,
            color: 'white',
            maxWidth: 300,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 8 }}>
            {selectedPoint.label || `Point ${selectedPoint.id.slice(0, 8)}`}
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.5 }}>
            <div>Cluster: {selectedPoint.clusterId}</div>
            <div>Confidence: {selectedPoint.confidence}/5</div>
            <div>
              Position: ({selectedPoint.position.map((v) => v.toFixed(2)).join(', ')})
            </div>
            {selectedPoint.metadata && (
              <div style={{ marginTop: 8, borderTop: '1px solid #333', paddingTop: 8 }}>
                {Object.entries(selectedPoint.metadata).map(([key, value]) => (
                  <div key={key}>
                    <span style={{ opacity: 0.7 }}>{key}:</span> {String(value)}
                  </div>
                ))}
              </div>
            )}
          </div>
          <button
            onClick={() => {
              setSelectedPoint(null);
              onPointSelect?.(null);
            }}
            style={{
              marginTop: 12,
              padding: '6px 12px',
              background: '#3b82f6',
              border: 'none',
              borderRadius: 4,
              color: 'white',
              cursor: 'pointer',
            }}
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}

export default PreferenceLandscape;
