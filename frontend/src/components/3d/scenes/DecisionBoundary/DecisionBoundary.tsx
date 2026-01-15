/**
 * Decision Boundary 3D Scene
 *
 * Purpose: Visualize model decision boundaries as a 3D surface mesh
 * Shows confidence gradients from model predictions
 */

import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, Float } from '@react-three/drei';
import * as THREE from 'three';

import { getCanvasPropsForPreset, isWebGLSupported } from '@/lib/three/canvas-config';
import { usePerformanceMonitor } from '@/lib/three/performance-monitor';
import { EffectsPipeline, effectPresets } from '@/lib/three/effects/EffectsPipeline';
import {
  A11yProvider,
  A11yCanvas,
  KeyboardControls,
  SceneFallback,
  useA11y,
} from '@/lib/three/a11y/A11yScene';

// ============================================================================
// Types
// ============================================================================

export interface SurfaceData {
  /** Grid resolution (e.g., 100 for 100x100) */
  resolution: number;
  /** Z values as flat array of length resolution*resolution */
  values: number[];
  /** X range [min, max] */
  xRange: [number, number];
  /** Y range [min, max] */
  yRange: [number, number];
  /** Z range for color mapping [min, max] */
  zRange: [number, number];
}

export interface DecisionBoundaryProps {
  /** Surface data from model predictions */
  surfaceData: SurfaceData;
  /** Color gradient for confidence (low to high) */
  colorGradient?: [string, string, string];
  /** Enable auto-rotation */
  autoRotate?: boolean;
  /** Auto-rotation speed in rad/s */
  autoRotateSpeed?: number;
  /** Show wireframe overlay */
  showWireframe?: boolean;
  /** Enable hover highlighting */
  enableHover?: boolean;
  /** Show axis labels */
  showAxes?: boolean;
  /** Quality setting */
  quality?: 'high' | 'medium' | 'low';
  /** Custom className */
  className?: string;
}

// ============================================================================
// Surface Mesh Component
// ============================================================================

interface SurfaceMeshProps {
  surfaceData: SurfaceData;
  colorGradient: [string, string, string];
  showWireframe: boolean;
  quality: 'high' | 'medium' | 'low';
}

function SurfaceMesh({
  surfaceData,
  colorGradient,
  showWireframe,
  quality,
}: SurfaceMeshProps): React.ReactElement {
  const meshRef = useRef<THREE.Mesh>(null);
  const wireframeRef = useRef<THREE.LineSegments>(null);

  // Create geometry from surface data
  const geometry = useMemo(() => {
    const { resolution, values, xRange, yRange, zRange } = surfaceData;
    const effectiveRes = quality === 'low' ? Math.min(resolution, 50) : resolution;

    const geo = new THREE.PlaneGeometry(
      xRange[1] - xRange[0],
      yRange[1] - yRange[0],
      effectiveRes - 1,
      effectiveRes - 1
    );

    // Apply height values
    const positions = geo.attributes.position.array as Float32Array;
    const colors = new Float32Array(positions.length);
    const step = resolution / effectiveRes;

    // Parse colors
    const colorLow = new THREE.Color(colorGradient[0]);
    const colorMid = new THREE.Color(colorGradient[1]);
    const colorHigh = new THREE.Color(colorGradient[2]);

    for (let i = 0; i < effectiveRes; i++) {
      for (let j = 0; j < effectiveRes; j++) {
        const idx = i * effectiveRes + j;
        const vertexIdx = idx * 3;

        // Sample from original resolution
        const origI = Math.min(Math.floor(i * step), resolution - 1);
        const origJ = Math.min(Math.floor(j * step), resolution - 1);
        const value = values[origI * resolution + origJ];

        // Set height (z becomes y in default plane orientation)
        positions[vertexIdx + 2] = value;

        // Calculate color based on normalized value
        const normalizedValue = (value - zRange[0]) / (zRange[1] - zRange[0]);

        let color: THREE.Color;
        if (normalizedValue < 0.5) {
          color = colorLow.clone().lerp(colorMid, normalizedValue * 2);
        } else {
          color = colorMid.clone().lerp(colorHigh, (normalizedValue - 0.5) * 2);
        }

        colors[vertexIdx] = color.r;
        colors[vertexIdx + 1] = color.g;
        colors[vertexIdx + 2] = color.b;
      }
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();

    return geo;
  }, [surfaceData, colorGradient, quality]);

  React.useEffect(() => {
    return () => {
      geometry.dispose();
    };
  }, [geometry]);

  // Create wireframe geometry
  const wireframeGeometry = useMemo(() => {
    if (!showWireframe) return null;
    return new THREE.WireframeGeometry(geometry);
  }, [geometry, showWireframe]);

  React.useEffect(() => {
    return () => {
      wireframeGeometry?.dispose();
    };
  }, [wireframeGeometry]);

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {/* Main surface */}
      <mesh ref={meshRef} geometry={geometry}>
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          roughness={0.6}
          metalness={0.2}
        />
      </mesh>

      {/* Wireframe overlay */}
      {showWireframe && wireframeGeometry && (
        <lineSegments ref={wireframeRef} geometry={wireframeGeometry}>
          <lineBasicMaterial color="#ffffff" opacity={0.1} transparent />
        </lineSegments>
      )}
    </group>
  );
}

// ============================================================================
// Auto-Rotating Container
// ============================================================================

interface AutoRotateContainerProps {
  speed: number;
  enabled: boolean;
  children: React.ReactNode;
}

function AutoRotateContainer({
  speed,
  enabled,
  children,
}: AutoRotateContainerProps): React.ReactElement {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((_, delta) => {
    if (enabled && groupRef.current) {
      groupRef.current.rotation.y += speed * delta;
    }
  });

  return <group ref={groupRef}>{children}</group>;
}

// ============================================================================
// Axis Labels
// ============================================================================

interface AxisLabelsProps {
  xRange: [number, number];
  yRange: [number, number];
  zRange: [number, number];
}

function AxisLabels({ xRange, yRange, zRange }: AxisLabelsProps): React.ReactElement {
  // unused labelStyle removed
  const xCenter = (xRange[0] + xRange[1]) / 2;
  // unused yCenter removed

  return (
    <group>
      {/* X axis */}
      <group position={[xCenter, -1, yRange[0] - 2]}>
        <axesHelper args={[2]} />
      </group>

      {/* Axis lines */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([xRange[0], 0, yRange[0], xRange[1], 0, yRange[0]])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#ff6b6b" />
      </line>

      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([xRange[0], 0, yRange[0], xRange[0], 0, yRange[1]])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#4ecdc4" />
      </line>

      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([xRange[0], zRange[0], yRange[0], xRange[0], zRange[1], yRange[0]])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#45b7d1" />
      </line>
    </group>
  );
}

// ============================================================================
// Scene Content
// ============================================================================

interface SceneContentProps extends Omit<DecisionBoundaryProps, 'className'> {}

function SceneContent({
  surfaceData,
  colorGradient = ['#3b82f6', '#fbbf24', '#ef4444'],
  autoRotate = true,
  autoRotateSpeed = 0.5,
  showWireframe = false,
  showAxes = true,
  quality = 'high',
}: SceneContentProps): React.ReactElement {
  const { prefersReducedMotion } = useA11y();
  const shouldAutoRotate = autoRotate && !prefersReducedMotion;
  const floatSpeed = prefersReducedMotion ? 0 : 0.5;
  const floatIntensity = prefersReducedMotion ? 0 : 0.1;

  // Performance monitoring
  usePerformanceMonitor({
    targetFps: quality === 'low' ? 30 : 60,
    autoAdjust: true,
  });

  const xSize = surfaceData.xRange[1] - surfaceData.xRange[0];
  const ySize = surfaceData.yRange[1] - surfaceData.yRange[0];
  const cameraDistance = Math.max(xSize, ySize) * 1.5;

  return (
    <>
      {/* Camera */}
      <PerspectiveCamera
        makeDefault
        position={[cameraDistance * 0.7, cameraDistance * 0.5, cameraDistance * 0.7]}
        fov={45}
      />

      {/* Controls */}
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={cameraDistance * 0.3}
        maxDistance={cameraDistance * 3}
        target={[
          (surfaceData.xRange[0] + surfaceData.xRange[1]) / 2,
          0,
          (surfaceData.yRange[0] + surfaceData.yRange[1]) / 2,
        ]}
      />

      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 10]} intensity={1} castShadow />
      <directionalLight position={[-10, 10, -10]} intensity={0.5} />

      {/* Environment */}
      <Environment preset="city" />

      {/* Surface */}
      <AutoRotateContainer speed={autoRotateSpeed} enabled={shouldAutoRotate}>
        <group
          position={[
            -(surfaceData.xRange[0] + surfaceData.xRange[1]) / 2,
            0,
            -(surfaceData.yRange[0] + surfaceData.yRange[1]) / 2,
          ]}
        >
          <Float speed={floatSpeed} rotationIntensity={0} floatIntensity={floatIntensity}>
            <SurfaceMesh
              surfaceData={surfaceData}
              colorGradient={colorGradient}
              showWireframe={showWireframe}
              quality={quality}
            />
          </Float>

          {showAxes && (
            <AxisLabels
              xRange={surfaceData.xRange}
              yRange={surfaceData.yRange}
              zRange={surfaceData.zRange}
            />
          )}
        </group>
      </AutoRotateContainer>

      {/* Ground plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]} receiveShadow>
        <planeGeometry args={[100, 100]} />
        <shadowMaterial opacity={0.3} />
      </mesh>

      {/* Effects */}
      <EffectsPipeline config={effectPresets.cinematic} quality={quality} />
    </>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function DecisionBoundary({
  surfaceData,
  colorGradient = ['#3b82f6', '#fbbf24', '#ef4444'],
  autoRotate = true,
  autoRotateSpeed = 0.5,
  showWireframe = false,
  // enableHover unused
  showAxes = true,
  quality = 'high',
  className,
}: DecisionBoundaryProps): React.ReactElement {
  // Check WebGL support
  if (!isWebGLSupported()) {
    return (
      <SceneFallback
        description={`Decision boundary surface visualization with ${surfaceData.resolution}x${surfaceData.resolution} resolution`}
      />
    );
  }

  const canvasProps = getCanvasPropsForPreset('decision-boundary');

  return (
    <div className={className} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        {...canvasProps}
        shadows
        style={{ background: 'linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)' }}
      >
        <A11yProvider>
          <A11yCanvas sceneDescription="3D decision boundary surface showing model confidence levels">
            <KeyboardControls />
            <SceneContent
              surfaceData={surfaceData}
              colorGradient={colorGradient}
              autoRotate={autoRotate}
              autoRotateSpeed={autoRotateSpeed}
              showWireframe={showWireframe}
              showAxes={showAxes}
              quality={quality}
            />
          </A11yCanvas>
        </A11yProvider>
      </Canvas>

      {/* Color legend */}
      <div
        style={{
          position: 'absolute',
          bottom: 16,
          right: 16,
          background: 'rgba(0, 0, 0, 0.7)',
          padding: 12,
          borderRadius: 8,
          color: 'white',
          fontSize: 12,
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 8 }}>Confidence</div>
        <div
          style={{
            width: 150,
            height: 16,
            background: `linear-gradient(to right, ${colorGradient.join(', ')})`,
            borderRadius: 4,
          }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
          <span>Low</span>
          <span>High</span>
        </div>
      </div>

      {/* Controls hint */}
      <div
        style={{
          position: 'absolute',
          bottom: 16,
          left: 16,
          background: 'rgba(0, 0, 0, 0.7)',
          padding: 8,
          borderRadius: 8,
          color: 'white',
          fontSize: 11,
          opacity: 0.7,
        }}
      >
        Drag to rotate • Scroll to zoom
      </div>
    </div>
  );
}

export default DecisionBoundary;
