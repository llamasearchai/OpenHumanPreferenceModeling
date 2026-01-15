/**
 * Model Configurator 3D Scene
 *
 * Purpose: Interactive parameter tuning with real-time 3D preview
 * Allows adjusting model parameters and seeing predicted accuracy surface
 */

import React, { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, Html, MeshDistortMaterial } from '@react-three/drei';
import * as THREE from 'three';

import { getCanvasPropsForPreset, isWebGLSupported } from '@/lib/three/canvas-config';
import { usePerformanceMonitor } from '@/lib/three/performance-monitor';
import { useLerp } from '@/lib/three/animations';
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

export interface ModelParameter {
  id: string;
  name: string;
  min: number;
  max: number;
  step: number;
  value: number;
  unit?: string;
  description?: string;
}

export interface ModelConfiguratorProps {
  /** Model parameters to configure */
  parameters: ModelParameter[];
  /** Callback when parameters change */
  onParameterChange?: (parameterId: string, value: number) => void;
  /** Callback to get predicted accuracy for current parameters */
  getPredictedAccuracy?: (params: Record<string, number>) => number;
  /** Show parameter panel */
  showPanel?: boolean;
  /** Quality setting */
  quality?: 'high' | 'medium' | 'low';
  /** Custom className */
  className?: string;
}

// ============================================================================
// Accuracy Preview Sphere
// ============================================================================

interface AccuracyPreviewProps {
  accuracy: number;
  targetAccuracy: number;
}

function AccuracyPreview({ accuracy, targetAccuracy }: AccuracyPreviewProps): React.ReactElement {
  const meshRef = useRef<THREE.Mesh>(null);
  const { value: animatedAccuracy } = useLerp({ target: accuracy, factor: 0.05 });
  const { prefersReducedMotion } = useA11y();

  // Color based on accuracy
  const color = useMemo(() => {
    const acc = animatedAccuracy as number;
    if (acc >= 0.9) return '#22c55e';
    if (acc >= 0.75) return '#eab308';
    return '#ef4444';
  }, [animatedAccuracy]);

  // Scale based on accuracy
  const scale = useMemo(() => {
    return 0.5 + (animatedAccuracy as number) * 0.5;
  }, [animatedAccuracy]);

  useFrame((state) => {
    if (prefersReducedMotion) return;
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.2;
    }
  });

  return (
    <group position={[0, 0, 0]}>
      {/* Main accuracy sphere */}
      <mesh ref={meshRef} scale={scale}>
        <sphereGeometry args={[1, 64, 64]} />
        <MeshDistortMaterial
          color={color}
          distort={prefersReducedMotion ? 0 : 0.3}
          speed={prefersReducedMotion ? 0 : 2}
          roughness={0.2}
          metalness={0.8}
        />
      </mesh>

      {/* Target ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]} scale={targetAccuracy + 0.5}>
        <torusGeometry args={[1, 0.02, 16, 100]} />
        <meshBasicMaterial color="#3b82f6" transparent opacity={0.5} />
      </mesh>

      {/* Accuracy label */}
      <Html center position={[0, -2, 0]}>
        <div
          style={{
            background: 'rgba(0, 0, 0, 0.8)',
            padding: '8px 16px',
            borderRadius: 8,
            color: 'white',
            fontSize: 18,
            fontWeight: 600,
            fontFamily: 'monospace',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 4 }}>Predicted Accuracy</div>
          <div style={{ color }}>{((animatedAccuracy as number) * 100).toFixed(1)}%</div>
        </div>
      </Html>
    </group>
  );
}

// ============================================================================
// Parameter Ring
// ============================================================================

interface ParameterRingProps {
  parameter: ModelParameter;
  angle: number;
  radius: number;
  onValueChange: (value: number) => void;
  isActive: boolean;
  onActivate: () => void;
}

function ParameterRing({
  parameter,
  angle,
  radius,
  onValueChange: _onValueChange,
  isActive,
  onActivate,
}: ParameterRingProps): React.ReactElement {
  const groupRef = useRef<THREE.Group>(null);

  // Position on circle
  const position: [number, number, number] = [
    Math.cos(angle) * radius,
    0,
    Math.sin(angle) * radius,
  ];

  // Normalized value
  const normalizedValue = (parameter.value - parameter.min) / (parameter.max - parameter.min);

  // Ring color
  const ringColor = isActive ? '#3b82f6' : '#64748b';
  const fillColor = `hsl(${normalizedValue * 120}, 70%, 50%)`; // Green to red

  return (
    <group ref={groupRef} position={position}>
      {/* Background ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.4, 0.5, 32]} />
        <meshBasicMaterial color="#1e293b" side={THREE.DoubleSide} />
      </mesh>

      {/* Value arc */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.42, 0.48, 32, 1, 0, normalizedValue * Math.PI * 2]} />
        <meshBasicMaterial color={fillColor} side={THREE.DoubleSide} />
      </mesh>

      {/* Border ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.45, 0.02, 16, 32]} />
        <meshBasicMaterial color={ringColor} />
      </mesh>

      {/* Interactive hit area */}
      <mesh
        onClick={(e) => {
          e.stopPropagation();
          onActivate();
        }}
        visible={false}
      >
        <cylinderGeometry args={[0.6, 0.6, 0.2, 32]} />
        <meshBasicMaterial transparent opacity={0} />
      </mesh>

      {/* Label */}
      <Html center position={[0, 0.8, 0]}>
        <div
          style={{
            background: isActive ? 'rgba(59, 130, 246, 0.9)' : 'rgba(0, 0, 0, 0.8)',
            padding: '6px 12px',
            borderRadius: 6,
            color: 'white',
            fontSize: 11,
            fontFamily: 'sans-serif',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'background 0.2s',
          }}
          onClick={() => onActivate()}
        >
          <div style={{ fontWeight: 600 }}>{parameter.name}</div>
          <div style={{ fontSize: 14, marginTop: 2 }}>
            {parameter.value.toFixed(parameter.step < 1 ? 2 : 0)}
            {parameter.unit && ` ${parameter.unit}`}
          </div>
        </div>
      </Html>
    </group>
  );
}

// ============================================================================
// Scene Content
// ============================================================================

interface SceneContentProps {
  parameters: ModelParameter[];
  onParameterChange?: (parameterId: string, value: number) => void;
  predictedAccuracy: number;
  activeParameter: string | null;
  setActiveParameter: (id: string | null) => void;
  quality: 'high' | 'medium' | 'low';
}

function SceneContent({
  parameters,
  onParameterChange,
  predictedAccuracy,
  activeParameter,
  setActiveParameter,
  quality,
}: SceneContentProps): React.ReactElement {
  // Performance monitoring
  usePerformanceMonitor({
    targetFps: quality === 'low' ? 30 : 60,
    autoAdjust: true,
  });

  const ringRadius = 3;

  return (
    <>
      {/* Camera */}
      <PerspectiveCamera makeDefault position={[0, 5, 8]} fov={50} />

      {/* Controls */}
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={15}
        maxPolarAngle={Math.PI / 2}
        target={[0, 0, 0]}
      />

      {/* Lighting */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, 5, -10]} intensity={0.5} color="#3b82f6" />

      {/* Environment */}
      <Environment preset="night" />

      {/* Accuracy preview */}
      <AccuracyPreview accuracy={predictedAccuracy} targetAccuracy={0.75} />

      {/* Parameter rings */}
      {parameters.map((param, index) => {
        const angle = (index / parameters.length) * Math.PI * 2 - Math.PI / 2;
        return (
          <ParameterRing
            key={param.id}
            parameter={param}
            angle={angle}
            radius={ringRadius}
            onValueChange={(value) => onParameterChange?.(param.id, value)}
            isActive={activeParameter === param.id}
            onActivate={() => setActiveParameter(activeParameter === param.id ? null : param.id)}
          />
        );
      })}

      {/* Connecting lines */}
      {parameters.map((param, index) => {
        const angle = (index / parameters.length) * Math.PI * 2 - Math.PI / 2;
        const x = Math.cos(angle) * ringRadius;
        const z = Math.sin(angle) * ringRadius;
        return (
          <line key={`line-${param.id}`}>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([0, 0, 0, x, 0, z])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial
              color={activeParameter === param.id ? '#3b82f6' : '#334155'}
              opacity={0.5}
              transparent
            />
          </line>
        );
      })}

      {/* Ground grid */}
      <gridHelper args={[20, 20, '#1e293b', '#0f172a']} position={[0, -0.5, 0]} />
    </>
  );
}

// ============================================================================
// Parameter Panel
// ============================================================================

interface ParameterPanelProps {
  parameters: ModelParameter[];
  onParameterChange: (parameterId: string, value: number) => void;
  activeParameter: string | null;
  setActiveParameter: (id: string | null) => void;
}

function ParameterPanel({
  parameters,
  onParameterChange,
  activeParameter,
  setActiveParameter,
}: ParameterPanelProps): React.ReactElement {
  return (
    <div
      style={{
        position: 'absolute',
        top: 16,
        left: 16,
        background: 'rgba(0, 0, 0, 0.85)',
        padding: 16,
        borderRadius: 12,
        color: 'white',
        width: 280,
        maxHeight: 'calc(100% - 32px)',
        overflow: 'auto',
      }}
    >
      <h3 style={{ margin: 0, marginBottom: 16, fontSize: 14, fontWeight: 600 }}>
        Model Parameters
      </h3>

      {parameters.map((param) => (
        <div
          key={param.id}
          style={{
            marginBottom: 16,
            padding: 12,
            background: activeParameter === param.id ? 'rgba(59, 130, 246, 0.2)' : 'rgba(255, 255, 255, 0.05)',
            borderRadius: 8,
            border: activeParameter === param.id ? '1px solid #3b82f6' : '1px solid transparent',
            cursor: 'pointer',
          }}
          onClick={() => setActiveParameter(activeParameter === param.id ? null : param.id)}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontWeight: 500 }}>{param.name}</span>
            <span style={{ fontFamily: 'monospace' }}>
              {param.value.toFixed(param.step < 1 ? 2 : 0)}
              {param.unit && ` ${param.unit}`}
            </span>
          </div>

          <input
            type="range"
            min={param.min}
            max={param.max}
            step={param.step}
            value={param.value}
            onChange={(e) => onParameterChange(param.id, parseFloat(e.target.value))}
            onClick={(e) => e.stopPropagation()}
            style={{
              width: '100%',
              accentColor: '#3b82f6',
            }}
          />

          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 10,
              opacity: 0.6,
              marginTop: 4,
            }}
          >
            <span>{param.min}</span>
            <span>{param.max}</span>
          </div>

          {param.description && (
            <p style={{ fontSize: 11, opacity: 0.7, margin: '8px 0 0' }}>
              {param.description}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function ModelConfigurator({
  parameters: initialParameters,
  onParameterChange,
  getPredictedAccuracy,
  showPanel = true,
  quality = 'high',
  className,
}: ModelConfiguratorProps): React.ReactElement {
  const [parameters, setParameters] = useState(initialParameters);
  const [activeParameter, setActiveParameter] = useState<string | null>(null);

  // Update parameters when props change
  useEffect(() => {
    setParameters(initialParameters);
  }, [initialParameters]);

  // Handle parameter change
  const handleParameterChange = useCallback(
    (parameterId: string, value: number) => {
      setParameters((prev) =>
        prev.map((p) => (p.id === parameterId ? { ...p, value } : p))
      );
      onParameterChange?.(parameterId, value);
    },
    [onParameterChange]
  );

  // Calculate predicted accuracy
  const predictedAccuracy = useMemo(() => {
    if (getPredictedAccuracy) {
      const paramsMap = parameters.reduce(
        (acc, p) => ({ ...acc, [p.id]: p.value }),
        {} as Record<string, number>
      );
      return getPredictedAccuracy(paramsMap);
    }
    // Default mock prediction based on parameter values
    const avgNormalized =
      parameters.reduce((acc, p) => {
        return acc + (p.value - p.min) / (p.max - p.min);
      }, 0) / parameters.length;
    return 0.6 + avgNormalized * 0.35; // Range 0.6 - 0.95
  }, [parameters, getPredictedAccuracy]);

  // Check WebGL support
  if (!isWebGLSupported()) {
    return (
      <SceneFallback description="Interactive model configurator with parameter tuning preview" />
    );
  }

  const canvasProps = getCanvasPropsForPreset('model-configurator');

  return (
    <div className={className} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        {...canvasProps}
        style={{ background: 'linear-gradient(180deg, #0f172a 0%, #1e293b 100%)' }}
      >
        <A11yProvider>
          <A11yCanvas sceneDescription="Interactive model parameter configurator with accuracy preview">
            <KeyboardControls />
            <SceneContent
              parameters={parameters}
              onParameterChange={handleParameterChange}
              predictedAccuracy={predictedAccuracy}
              activeParameter={activeParameter}
              setActiveParameter={setActiveParameter}
              quality={quality}
            />
          </A11yCanvas>
        </A11yProvider>
      </Canvas>

      {/* Parameter panel */}
      {showPanel && (
        <ParameterPanel
          parameters={parameters}
          onParameterChange={handleParameterChange}
          activeParameter={activeParameter}
          setActiveParameter={setActiveParameter}
        />
      )}

      {/* Actions */}
      <div
        style={{
          position: 'absolute',
          bottom: 16,
          right: 16,
          display: 'flex',
          gap: 8,
        }}
      >
        <button
          onClick={() => setParameters(initialParameters)}
          style={{
            padding: '8px 16px',
            background: '#334155',
            border: 'none',
            borderRadius: 6,
            color: 'white',
            cursor: 'pointer',
            fontSize: 13,
          }}
        >
          Reset
        </button>
        <button
          style={{
            padding: '8px 16px',
            background: '#3b82f6',
            border: 'none',
            borderRadius: 6,
            color: 'white',
            cursor: 'pointer',
            fontSize: 13,
            fontWeight: 500,
          }}
        >
          Save Configuration
        </button>
      </div>
    </div>
  );
}

export default ModelConfigurator;
