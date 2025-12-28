/**
 * API Type Definitions
 * 
 * Purpose: Complete TypeScript interfaces for all API contracts extracted from
 * the OpenHumanPreferenceModeling backend services.
 * 
 * Sources:
 * - calibration/api.py - Calibration service endpoints
 * - annotation_interface/backend - Annotation task management
 * - monitoring_dashboard/backend - Metrics and alerts
 * - integration/system_architecture.py - System orchestration
 * 
 * Design decisions:
 * - All types match server-side Pydantic models
 * - Zod schemas provided for runtime validation
 * - Discriminated unions for error handling
 */

import { z } from 'zod';

// ============================================================================
// Common Types
// ============================================================================

/** Standard API error response following RFC 7807 Problem Details */
export interface ApiProblemDetail {
  type: string;
  title: string;
  status: number;
  detail: string;
  code: string;
}

/** Pagination metadata for list endpoints */
export interface PaginationMeta {
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

/** Generic paginated response wrapper */
export interface PaginatedResponse<T> {
  data: T[];
  meta: PaginationMeta;
}

// ============================================================================
// Authentication Types
// ============================================================================

export interface JWTPayload {
  sub: string;
  exp: number;
  iat: number;
  aud?: string;
  iss?: string;
  scope?: string | string[];
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  tokenType: 'Bearer';
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  name: string;
}

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'annotator' | 'viewer';
  createdAt: string;
  updatedAt: string;
}

// ============================================================================
// Annotation Interface Types
// ============================================================================

/** Task types supported by the annotation system */
export type TaskType = 'pairwise' | 'ranking' | 'critique' | 'likert';

/** Task status lifecycle */
export type TaskStatus = 'unassigned' | 'assigned' | 'completed';

/** Pairwise comparison response options */
export type PairwiseWinner = 'A' | 'B' | 'tie' | 'both_poor';

/** Task content for pairwise comparisons */
export interface PairwiseTaskContent {
  prompt: string;
  response_a: string;
  response_b: string;
}

/** Task content for ranking tasks */
export interface RankingTaskContent {
  prompt: string;
  responses: string[];
}

/** Generic task content union */
export type TaskContent = PairwiseTaskContent | RankingTaskContent | Record<string, unknown>;

/** Annotation task as returned by the API */
export interface Task {
  id: string;
  type: TaskType;
  content: TaskContent;
  created_at: string;
  priority: number;
  assigned_to: string | null;
  assigned_at: string | null;
  status: TaskStatus;
}

/** Pairwise response data */
export interface PairwiseResponse {
  winner: PairwiseWinner;
  rationale?: string;
}

/** Annotation submission request */
export interface AnnotationRequest {
  task_id: string;
  annotator_id: string;
  annotation_type: string;
  response_data: PairwiseResponse | Record<string, unknown>;
  time_spent_seconds: number;
  confidence: number; // 1-5
}

/** Annotation record as stored */
export interface Annotation {
  id: string;
  task_id: string;
  annotator_id: string;
  annotation_type: string;
  response_data: Record<string, unknown>;
  time_spent_seconds: number;
  confidence: number;
  created_at: string;
}

/** Annotator profile */
export interface Annotator {
  id: string;
  skill_level: number;
  total_annotations: number;
  accuracy: number;
  status: 'active' | 'probation' | 'suspended';
}

/** Quality metrics for an annotator */
export interface QualityMetrics {
  annotator_id: string;
  agreement_score: number;
  gold_pass_rate: number;
  avg_time_per_task: number;
  flags: string[];
}

// ============================================================================
// Calibration Types
// ============================================================================

/** Recalibration request parameters */
export interface RecalibrationRequest {
  validation_data_uri: string;
  target_ece?: number; // Default: 0.08
  max_iterations?: number; // Default: 100
}

/** Recalibration result */
export interface RecalibrationResponse {
  temperature: number;
  pre_ece: number;
  post_ece: number;
  iterations: number;
}

/** Prediction record for calibration monitoring */
export interface PredictionRecordRequest {
  confidence: number; // 0.0-1.0
  correct: 0 | 1;
}

export interface PredictionRecordResponse {
  sampled: boolean;
}

// ============================================================================
// Monitoring Types
// ============================================================================

/** Metric data point */
export interface Metric {
  name: string;
  value: number;
  timestamp: string;
  tags: Record<string, string>;
}

/** Alert configuration */
export interface AlertConfig {
  name: string;
  expr: string;
  severity: 'critical' | 'warning' | 'info';
  period_minutes: number;
  description: string;
}

/** Alert status */
export type AlertStatus = 'pending' | 'firing' | 'resolved' | 'acknowledged';

/** Active alert */
export interface Alert {
  id: string;
  rule_name: string;
  severity: string;
  status: AlertStatus;
  timestamp: string;
  message: string;
}

// ============================================================================
// DPO Pipeline Types
// ============================================================================

/** Preference pair for DPO training */
export interface PreferencePair {
  prompt: string;
  chosen: string;
  rejected: string;
  score_gap: number;
  source: 'synthetic' | 'human';
}

/** DPO model prediction request */
export interface PredictRequest {
  state_vector: number[];
}

/** DPO model prediction response */
export interface PredictResponse {
  probabilities: number[];
  action_index: number;
  confidence: number;
}

// ============================================================================
// User State Encoder Types
// ============================================================================

/** Multi-objective head outputs */
export interface ObjectiveScores {
  aesthetic: number;
  functional: number;
  cost: number;
  safety: number;
  gate_weights: number[];
  final_score: number;
}

/** User state encoding request */
export interface EncodeStateRequest {
  user_id: string;
  event_text: string;
}

/** User state encoding response */
export interface EncodeStateResponse {
  state_embedding: number[];
  objective_scores: ObjectiveScores;
}

// ============================================================================
// Privacy Types
// ============================================================================

/** Federated learning gradient upload */
export interface GradientUploadRequest {
  encrypted_grads: string; // Base64 encoded encrypted gradients
}

/** Privacy budget status */
export interface PrivacyBudgetStatus {
  epsilon_spent: number;
  epsilon_remaining: number;
  delta: number;
  total_steps: number;
}

// ============================================================================
// System Health Types
// ============================================================================

export interface HealthCheckResponse {
  encoder: 'healthy' | 'unhealthy';
  dpo: 'healthy' | 'unhealthy';
  monitoring: 'healthy' | 'unhealthy';
  privacy: 'healthy' | 'unhealthy';
}

// ============================================================================
// Zod Schemas for Runtime Validation
// ============================================================================

export const TaskSchema = z.object({
  id: z.string().uuid(),
  type: z.enum(['pairwise', 'ranking', 'critique', 'likert']),
  content: z.record(z.unknown()),
  created_at: z.string().datetime(),
  priority: z.number().min(0).max(1),
  assigned_to: z.string().nullable(),
  assigned_at: z.string().datetime().nullable(),
  status: z.enum(['unassigned', 'assigned', 'completed']),
});

export const AnnotationRequestSchema = z.object({
  task_id: z.string().uuid(),
  annotator_id: z.string(),
  annotation_type: z.string(),
  response_data: z.record(z.unknown()),
  time_spent_seconds: z.number().positive(),
  confidence: z.number().int().min(1).max(5),
});

export const RecalibrationRequestSchema = z.object({
  validation_data_uri: z.string().url(),
  target_ece: z.number().min(0).max(1).optional().default(0.08),
  max_iterations: z.number().int().min(1).max(1000).optional().default(100),
});

export const PredictionRecordRequestSchema = z.object({
  confidence: z.number().min(0).max(1),
  correct: z.union([z.literal(0), z.literal(1)]),
});

export const MetricSchema = z.object({
  name: z.string(),
  value: z.number(),
  timestamp: z.string().datetime(),
  tags: z.record(z.string()),
});

export const AlertSchema = z.object({
  id: z.string().uuid(),
  rule_name: z.string(),
  severity: z.string(),
  status: z.enum(['pending', 'firing', 'resolved', 'acknowledged']),
  timestamp: z.string().datetime(),
  message: z.string(),
});

// ============================================================================
// API Response Type Helpers
// ============================================================================

/** Discriminated union for API responses with error handling */
export type ApiResult<T> =
  | { success: true; data: T }
  | { success: false; error: ApiProblemDetail };

/** Helper type for async API operations */
export type AsyncApiResult<T> = Promise<ApiResult<T>>;

// ============================================================================
// WebSocket Message Types
// ============================================================================

export interface WebSocketMessage<T = unknown> {
  type: string;
  payload: T;
  timestamp: string;
}

export interface MetricUpdateMessage {
  type: 'metric_update';
  payload: Metric;
}

export interface AlertUpdateMessage {
  type: 'alert_update';
  payload: Alert;
}

export interface TaskAssignedMessage {
  type: 'task_assigned';
  payload: Task;
}

export type RealtimeMessage = MetricUpdateMessage | AlertUpdateMessage | TaskAssignedMessage;

// ============================================================================
// Query Key Types for TanStack Query
// ============================================================================

export const queryKeys = {
  auth: {
    all: ['auth'] as const,
    user: () => [...queryKeys.auth.all, 'user'] as const,
  },
  tasks: {
    all: ['tasks'] as const,
    list: (filters: Record<string, unknown>) => [...queryKeys.tasks.all, 'list', filters] as const,
    detail: (id: string) => [...queryKeys.tasks.all, 'detail', id] as const,
    next: (userId: string) => [...queryKeys.tasks.all, 'next', userId] as const,
  },
  annotations: {
    all: ['annotations'] as const,
    list: (filters: Record<string, unknown>) => [...queryKeys.annotations.all, 'list', filters] as const,
    quality: (annotatorId: string) => [...queryKeys.annotations.all, 'quality', annotatorId] as const,
  },
  metrics: {
    all: ['metrics'] as const,
    byName: (name: string) => [...queryKeys.metrics.all, name] as const,
  },
  alerts: {
    all: ['alerts'] as const,
    active: () => [...queryKeys.alerts.all, 'active'] as const,
  },
  calibration: {
    all: ['calibration'] as const,
    status: () => [...queryKeys.calibration.all, 'status'] as const,
  },
} as const;
