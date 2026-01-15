import { http, HttpResponse } from 'msw';
import type {
  Alert,
  AuthTokens, 
  FederatedStatus,
  HealthCheckResponse, 
  Metric, 
  PaginatedResponse,
  QueueItem,
  RecalibrationResponse,
  RoundDetails,
  Task,
  User,
  ALConfig,
  ALStatus,
  ClientParticipation,
} from '../types/api';
import { mockSettings } from '../lib/mock-data';

// Deterministic, stable mock data for predictable rendering/tests.
const now = new Date('2026-01-01T12:00:00.000Z');

const mockUser: User = {
  id: 'user-123',
  email: 'demo@example.com',
  name: 'Demo User',
  role: 'annotator',
  createdAt: now.toISOString(),
  updatedAt: now.toISOString(),
};

const mockTokens: AuthTokens = {
  accessToken: 'mock-access-token',
  refreshToken: 'mock-refresh-token',
  expiresIn: 3600,
  tokenType: 'Bearer',
};

const mockTasks: Task[] = [
  {
    id: '00000000-0000-0000-0000-000000000001',
  type: 'pairwise',
  content: {
    prompt: 'Which response is more helpful?',
    response_a: 'Response A is concise.',
    response_b: 'Response B is more detailed but verbose.',
  },
    created_at: now.toISOString(),
    priority: 1,
    assigned_to: mockUser.id,
    assigned_at: now.toISOString(),
    status: 'assigned',
  },
  {
    id: '00000000-0000-0000-0000-000000000002',
    type: 'critique',
    content: {
      prompt: 'Provide a critique of the following response.',
      response: 'This response is generally helpful but misses key edge cases.',
    },
    created_at: now.toISOString(),
    priority: 1,
    assigned_to: mockUser.id,
    assigned_at: now.toISOString(),
    status: 'assigned',
  },
  {
    id: '00000000-0000-0000-0000-000000000003',
    type: 'likert',
    content: {
      prompt: 'Rate the response quality on a 1-5 scale.',
      response: 'This response is accurate and clear.',
    },
    created_at: now.toISOString(),
    priority: 1,
    assigned_to: mockUser.id,
    assigned_at: now.toISOString(),
    status: 'assigned',
  },
  {
    id: '00000000-0000-0000-0000-000000000004',
    type: 'ranking',
    content: {
      prompt: 'Rank the responses from best to worst.',
      responses: [
        'Response 1: concise and correct.',
        'Response 2: verbose but helpful.',
        'Response 3: partially incorrect.',
      ],
    },
    created_at: now.toISOString(),
  priority: 1,
    assigned_to: mockUser.id,
    assigned_at: now.toISOString(),
  status: 'assigned',
  },
];

const mockHealth: HealthCheckResponse = {
  encoder: 'healthy',
  dpo: 'healthy',
  monitoring: 'healthy',
  privacy: 'healthy',
};

const mockAlerts: Alert[] = [
  {
    id: 'alert-1',
    rule_name: 'High CPU',
    severity: 'warning',
    status: 'firing',
    timestamp: now.toISOString(),
    message: 'CPU usage > 80%',
  },
  {
    id: 'alert-2',
    rule_name: 'High Error Rate',
    severity: 'critical',
    status: 'acknowledged',
    timestamp: new Date(now.getTime() - 5 * 60 * 1000).toISOString(),
    message: '5xx error rate > 2%',
  },
  {
    id: 'alert-3',
    rule_name: 'Queue Depth',
    severity: 'info',
    status: 'resolved',
    timestamp: new Date(now.getTime() - 12 * 60 * 1000).toISOString(),
    message: 'Annotation queue depth stabilized',
  },
];

function makeMetrics(name: string): Metric[] {
  const base = now.getTime() - 30 * 60 * 1000;
  const values =
    name === 'model_accuracy'
      ? [0.71, 0.73, 0.74, 0.745, 0.748]
      : name === 'error_rate'
        ? [0.02, 0.021, 0.018, 0.019, 0.017]
        : name === 'ece'
          ? [0.12, 0.11, 0.095, 0.09, 0.085]
          : name === 'annotation_queue_depth'
            ? [120, 115, 108, 104, 99]
            : [180, 165, 155, 150, 145]; // encoder_latency

  return values.map((v, i) => ({
    name,
    value: v,
    timestamp: new Date(base + i * 5 * 60 * 1000).toISOString(),
    tags: { service: 'api' },
  }));
}

const mockRecalibration: RecalibrationResponse = {
  temperature: 1.2345,
  pre_ece: 0.121,
  post_ece: 0.078,
  iterations: 42,
};

const mockFederatedStatus: FederatedStatus = {
  round: 42,
  isActive: true,
  totalClients: 25,
  activeClients: 19,
  privacyBudget: {
    epsilonSpent: 4.2,
    epsilonRemaining: 1.8,
    delta: 1e-6,
    totalSteps: 1000,
  },
  modelChecksum: 'mock-checksum-123',
  lastUpdated: now.toISOString(),
};

const mockRounds: RoundDetails[] = [
  {
    roundId: 42,
    startedAt: now.toISOString(),
    completedAt: null,
    participatingClients: 18,
    selectedClients: 20,
    gradientStats: { meanNorm: 0.12, maxNorm: 0.78, noiseScale: 0.08 },
    status: 'in_progress',
  },
  {
    roundId: 41,
    startedAt: new Date(now.getTime() - 30 * 60 * 1000).toISOString(),
    completedAt: new Date(now.getTime() - 22 * 60 * 1000).toISOString(),
    participatingClients: 20,
    selectedClients: 20,
    gradientStats: { meanNorm: 0.15, maxNorm: 0.81, noiseScale: 0.08 },
    status: 'completed',
  },
];

const mockClients: ClientParticipation[] = Array.from({ length: 12 }, (_, i) => ({
  clientId: `client-${String(i + 1).padStart(2, '0')}`,
  rounds: [42, 41].filter((r) => (i + r) % 2 === 0),
  lastSeen: new Date(now.getTime() - i * 60 * 1000).toISOString(),
  status: i < 8 ? 'active' : i < 10 ? 'straggler' : 'offline',
}));

const mockALConfig: ALConfig = {
  budget: 1000,
  batch_size: 10,
  seed_size: 50,
  strategy: 'uncertainty',
};

const mockALStatus: ALStatus = {
  labeledCount: 320,
  unlabeledCount: 1420,
  budgetRemaining: 680,
  currentStrategy: 'uncertainty',
  lastUpdated: now.toISOString(),
};

const mockALQueue: QueueItem[] = Array.from({ length: 20 }, (_, i) => ({
  id: `queue-${i + 1}`,
  text: `Queue item text ${i + 1}: evaluate a short response for helpfulness.`,
  uncertaintyScore: 0.9 - i * 0.02,
  diversityScore: 0.5 + (i % 5) * 0.08,
  iidScore: 0.7 - (i % 3) * 0.05,
  compositeRank: i + 1,
  createdAt: new Date(now.getTime() - i * 2 * 60 * 1000).toISOString(),
}));

export const handlers = [
  // ==========================================================================
  // Auth
  // ==========================================================================
  http.get('/api/auth/dev-status', () => {
    return HttpResponse.json({ devMode: true });
  }),

  http.post('/api/auth/login', async () => {
    return HttpResponse.json(mockTokens);
  }),

  http.post('/api/auth/register', async () => {
    return HttpResponse.json(mockTokens);
  }),

  http.post('/api/auth/refresh', async () => {
    return HttpResponse.json(mockTokens);
  }),

  http.post('/api/auth/dev-login', async () => {
    return HttpResponse.json(mockTokens);
  }),

  http.get('/api/auth/me', () => {
    return HttpResponse.json(mockUser);
  }),

  http.post('/api/auth/logout', () => {
    return new HttpResponse(null, { status: 200 });
  }),

  // ==========================================================================
  // Tasks / Annotations
  // ==========================================================================
  http.get('/api/tasks/next', ({ request }) => {
    const url = new URL(request.url);
    const annotatorId = url.searchParams.get('annotator_id') || 'unknown';
    const idx = annotatorId.length % mockTasks.length;
    return HttpResponse.json(mockTasks[idx] ?? mockTasks[0]);
  }),

  http.post('/api/annotations', async () => {
    return HttpResponse.json({ status: 'success', id: 'annotation-123' });
  }),

  http.get('/api/annotations', ({ request }) => {
    const url = new URL(request.url);
    const page = Number(url.searchParams.get('page') || '1');
    const pageSize = Number(url.searchParams.get('page_size') || '10');

    const total = 25;
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    const safePage = Math.min(Math.max(page, 1), totalPages);
    const start = (safePage - 1) * pageSize;
    const end = Math.min(total, start + pageSize);

    const data = Array.from({ length: Math.max(0, end - start) }, (_, i) => {
      const n = start + i + 1;
      return {
        id: `annotation-${n}`,
        task_id: mockTasks[n % mockTasks.length]!.id,
        annotator_id: mockUser.id,
        annotation_type: 'pairwise',
        response_data: { winner: n % 2 === 0 ? 'A' : 'B' },
        time_spent_seconds: 12 + (n % 10),
        confidence: (n % 5) + 1,
        created_at: new Date(now.getTime() - n * 60 * 1000).toISOString(),
      };
    });

    const response: PaginatedResponse<Record<string, unknown>> = {
      data,
      meta: {
        page: safePage,
        pageSize,
        total,
        totalPages,
        hasNext: safePage < totalPages,
        hasPrev: safePage > 1,
      },
    };

    return HttpResponse.json(response);
  }),

  http.get('/api/quality/metrics', ({ request }) => {
    const url = new URL(request.url);
    const annotatorId = url.searchParams.get('annotator_id') || mockUser.id;
    return HttpResponse.json({
      annotator_id: annotatorId,
      agreement_score: 0.85,
      gold_pass_rate: 0.92,
      avg_time_per_task: 45,
      flags: [],
    });
  }),

  // ==========================================================================
  // Health / Monitoring
  // ==========================================================================
  http.get('/api/health', () => {
    return HttpResponse.json(mockHealth);
  }),

  http.get('/api/metrics', ({ request }) => {
    const url = new URL(request.url);
    const name = url.searchParams.get('name') || 'encoder_latency';
    return HttpResponse.json(makeMetrics(name));
  }),

  http.get('/api/alerts', () => {
    return HttpResponse.json(mockAlerts);
  }),

  http.post('/api/alerts/:id/ack', ({ params }) => {
    const id = String(params.id);
    const exists = mockAlerts.some((a) => a.id === id);
    if (!exists) {
      return HttpResponse.json(
        { detail: 'Alert not found' },
        { status: 404 }
      );
    }
    return HttpResponse.json({ status: 'acknowledged' });
  }),

  // ==========================================================================
  // Calibration
  // ==========================================================================
  http.post('/api/calibration/recalibrate', async () => {
    return HttpResponse.json(mockRecalibration);
  }),

  http.post('/api/calibration/predictions', async () => {
    return HttpResponse.json({ sampled: true });
  }),

  // ==========================================================================
  // Settings
  // ==========================================================================
  http.get('/api/settings', () => {
    return HttpResponse.json(mockSettings);
  }),

  http.put('/api/settings', async ({ request }) => {
    const body = (await request.json().catch(() => ({}))) as Partial<typeof mockSettings>;
    const next = { ...mockSettings, ...body };
    return HttpResponse.json(next);
  }),

  // ==========================================================================
  // Federated Learning
  // ==========================================================================
  http.get('/api/federated/status', () => {
    return HttpResponse.json(mockFederatedStatus);
  }),

  http.get('/api/federated/rounds', () => {
    return HttpResponse.json(mockRounds);
  }),

  http.get('/api/federated/clients', () => {
    return HttpResponse.json(mockClients);
  }),

  http.post('/api/federated/start', () => {
    return HttpResponse.json({ status: 'started' });
  }),

  http.post('/api/federated/pause', () => {
    return HttpResponse.json({ status: 'paused' });
  }),

  // ==========================================================================
  // Active Learning
  // ==========================================================================
  http.get('/api/active-learning/config', () => {
    return HttpResponse.json(mockALConfig);
  }),

  http.patch('/api/active-learning/config', async ({ request }) => {
    const body = (await request.json().catch(() => ({}))) as Partial<ALConfig>;
    const next: ALConfig = { ...mockALConfig, ...body };
    return HttpResponse.json(next);
  }),

  http.get('/api/active-learning/status', () => {
    return HttpResponse.json(mockALStatus);
  }),

  http.get('/api/active-learning/queue', () => {
    return HttpResponse.json(mockALQueue);
  }),

  http.post('/api/active-learning/refresh', () => {
    return HttpResponse.json({ status: 'refreshed' });
  }),

  // ==========================================================================
  // Playground
  // ==========================================================================
  http.post('/api/predict', async () => {
    return HttpResponse.json({
        probabilities: [0.1, 0.7, 0.2],
        action_index: 1,
      confidence: 0.85,
    });
  }),
];
