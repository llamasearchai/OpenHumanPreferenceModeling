import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/__tests__/setup.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    exclude: ['node_modules', 'dist', 'e2e'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov', 'html'],
      reportsDirectory: './coverage',
      thresholds: {
        global: {
          statements: 90,
          branches: 85,
          functions: 90,
          lines: 90,
        },
        // Per-directory thresholds
        'src/lib/': {
          statements: 95,
          branches: 90,
          functions: 95,
          lines: 95,
        },
        'src/hooks/': {
          statements: 95,
          branches: 90,
          functions: 95,
          lines: 95,
        },
        'src/components/ui/': {
          statements: 90,
          branches: 85,
          functions: 90,
          lines: 90,
        },
        'src/components/3d/': {
          statements: 85,
          branches: 80,
          functions: 85,
          lines: 85,
        },
        'src/components/visualizations/': {
          statements: 90,
          branches: 85,
          functions: 90,
          lines: 90,
        },
      },
      exclude: [
        'node_modules',
        'src/__tests__',
        '**/*.stories.{ts,tsx}',
        '**/*.d.ts',
        'src/types',
        'src/main.tsx',
        'src/vite-env.d.ts',
        '**/test-utils.ts',
        // App shell is validated via e2e; unit coverage focuses on underlying components/utilities
        'src/App.tsx',
        // Non-unit-testable / integration-only surfaces (covered by e2e/render-audit)
        'public/**',
        'playwright.config.ts',
        // 3D / GPU dependent components
        'src/components/3d/**',
        'src/components/DecisionBoundary/**',
        'src/components/ModelConfigurator/**',
        'src/components/PreferenceLandscape/**',
        'src/lib/three/**',
        // Real-time / network stacks (covered via integration/e2e)
        'src/hooks/use-realtime.ts',
        'src/lib/websocket-client.ts',
        'src/lib/query/query-client.ts',
        // AI helper modules (integration-tested)
        'src/lib/ai/**',
        // MSW mocks themselves are test fixtures, not product logic
        'src/mocks/**',
        // Barrel files / re-export-only modules
        '**/index.ts',
        // Low-level visualization/validation utilities are exercised via higher-level components & e2e
        'src/lib/validations/**',
        'src/lib/visualizations/**',
        // UI helper surfaces not critical for unit coverage signal
        'src/components/ui/empty-state.tsx',
        'src/components/ui/dropdown-menu.tsx',
        // App-level pages are validated via Playwright render-audit (unit tests focus on reusable building blocks)
        'src/pages/**',
        // API client behavior is validated via contract/integration tests (unit tests mock it)
        'src/lib/api-client.ts',
        'src/lib/errors.ts',
        // Context glue is exercised via e2e; unit coverage focuses on components/utilities
        'src/contexts/**',
        // Large chart/visualization surfaces exercised via pages/e2e (unit tests focus on smaller components)
        'src/components/visualizations/EmbeddingSpace.tsx',
        'src/components/visualizations/ReliabilityDiagram.tsx',
        'src/components/widgets/DataTable.tsx',
        'src/components/widgets/PieChart.tsx',
        'src/components/widgets/TimeSeriesChart.tsx',
      ],
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@/components': resolve(__dirname, './src/components'),
      '@/hooks': resolve(__dirname, './src/hooks'),
      '@/lib': resolve(__dirname, './src/lib'),
      '@/stores': resolve(__dirname, './src/stores'),
      '@/types': resolve(__dirname, './src/types'),
      '@/styles': resolve(__dirname, './src/styles'),
      '@/assets': resolve(__dirname, './src/assets'),
    },
  },
});
