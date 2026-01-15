import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright E2E Test Configuration
 * 
 * Purpose: Cross-browser end-to-end testing for critical user journeys
 * Covers: Authentication, CRUD operations, 3D interactions, visualizations
 * 
 * Design decisions:
 * - Parallel execution for faster CI runs
 * - Retries in CI to handle flaky tests
 * - Screenshot and video capture for debugging
 * - Accessibility testing with axe-core integration
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  ...(process.env.CI ? { workers: 4 } : {}),
  reporter: (() => {
    const base = [
      ['html', { outputFolder: 'playwright-report' }],
      ['list'],
    ] as const;

    const ci = process.env.CI ? ([['github', {}]] as const) : ([] as const);
    return [...base, ...ci];
  })(),
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 10000,
    navigationTimeout: 30000,
  },
  projects: [
    // Desktop browsers
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    // Mobile browsers
    {
      name: 'mobile-chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'mobile-safari',
      use: { ...devices['iPhone 13'] },
    },
  ],
  webServer: {
    command: 'pnpm dev --host 127.0.0.1 --port 5173 --strictPort',
    url: 'http://127.0.0.1:5173',
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
    env: {
      ...process.env,
      // Disable MSW mocks for E2E tests to rely on page.route
      VITE_USE_MOCKS: 'false',
      // Bypass auth entirely per user request
      VITE_AUTH_BYPASS: 'true',
    },
  },
  expect: {
    timeout: 10000,
    toHaveScreenshot: {
      maxDiffPixels: 100,
    },
  },
});
