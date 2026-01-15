import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

type RouteAudit = {
  path: string;
  name: string;
  requiresAuth: boolean;
  expectHeading?: RegExp;
  /**
   * Optional heading level assertion. Use this for pages that should have a true
   * page-level <h1>. Public auth pages often use CardTitle (h3).
   */
  expectHeadingLevel?: number;
  run?: (page: import('@playwright/test').Page) => Promise<void>;
};

function slugify(value: string): string {
  return value
    .replace(/^\//, '')
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .toLowerCase() || 'root';
}

async function waitForAppToRender(page: import('@playwright/test').Page) {
  await page.waitForLoadState('domcontentloaded');
  await page.waitForFunction(() => {
    const el = document.getElementById('root');
    return !!el && (el.textContent?.trim().length ?? 0) > 0;
  });
}

function installConsoleGuards(page: import('@playwright/test').Page) {
  const errors: string[] = [];
  const warnings: string[] = [];

  page.on('pageerror', (err) => {
    errors.push(`pageerror: ${err.message}`);
  });

  page.on('console', (msg) => {
    const text = msg.text();

    // Filter out a few known non-actionable messages.
    if (text.includes('Download the React DevTools')) {
      return;
    }

    if (msg.type() === 'error') {
      errors.push(`console.error: ${text}`);
      return;
    }

    if (msg.type() === 'warning') {
      warnings.push(`console.warn: ${text}`);
    }
  });

  return { errors, warnings };
}

async function runA11ySweep(page: import('@playwright/test').Page) {
  const results = await new AxeBuilder({ page })
    // Color contrast can be theme-dependent and often requires design tweaks;
    // treat it as non-blocking for this audit.
    .disableRules(['color-contrast'])
    .analyze();

  const blocking = results.violations.filter((v) =>
    v.impact === 'critical' || v.impact === 'serious'
  );

  return { results, blocking };
}

async function clickThroughTabs(page: import('@playwright/test').Page) {
  const tabs = page.getByRole('tab');
  const count = await tabs.count();
  if (count <= 1) return;

  for (let i = 0; i < count; i++) {
    const tab = tabs.nth(i);
    if (!(await tab.isVisible())) continue;
    await tab.click();
    // Give Radix Tabs a moment to mount/unmount panels.
    await page.waitForTimeout(50);
  }
}

const routes: RouteAudit[] = [
  // Public routes (auth removed)
  {
    path: '/unknown-page',
    name: 'Not Found',
    requiresAuth: false,
    expectHeading: /^404$/i,
    expectHeadingLevel: 1,
  },

  // App routes
  { path: '/', name: 'Dashboard', requiresAuth: false, expectHeading: /dashboard/i, expectHeadingLevel: 1, run: clickThroughTabs },
  { path: '/annotations', name: 'Annotations', requiresAuth: false, expectHeading: /annotations/i, expectHeadingLevel: 1, run: async (page) => {
      await clickThroughTabs(page);
      await expect(
        page.getByRole('application', { name: /3d embedding space visualization/i })
      ).toBeVisible();
    } },
  { path: '/metrics', name: 'Metrics', requiresAuth: false, expectHeading: /metrics/i, expectHeadingLevel: 1 },
  { path: '/alerts', name: 'Alerts', requiresAuth: false, expectHeading: /alerts/i, expectHeadingLevel: 1 },
  { path: '/calibration', name: 'Calibration', requiresAuth: false, expectHeading: /calibration/i, expectHeadingLevel: 1, run: async (page) => {
      // Exercise the mutation path with valid URL input to ensure the results panel renders.
      await page.getByLabel(/validation data uri/i).fill('https://example.com/validation_data.npz');
      await page.getByRole('button', { name: /start recalibration/i }).click();
      await expect(page.getByText(/calibration successful/i)).toBeVisible();
      await expect(page.getByRole('heading', { name: /confidence distribution/i })).toBeVisible();
      await expect(page.getByRole('heading', { name: /reliability diagram/i })).toBeVisible();
    } },
  { path: '/active-learning', name: 'Active Learning', requiresAuth: false, expectHeading: /active learning/i, expectHeadingLevel: 1, run: clickThroughTabs },
  { path: '/federated-learning', name: 'Federated Learning', requiresAuth: false, expectHeading: /federated learning/i, expectHeadingLevel: 1, run: clickThroughTabs },
  { path: '/quality-control', name: 'Quality Control', requiresAuth: false, expectHeading: /quality control/i, expectHeadingLevel: 1, run: clickThroughTabs },
  { path: '/training', name: 'Training', requiresAuth: false, expectHeading: /training/i, expectHeadingLevel: 1, run: clickThroughTabs },
  { path: '/playground', name: 'Playground', requiresAuth: false, expectHeading: /model playground/i, expectHeadingLevel: 1 },
  { path: '/settings', name: 'Settings', requiresAuth: false, expectHeading: /settings/i, expectHeadingLevel: 1 },
];

test.describe('Render Audit', () => {
  test.beforeEach(async ({ page }) => {
    // Auth has been removed; ensure no stale auth state leaks into tests.
    await page.context().addInitScript(() => {
      window.localStorage.removeItem('refreshToken');
    });

    // Mock API dependencies to ensure render audits verify frontend resilience
    // independent of backend state.
    await page.route('**/api/health', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          data: {
            success: true,
            encoder: 'healthy',
            dpo: 'healthy',
            monitoring: 'healthy',
            privacy: 'healthy',
          },
        }),
      });
    });

    await page.route('**/api/alerts', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          data: [],
        }),
      });
    });

    await page.route('**/api/metrics*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          data: [],
        }),
      });
    });

    await page.route('**/api/annotations*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          data: { data: [] },
        }),
      });
    });
  });

  for (const route of routes) {
    const titlePrefix = route.requiresAuth ? '[auth]' : '[public]';
    test(`${titlePrefix} ${route.name} renders and passes critique checks`, async ({ page }, testInfo) => {
      const { errors, warnings } = installConsoleGuards(page);

      await page.goto(route.path);
      await waitForAppToRender(page);

      if (route.expectHeading) {
        const locator = route.expectHeadingLevel
          ? page.getByRole('heading', { level: route.expectHeadingLevel, name: route.expectHeading })
          : page.getByRole('heading', { name: route.expectHeading });
        await expect(locator).toBeVisible();
      }

      // If a protected route redirects to login, treat it as a failure.
      if (route.requiresAuth) {
        await expect(page).not.toHaveURL(/\/login$/);
      }

      if (route.run) {
        await route.run(page);
      }

      // A11y sweep + attach report for triage.
      const { results, blocking } = await runA11ySweep(page);
      await testInfo.attach('axe-violations.json', {
        body: JSON.stringify(results.violations, null, 2),
        contentType: 'application/json',
      });
      expect(blocking, `Blocking a11y violations on ${route.path}`).toEqual([]);

      // Screenshot for human review.
      await testInfo.attach(`screenshot-${slugify(route.path)}.png`, {
        body: await page.screenshot({ fullPage: true }),
        contentType: 'image/png',
      });

      // Console/page errors are hard failures.
      if (warnings.length > 0) {
        await testInfo.attach('console-warnings.txt', {
          body: warnings.join('\n'),
          contentType: 'text/plain',
        });
      }

      expect(errors, `Console/page errors on ${route.path}`).toEqual([]);
    });
  }
});

