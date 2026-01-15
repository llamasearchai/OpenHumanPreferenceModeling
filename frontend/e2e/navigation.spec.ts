/**
 * Navigation E2E Tests
 */

import { test, expect } from '@playwright/test';
import { setupGlobalMocks } from './test-utils';

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    // Auth is removed, so we just setup default mocks (unauthenticated by default implies no errors, 
    // but globally we might want 'authenticated: true' simulated mock data if components require it, 
    // but new Dashboard uses custom-data.ts so no API calls needed for it!).
    await setupGlobalMocks(page, { authenticated: true }); // Keep true to avoid any legacy check issues if any remain
    await page.goto('/');
  });

  test('dashboard renders correctly', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
    await expect(page.getByText('Unified OpenBiology Control Plane')).toBeVisible();
    
    // Check for some new custom components
    await expect(page.getByRole('heading', { name: 'Pipelines live' })).toBeVisible();
    await expect(page.getByText('OpenBiology Modules').first()).toBeVisible();
  });

  test('404 page for unknown routes', async ({ page }) => {
    await page.goto('/unknown-page');

    await expect(page.getByText('404')).toBeVisible();
    await expect(page.getByText(/page not found/i)).toBeVisible();
    
    // Test Home link
    await page.getByRole('link', { name: /go home/i }).click();
    await expect(page).toHaveURL('/');
  });

  test('navigates sidebar links', async ({ page }) => {
    // Verify Dashboard is active
    await expect(page.getByRole('link', { name: 'Dashboard' })).toBeVisible();

    // Navigate to Metrics (if it still exists in Sidebar - presumably yes)
    await page.locator('nav, aside').getByRole('link', { name: 'Metrics', exact: true }).first().click();
    await expect(page).toHaveURL('/metrics');
    await expect(page.getByRole('heading', { name: 'Metrics' })).toBeVisible();
  });
});
