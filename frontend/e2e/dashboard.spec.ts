/**
 * Dashboard & Metrics E2E Tests
 */

import { test, expect } from '@playwright/test';
import { setupGlobalMocks } from './test-utils';

test.describe('Dashboard Interactions', () => {
  test.beforeEach(async ({ page }) => {
    await setupGlobalMocks(page, { authenticated: true });
    await page.goto('/');
  });

  test('dashboard renders custom components', async ({ page }) => {
    // Check for the new custom dashboard components
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
    await expect(page.getByText('Unified OpenBiology Control Plane')).toBeVisible();
    
    // Check for SiteOverviewStatistics
    await expect(page.getByRole('heading', { name: 'Pipelines live' })).toBeVisible();
    
    // Check for ModuleGrid
    await expect(page.getByText('OpenBiology Modules').first()).toBeVisible();
  });

  test('metrics page renders correctly', async ({ page }) => {
    await page.goto('/metrics');

    // Verify page title - updated to match actual heading
    await expect(page.getByRole('heading', { name: 'Metrics' })).toBeVisible();
    await expect(page.getByText('Monitor system performance and health')).toBeVisible();
  });

  test('modules page is accessible from sidebar', async ({ page }) => {
    // Navigate to Modules directly from sidebar
    await page.getByRole('link', { name: 'Modules', exact: true }).click();
    await expect(page).toHaveURL('/modules');
    await expect(page.getByRole('heading', { name: 'Modules' })).toBeVisible();
  });
});
