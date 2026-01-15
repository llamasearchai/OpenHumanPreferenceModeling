/**
 * Basic Smoke Test
 */

import { test, expect } from '@playwright/test';

test('has title and renders dashboard', async ({ page }) => {
  await page.goto('/');

  // App should have the correct title (includes OHPM or Open Human Preference)
  await expect(page).toHaveTitle(/OHPM|Open Human Preference/);
  
  // Auth is removed, so we should land on Dashboard directly
  await expect(page).toHaveURL('/');
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
});
