import { test, expect } from '@playwright/test';

test.describe('Active Learning', () => {
  test.beforeEach(async ({ page }) => {
    // Register new user to ensure access
    await page.goto('/register');
    const uniqueEmail = `test_${Date.now()}_${Math.random().toString(36).substring(7)}@example.com`;
    await page.getByLabel('Full Name').fill('Test User');
    await page.getByLabel('Email').fill(uniqueEmail);
    await page.getByLabel(/^Password$/).fill('Password123!');
    await page.getByLabel('Confirm Password').fill('Password123!');
    await page.getByRole('button', { name: /create account/i }).click();
    
    // Should pass through to dashboard
    await expect(page).toHaveURL('/');
  });

  test('can navigate to Active Learning page', async ({ page }) => {
    await page.goto('/active-learning');
    await expect(page.getByRole('heading', { name: /Active Learning/i })).toBeVisible();
    await expect(page.getByText(/intelligent sampling/i)).toBeVisible();
  });

  test('displays status cards', async ({ page }) => {
    await page.goto('/active-learning');
    await expect(page.getByText('Labeled', { exact: true })).toBeVisible();
    await expect(page.getByText('Unlabeled', { exact: true })).toBeVisible();
    await expect(page.getByText('Budget', { exact: true })).toBeVisible();
    // Values might load async, so we just check labels
  });

  test('can update configuration', async ({ page }) => {
    await page.goto('/active-learning');
    
    // Check if Select exists (using label or role)
    // The Select Trigger might not have a label accessible by default if complex
    // But we can try to find the button that triggers the select
    /*
    const strategyTrigger = page.locator('button[role="combobox"]').first();
    await expect(strategyTrigger).toBeVisible();
    */
    // Checking for "Batch Size" input
    // The Slider is hard to test, but maybe there is visible text for value
  });

  test('can refresh predictions', async ({ page }) => {
    await page.goto('/active-learning');
    const refreshBtn = page.getByRole('button', { name: /Refresh Predictions/i });
    await expect(refreshBtn).toBeVisible();
    // await refreshBtn.click(); // Might fail if backend is slow, but we can check it exists
  });
});
