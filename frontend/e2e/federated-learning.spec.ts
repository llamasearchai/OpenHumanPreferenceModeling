import { test, expect } from '@playwright/test';

test.describe('Federated Learning', () => {
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

  test('can navigate to Federated Learning page', async ({ page }) => {
    await page.goto('/federated-learning');
    await expect(page.getByRole('heading', { name: /Federated Learning/i })).toBeVisible();
    await expect(page.getByText(/Privacy-preserving/i)).toBeVisible();
  });

  test('displays privacy budget', async ({ page }) => {
    await page.goto('/federated-learning');
    await expect(page.getByText('Privacy Budget', { exact: true })).toBeVisible();
    // Check for "Budget Used" which is part of the gauge
    await expect(page.getByText('Budget Used')).toBeVisible();
  });

  test('displays client heatmap', async ({ page }) => {
    await page.goto('/federated-learning');
    // Click tab if needed? Default might be "Round History"
    // "Client Participation" tab
    await page.getByRole('tab', { name: /Client Participation/i }).click();
    await expect(page.getByText('Client Participation Heatmap')).toBeVisible();
  });
});
