import { Page } from '@playwright/test';

export const mockUser = {
  id: 'user-123',
  email: 'demo@example.com',
  name: 'Demo User',
  role: 'annotator',
};

export const mockTokens = {
  accessToken: 'mock-access-token',
  refreshToken: 'mock-refresh-token',
  expiresIn: 3600,
  tokenType: 'Bearer',
};

/**
 * Mocks common API endpoints to prevent network errors during E2E tests.
 * Covers: dev-status, health, alerts, and auth validation.
 */
export async function setupGlobalMocks(page: Page, options: { authenticated?: boolean } = {}) {
  const { authenticated = false } = options;

  // Unregister any Service Workers that might be lingering from previous runs (MSW)
  await page.evaluate(async () => {
    if ('serviceWorker' in navigator) {
      const registrations = await navigator.serviceWorker.getRegistrations();
      for (const registration of registrations) {
        console.log('[Test] Unregistering SW:', registration.scope);
        await registration.unregister();
      }
    }
  });

  // Use a catch-all for /api/ to ensure we intercept everything
  await page.route('**/api/**', async route => {
    const url = route.request().url();
    // console.log('[Playwright Mock] Intercepted:', url);
    
    if (url.includes('/auth/dev-status')) {
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ devMode: true }) });
    }
    
    if (url.includes('/health')) {
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'healthy' }) });
    }
    
    if (url.includes('/alerts')) {
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) });
    }
    
    if (url.includes('/auth/refresh')) {
      if (authenticated) {
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(mockTokens) });
      } else {
        return route.fulfill({ status: 401, contentType: 'application/json', body: JSON.stringify({ detail: 'Unauthorized' }) });
      }
    }
    
    if (url.includes('/auth/me')) {
      if (authenticated) {
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(mockUser) });
      } else {
        return route.fulfill({ status: 401, contentType: 'application/json', body: JSON.stringify({ detail: 'Unauthorized' }) });
      }
    }
    
    if (url.includes('/metrics')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([{
          name: 'encoder_latency',
          value: 123.45,
          timestamp: new Date().toISOString(),
          tags: { service: 'api' },
        }]),
      });
    }

    if (url.includes('/auth/login')) {
       return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(mockTokens) });
    }

    // Default fallback for other API calls -> 200 OK empty
    // console.log('[Playwright Mock] Fallback 200 for:', url);
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) });
  });
}

/**
 * Bypasses the login screen by manually setting auth state
 * AND mocking the necessary API validation calls.
 */
export async function bypassLogin(page: Page) {
  // 1. Set LocalStorage
  await page.addInitScript(({ user, tokens }) => {
    window.localStorage.setItem('auth_tokens', JSON.stringify(tokens));
    window.localStorage.setItem('auth_user', JSON.stringify(user));
  }, { user: mockUser, tokens: mockTokens });

  // 2. Setup Mocks (Authenticated)
  await setupGlobalMocks(page, { authenticated: true });
}
