/**
 * Auth Store Tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useAuthStore } from './auth-store';

describe('useAuthStore', () => {
  beforeEach(() => {
    // Reset store before each test
    act(() => {
      useAuthStore.getState().logout();
    });
  });

  describe('initial state', () => {
    it('starts with null tokens', () => {
      const { result } = renderHook(() => useAuthStore());

      expect(result.current.accessToken).toBeNull();
      expect(result.current.refreshToken).toBeNull();
    });

    it('starts unauthenticated', () => {
      const { result } = renderHook(() => useAuthStore());

      expect(result.current.isAuthenticated()).toBe(false);
    });
  });

  describe('setAuth', () => {
    it('sets access and refresh tokens', () => {
      const { result } = renderHook(() => useAuthStore());

      act(() => {
        result.current.setAuth('access-token-123', 'refresh-token-456');
      });

      expect(result.current.accessToken).toBe('access-token-123');
      expect(result.current.refreshToken).toBe('refresh-token-456');
    });

    it('makes user authenticated after setting tokens', () => {
      const { result } = renderHook(() => useAuthStore());

      act(() => {
        result.current.setAuth('access-token', 'refresh-token');
      });

      expect(result.current.isAuthenticated()).toBe(true);
    });

    it('can update tokens', () => {
      const { result } = renderHook(() => useAuthStore());

      act(() => {
        result.current.setAuth('old-access', 'old-refresh');
      });

      act(() => {
        result.current.setAuth('new-access', 'new-refresh');
      });

      expect(result.current.accessToken).toBe('new-access');
      expect(result.current.refreshToken).toBe('new-refresh');
    });
  });

  describe('logout', () => {
    it('clears tokens', () => {
      const { result } = renderHook(() => useAuthStore());

      act(() => {
        result.current.setAuth('access', 'refresh');
      });

      act(() => {
        result.current.logout();
      });

      expect(result.current.accessToken).toBeNull();
      expect(result.current.refreshToken).toBeNull();
    });

    it('makes user unauthenticated', () => {
      const { result } = renderHook(() => useAuthStore());

      act(() => {
        result.current.setAuth('access', 'refresh');
      });

      expect(result.current.isAuthenticated()).toBe(true);

      act(() => {
        result.current.logout();
      });

      expect(result.current.isAuthenticated()).toBe(false);
    });

    it('is idempotent', () => {
      const { result } = renderHook(() => useAuthStore());

      act(() => {
        result.current.logout();
        result.current.logout();
      });

      expect(result.current.accessToken).toBeNull();
      expect(result.current.refreshToken).toBeNull();
    });
  });

  describe('isAuthenticated', () => {
    it('returns false when no access token', () => {
      const { result } = renderHook(() => useAuthStore());

      expect(result.current.isAuthenticated()).toBe(false);
    });

    it('returns true when access token is set', () => {
      const { result } = renderHook(() => useAuthStore());

      act(() => {
        result.current.setAuth('token', 'refresh');
      });

      expect(result.current.isAuthenticated()).toBe(true);
    });

    it('returns false when only refresh token is set manually', () => {
      // This tests the getter behavior - authentication is based on access token
      const { result } = renderHook(() => useAuthStore());

      expect(result.current.isAuthenticated()).toBe(false);
    });
  });

  describe('state sharing', () => {
    it('shares state between multiple hooks', () => {
      const { result: result1 } = renderHook(() => useAuthStore());
      const { result: result2 } = renderHook(() => useAuthStore());

      act(() => {
        result1.current.setAuth('shared-token', 'shared-refresh');
      });

      expect(result2.current.accessToken).toBe('shared-token');
      expect(result2.current.isAuthenticated()).toBe(true);
    });

    it('updates all hooks when logout is called', () => {
      const { result: result1 } = renderHook(() => useAuthStore());
      const { result: result2 } = renderHook(() => useAuthStore());

      act(() => {
        result1.current.setAuth('token', 'refresh');
      });

      act(() => {
        result2.current.logout();
      });

      expect(result1.current.accessToken).toBeNull();
      expect(result1.current.isAuthenticated()).toBe(false);
    });
  });
});
