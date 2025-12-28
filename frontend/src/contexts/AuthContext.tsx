/**
 * Authentication Context
 *
 * Purpose: Global authentication state management
 * Provides user data, auth methods, and loading states
 */

import * as React from 'react';
import { authApi, clearTokens, getAccessToken } from '@/lib/api-client';
import type { User, LoginRequest, RegisterRequest } from '@/types/api';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (credentials: LoginRequest) => Promise<boolean>;
  register: (data: RegisterRequest) => Promise<boolean>;
  logout: () => Promise<void>;
  devLogin: (userId?: string) => Promise<boolean>;
  clearError: () => void;
}

const AuthContext = React.createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = React.useState<User | null>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // Check for existing session on mount
  React.useEffect(() => {
    const checkAuth = async () => {
      const token = getAccessToken() || localStorage.getItem('refreshToken');
      if (!token) {
        setIsLoading(false);
        return;
      }

      // Try to refresh token and get user
      try {
        const refreshResult = await authApi.refreshToken();
        if (refreshResult.success) {
          const userResult = await authApi.getCurrentUser();
          if (userResult.success) {
            setUser(userResult.data);
          }
        }
      } catch (error) {
        clearTokens();
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, []);

  const login = React.useCallback(async (credentials: LoginRequest): Promise<boolean> => {
    setIsLoading(true);
    setError(null);

    const result = await authApi.login(credentials);

    if (result.success) {
      // Get user data after login
      const userResult = await authApi.getCurrentUser();
      if (userResult.success) {
        setUser(userResult.data);
        setIsLoading(false);
        return true;
      }
    }

    setError(result.success ? 'Failed to get user data' : result.error.detail);
    setIsLoading(false);
    return false;
  }, []);

  const register = React.useCallback(async (data: RegisterRequest): Promise<boolean> => {
    setIsLoading(true);
    setError(null);

    const result = await authApi.register(data);

    if (result.success) {
      const userResult = await authApi.getCurrentUser();
      if (userResult.success) {
        setUser(userResult.data);
        setIsLoading(false);
        return true;
      }
    }

    setError(result.success ? 'Failed to get user data' : result.error.detail);
    setIsLoading(false);
    return false;
  }, []);

  const logout = React.useCallback(async () => {
    await authApi.logout();
    setUser(null);
    setError(null);
  }, []);

  const devLogin = React.useCallback(async (userId: string = 'demo-user-id'): Promise<boolean> => {
    setIsLoading(true);
    setError(null);

    const result = await authApi.devLogin(userId);

    if (result.success) {
      const userResult = await authApi.getCurrentUser();
      if (userResult.success) {
        setUser(userResult.data);
        setIsLoading(false);
        return true;
      }
    }

    setError(result.success ? 'Failed to get user data' : result.error.detail);
    setIsLoading(false);
    return false;
  }, []);

  const clearError = React.useCallback(() => {
    setError(null);
  }, []);

  const value = React.useMemo(
    () => ({
      user,
      isAuthenticated: !!user,
      isLoading,
      error,
      login,
      register,
      logout,
      devLogin,
      clearError,
    }),
    [user, isLoading, error, login, register, logout, devLogin, clearError]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = React.useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
