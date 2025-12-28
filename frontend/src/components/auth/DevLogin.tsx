/**
 * DevLogin Component
 *
 * Purpose: Provides instant login bypass for development mode.
 * This component only renders when the backend has DEV_MODE enabled.
 */


import * as React from 'react';
import { useQuery } from '@tanstack/react-query';
import { AlertTriangle, Zap, User, Shield, Code } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useAuth } from '@/contexts/AuthContext';
import { api } from '@/lib/api-client';

interface DevUser {
  id: string;
  name: string;
  role: string;
  icon: React.ElementType;
}

const DEV_USERS: DevUser[] = [
  { id: 'demo-user-id', name: 'Demo Admin', role: 'admin', icon: Shield },
  { id: 'dev-annotator', name: 'Dev Annotator', role: 'annotator', icon: User },
  { id: 'dev-reviewer', name: 'Dev Reviewer', role: 'reviewer', icon: Code },
];

export function DevLogin() {
  const { devLogin } = useAuth();
  const [selectedUser, setSelectedUser] = React.useState<string | null>(null);
  const [isLoggingIn, setIsLoggingIn] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  // Check if dev mode is enabled
  const { data: devStatus, isLoading: checkingDevMode } = useQuery({
    queryKey: ['auth', 'dev-status'],
    queryFn: async () => {
      const response = await api.get<{ devMode: boolean }>('/api/auth/dev-status');
      if (!response.success) throw new Error(response.error?.detail || 'Failed to check dev status');
      return response.data;
    },
    retry: false,
  });

  const handleDevLogin = async (userId: string) => {
    setSelectedUser(userId);
    setIsLoggingIn(true);
    setError(null);
    
    try {
      const success = await devLogin(userId);
      if (!success) {
        setError("Dev login failed. The server might not be in dev mode.");
      }
      // Navigation is handled by Login.tsx observing AuthContext state or explicit navigate calls there, 
      // but standard login function usually returns success boolean.
      // Wait, Login.tsx uses login() which redirects.
      // Here we might need to rely on the parent or effect. 
      // But actually, seeing as DevLogin is inside Login page, successful auth updates context, 
      // and Login page might react to that? 
      // Re-reading Login.tsx: it uses login() which navigates.
      // DevLogin component is just rendered. If auth state changes to authenticated, 
      // ProtectedRoute logic (if we were protected) would work. 
      // But Login page doesn't auto-redirect if already logged in unless we add an effect.
      // Actually, Login.tsx SHOULD redirect if already logged in. 
      // Let's modify Login.tsx to do that too if needed.
    } catch (e) {
      setError("An unexpected error occurred.");
    } finally {
      setIsLoggingIn(false);
      setSelectedUser(null);
    }
  };

  // Don't render if still checking or dev mode is disabled
  if (checkingDevMode || !devStatus?.devMode) {
    return null;
  }

  return (
    <Card className="border-yellow-500/50 bg-yellow-500/5 mb-6">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-yellow-500" />
          <CardTitle className="text-lg">Development Mode</CardTitle>
        </div>
        <CardDescription>
          Quick login for development. Not available in production.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-2">
          {DEV_USERS.map((user) => {
            const Icon = user.icon;
            const isLoading = isLoggingIn && selectedUser === user.id;

            return (
              <Button
                key={user.id}
                variant="outline"
                className="justify-start gap-3 h-auto py-3 bg-background"
                onClick={() => handleDevLogin(user.id)}
                disabled={isLoggingIn}
              >
                <Icon className="h-4 w-4 text-muted-foreground" />
                <div className="flex flex-col items-start">
                  <span className="font-medium">{user.name}</span>
                  <span className="text-xs text-muted-foreground">
                    Role: {user.role}
                  </span>
                </div>
                {isLoading && (
                  <Zap className="ml-auto h-4 w-4 animate-pulse text-yellow-500" />
                )}
              </Button>
            );
          })}
        </div>

        {error && (
          <p className="mt-3 text-sm text-destructive">
            {error}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export default DevLogin;
