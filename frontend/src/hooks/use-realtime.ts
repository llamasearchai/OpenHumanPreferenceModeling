/**
 * Real-time Hooks
 *
 * Purpose: React hooks for consuming WebSocket updates
 * with automatic cleanup and TanStack Query integration.
 */

import { useEffect, useCallback, useState, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  RealtimeClient,
  getRealtimeClient,
  MessageType,
  WebSocketMessage,
  ConnectionState,
} from '@/lib/websocket-client';
import { useAuth } from '@/contexts/AuthContext';

/**
 * Hook to manage WebSocket connection lifecycle
 */
export function useRealtimeConnection() {
  const { isAuthenticated } = useAuth();
  const [state, setState] = useState<ConnectionState>('disconnected');
  const [error, setError] = useState<Error | null>(null);
  const clientRef = useRef<RealtimeClient | null>(null);

  useEffect(() => {
    if (!isAuthenticated) {
      return;
    }

    const client = getRealtimeClient();
    clientRef.current = client;

    const unsubState = client.onStateChange((newState) => {
      setState(newState);
      if (newState === 'connected') {
        setError(null);
      }
    });

    const unsubError = client.onError((err) => {
      setError(err);
    });

    client.connect();

    return () => {
      unsubState();
      unsubError();
    };
  }, [isAuthenticated]);

  const reconnect = useCallback(() => {
    clientRef.current?.connect();
  }, []);

  return {
    state,
    error,
    isConnected: state === 'connected',
    isReconnecting: state === 'reconnecting',
    reconnect,
  };
}

/**
 * Hook to subscribe to specific message types
 */
export function useRealtimeMessage<T = unknown>(
  messageType: string,
  handler: (payload: T, message: WebSocketMessage) => void
) {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    const client = getRealtimeClient();

    const unsubscribe = client.on(messageType, (message) => {
      handlerRef.current(message.payload as T, message);
    });

    return unsubscribe;
  }, [messageType]);
}

/**
 * Hook to automatically invalidate queries on real-time updates
 */
export function useRealtimeQueryInvalidation() {
  const queryClient = useQueryClient();

  useEffect(() => {
    const client = getRealtimeClient();

    const unsubMetrics = client.on(MessageType.METRIC_UPDATE, () => {
      queryClient.invalidateQueries({ queryKey: ['metrics'] });
    });

    const unsubAlerts = client.on(MessageType.ALERT_UPDATE, () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
    });

    const unsubTasks = client.on(MessageType.TASK_ASSIGNED, () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    });

    const unsubCalibration = client.on(MessageType.CALIBRATION_STATUS, () => {
      queryClient.invalidateQueries({ queryKey: ['calibration'] });
    });

    return () => {
      unsubMetrics();
      unsubAlerts();
      unsubTasks();
      unsubCalibration();
    };
  }, [queryClient]);
}

/**
 * Hook for real-time metric updates
 */
export function useRealtimeMetrics(
  onUpdate?: (metric: { name: string; value: number; timestamp: string }) => void
) {
  const [latestMetric, setLatestMetric] = useState<{
    name: string;
    value: number;
    timestamp: string;
  } | null>(null);

  useRealtimeMessage<{ name: string; value: number; timestamp: string }>(
    MessageType.METRIC_UPDATE,
    (payload) => {
      setLatestMetric(payload);
      onUpdate?.(payload);
    }
  );

  return latestMetric;
}

/**
 * Hook for real-time alert updates
 */
export function useRealtimeAlerts(
  onUpdate?: (alert: {
    id: string;
    rule_name: string;
    severity: string;
    status: string;
    message: string;
  }) => void
) {
  const [latestAlert, setLatestAlert] = useState<{
    id: string;
    rule_name: string;
    severity: string;
    status: string;
    message: string;
  } | null>(null);

  useRealtimeMessage<{
    id: string;
    rule_name: string;
    severity: string;
    status: string;
    message: string;
  }>(MessageType.ALERT_UPDATE, (payload) => {
    setLatestAlert(payload);
    onUpdate?.(payload);
  });

  return latestAlert;
}

/**
 * Hook for real-time training progress
 */
export function useRealtimeTrainingProgress(
  onUpdate?: (progress: {
    run_id: string;
    step: number;
    loss: number;
    learning_rate: number;
    eta_seconds: number;
  }) => void
) {
  const [progress, setProgress] = useState<{
    run_id: string;
    step: number;
    loss: number;
    learning_rate: number;
    eta_seconds: number;
  } | null>(null);

  useRealtimeMessage<{
    run_id: string;
    step: number;
    loss: number;
    learning_rate: number;
    eta_seconds: number;
  }>(MessageType.TRAINING_PROGRESS, (payload) => {
    setProgress(payload);
    onUpdate?.(payload);
  });

  return progress;
}

/**
 * Hook for real-time calibration status
 */
export function useRealtimeCalibration(
  onUpdate?: (status: {
    ece: number;
    triggered: boolean;
    in_progress: boolean;
  }) => void
) {
  const [status, setStatus] = useState<{
    ece: number;
    triggered: boolean;
    in_progress: boolean;
  } | null>(null);

  useRealtimeMessage<{
    ece: number;
    triggered: boolean;
    in_progress: boolean;
  }>(MessageType.CALIBRATION_STATUS, (payload) => {
    setStatus(payload);
    onUpdate?.(payload);
  });

  return status;
}

/**
 * Connection status indicator component hook
 */
export function useConnectionStatus() {
  const { state, isConnected, isReconnecting, error, reconnect } =
    useRealtimeConnection();

  return {
    state,
    isConnected,
    isReconnecting,
    hasError: !!error,
    errorMessage: error?.message,
    reconnect,
    statusColor: isConnected
      ? 'green'
      : isReconnecting
      ? 'yellow'
      : error
      ? 'red'
      : 'gray',
    statusText: isConnected
      ? 'Connected'
      : isReconnecting
      ? 'Reconnecting...'
      : error
      ? 'Connection Error'
      : 'Disconnected',
  };
}
