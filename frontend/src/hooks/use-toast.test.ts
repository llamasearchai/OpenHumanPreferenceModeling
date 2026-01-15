/**
 * useToast Hook Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { reducer, useToast, toast, ToasterToast } from './use-toast';

describe('useToast', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('reducer', () => {
    const initialState: { toasts: ToasterToast[] } = { toasts: [] };

    it('handles ADD_TOAST', () => {
      const newToast = { id: '1', title: 'Test Toast' };
      const result = reducer(initialState, { type: 'ADD_TOAST', toast: newToast as never });

      expect(result.toasts).toHaveLength(1);
      expect(result.toasts[0]).toEqual(newToast);
    });

    it('limits toasts to TOAST_LIMIT', () => {
      let state = initialState;
      for (let i = 0; i < 10; i++) {
        state = reducer(state, {
          type: 'ADD_TOAST',
          toast: { id: String(i), title: `Toast ${i}` } as never,
        });
      }

      expect(state.toasts.length).toBeLessThanOrEqual(5);
    });

    it('handles UPDATE_TOAST', () => {
      const state = { toasts: [{ id: '1', title: 'Original' }] as never[] };
      const result = reducer(state, {
        type: 'UPDATE_TOAST',
        toast: { id: '1', title: 'Updated' },
      });

      expect(result.toasts[0].title).toBe('Updated');
    });

    it('handles DISMISS_TOAST with specific id', () => {
      const state = {
        toasts: [
          { id: '1', title: 'Toast 1', open: true },
          { id: '2', title: 'Toast 2', open: true },
        ] as never[],
      };

      const result = reducer(state, { type: 'DISMISS_TOAST', toastId: '1' });

      expect(result.toasts[0].open).toBe(false);
      expect(result.toasts[1].open).toBe(true);
    });

    it('handles DISMISS_TOAST without id (dismiss all)', () => {
      const state = {
        toasts: [
          { id: '1', title: 'Toast 1', open: true },
          { id: '2', title: 'Toast 2', open: true },
        ] as never[],
      };

      const result = reducer(state, { type: 'DISMISS_TOAST' });

      expect(result.toasts[0].open).toBe(false);
      expect(result.toasts[1].open).toBe(false);
    });

    it('handles REMOVE_TOAST with specific id', () => {
      const state = {
        toasts: [
          { id: '1', title: 'Toast 1' },
          { id: '2', title: 'Toast 2' },
        ] as never[],
      };

      const result = reducer(state, { type: 'REMOVE_TOAST', toastId: '1' });

      expect(result.toasts).toHaveLength(1);
      expect(result.toasts[0].id).toBe('2');
    });

    it('handles REMOVE_TOAST without id (remove all)', () => {
      const state = {
        toasts: [
          { id: '1', title: 'Toast 1' },
          { id: '2', title: 'Toast 2' },
        ] as never[],
      };

      const result = reducer(state, { type: 'REMOVE_TOAST' });

      expect(result.toasts).toHaveLength(0);
    });
  });

  describe('toast function', () => {
    it('creates a toast with title', () => {
      const { id, dismiss, update } = toast({ title: 'Test Toast' });

      expect(id).toBeDefined();
      expect(typeof dismiss).toBe('function');
      expect(typeof update).toBe('function');
    });

    it('creates a toast with description', () => {
      const { id } = toast({
        title: 'Title',
        description: 'Description text',
      });

      expect(id).toBeDefined();
    });

    it('creates a toast with variant', () => {
      const { id } = toast({
        title: 'Error',
        variant: 'destructive',
      });

      expect(id).toBeDefined();
    });
  });

  describe('useToast hook', () => {
    it('returns toast function and toasts array', () => {
      const { result } = renderHook(() => useToast());

      expect(result.current.toast).toBeDefined();
      expect(result.current.toasts).toBeDefined();
      expect(result.current.dismiss).toBeDefined();
    });

    it('adds toast when toast function is called', () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ title: 'New Toast' });
      });

      expect(result.current.toasts.length).toBeGreaterThan(0);
    });

    it('dismisses toast when dismiss function is called', () => {
      const { result } = renderHook(() => useToast());

      let toastId: string;
      act(() => {
        const { id } = result.current.toast({ title: 'Toast to dismiss' });
        toastId = id;
      });

      act(() => {
        result.current.dismiss(toastId);
      });

      // Toast should be marked as closed
      const dismissedToast = result.current.toasts.find(t => t.id === toastId);
      expect(dismissedToast?.open).toBe(false);
    });

    it('removes a dismissed toast after the timeout elapses', () => {
      const { result } = renderHook(() => useToast());

      let toastId: string;
      act(() => {
        const { id } = result.current.toast({ title: 'Toast to remove' });
        toastId = id;
      });

      act(() => {
        result.current.dismiss(toastId);
      });

      // Fast-forward the removal timer
      act(() => {
        vi.runAllTimers();
      });

      expect(result.current.toasts.find(t => t.id === toastId)).toBeUndefined();
    });

    it('dismisses all toasts when dismiss is called without an id', () => {
      const { result } = renderHook(() => useToast());

      act(() => {
        result.current.toast({ title: 'Toast 1' });
        result.current.toast({ title: 'Toast 2' });
      });

      act(() => {
        result.current.dismiss();
      });

      expect(result.current.toasts.every(t => t.open === false)).toBe(true);
    });

    it('dismisses toast via onOpenChange when open becomes false', () => {
      const { result } = renderHook(() => useToast());

      let toastId: string;
      act(() => {
        const { id } = result.current.toast({ title: 'Toast with onOpenChange' });
        toastId = id;
      });

      const created = result.current.toasts.find(t => t.id === toastId);
      expect(created).toBeTruthy();

      act(() => {
        created?.onOpenChange?.(false);
      });

      const dismissed = result.current.toasts.find(t => t.id === toastId);
      expect(dismissed?.open).toBe(false);
    });
  });
});
