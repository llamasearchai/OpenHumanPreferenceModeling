/**
 * UI Store (Zustand)
 *
 * Purpose: Client-side UI state management for global application state
 * that doesn't belong in URL or server state
 *
 * REFACTORED: Now uses the "Slice Pattern" to compose multiple store slices
 * into a single unified store for backward compatibility.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

import { createThemeSlice, ThemeSlice } from './slices/theme-slice';
import { createNotificationSlice, NotificationSlice } from './slices/notification-slice';
import { createLayoutSlice, LayoutSlice } from './slices/layout-slice';
import { createModalSlice, ModalSlice } from './slices/modal-slice';

// ============================================================================
// Types (Re-exported for backward compatibility)
// ============================================================================

export type { Theme } from './slices/theme-slice';
export type { Notification } from './slices/notification-slice';
export type { ModalState } from './slices/modal-slice';

// Unified Store Type
export interface StoreActions {
  reset: () => void;
}

export type UIState = ThemeSlice & NotificationSlice & LayoutSlice & ModalSlice & StoreActions;

// ============================================================================
// Store
// ============================================================================

export const useUIStore = create<UIState>()(
  persist(
    (...a) => ({
      ...createThemeSlice(...a),
      ...createNotificationSlice(...a),
      ...createLayoutSlice(...a),
      ...createModalSlice(...a),
      
      // Reset action (manually composed)
      reset: () => {
        const [set] = a;
        set({
          // Theme
          theme: 'system',
          resolvedTheme: 'light',
          // Notifications
          notifications: [],
          maxNotifications: 5,
          // Layout
          sidebarCollapsed: false,
          sidebarMobileOpen: false,
          commandPaletteOpen: false,
          globalLoading: false,
          loadingMessage: undefined,
          breakpoint: 'desktop',
          isMobile: false,
          isTablet: false,
          isDesktop: true,
          // Modals
          modals: {},
        });
      }
    }),
    {
      name: 'ui-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        // Only persist these fields
        sidebarCollapsed: state.sidebarCollapsed,
        theme: state.theme,
      }),
    }
  )
);

// ============================================================================
// Selectors (for optimized re-renders)
// ============================================================================

export const selectSidebarCollapsed = (state: UIState) => state.sidebarCollapsed;
export const selectSidebarMobileOpen = (state: UIState) => state.sidebarMobileOpen;
export const selectTheme = (state: UIState) => state.theme;
export const selectResolvedTheme = (state: UIState) => state.resolvedTheme;
export const selectNotifications = (state: UIState) => state.notifications;
export const selectCommandPaletteOpen = (state: UIState) => state.commandPaletteOpen;
export const selectGlobalLoading = (state: UIState) => state.globalLoading;
export const selectBreakpoint = (state: UIState) => state.breakpoint;
export const selectIsMobile = (state: UIState) => state.isMobile;

/** Select if a specific modal is open */
export const selectModalOpen = (id: string) => (state: UIState) =>
  state.modals[id]?.isOpen ?? false;

/** Select modal data */
export const selectModalData = <T = unknown>(id: string) => (state: UIState) =>
  state.modals[id]?.data as T | undefined;

// ============================================================================
// Hooks for common use cases
// ============================================================================

/** Hook for sidebar state */
export function useSidebar() {
  const collapsed = useUIStore(selectSidebarCollapsed);
  const mobileOpen = useUIStore(selectSidebarMobileOpen);
  const toggle = useUIStore((state) => state.toggleSidebar);
  const toggleMobile = useUIStore((state) => state.toggleMobileSidebar);
  const setCollapsed = useUIStore((state) => state.setSidebarCollapsed);
  const setMobileOpen = useUIStore((state) => state.setMobileSidebarOpen);

  return {
    collapsed,
    mobileOpen,
    toggle,
    toggleMobile,
    setCollapsed,
    setMobileOpen,
  };
}

/** Hook for theme state */
export function useTheme() {
  const theme = useUIStore(selectTheme);
  const resolvedTheme = useUIStore(selectResolvedTheme);
  const setTheme = useUIStore((state) => state.setTheme);
  const setResolvedTheme = useUIStore((state) => state.setResolvedTheme);

  return {
    theme,
    resolvedTheme,
    setTheme,
    setResolvedTheme,
    isDark: resolvedTheme === 'dark',
    isLight: resolvedTheme === 'light',
  };
}

/** Hook for modal state */
export function useModal(id: string) {
  const isOpen = useUIStore(selectModalOpen(id));
  const data = useUIStore(selectModalData(id));
  const open = useUIStore((state) => state.openModal);
  const close = useUIStore((state) => state.closeModal);
  const toggle = useUIStore((state) => state.toggleModal);

  return {
    isOpen,
    data,
    open: (modalData?: unknown) => open(id, modalData),
    close: () => close(id),
    toggle: () => toggle(id),
  };
}

/** Hook for notifications */
export function useNotifications() {
  const notifications = useUIStore(selectNotifications);
  const add = useUIStore((state) => state.addNotification);
  const remove = useUIStore((state) => state.removeNotification);
  const clear = useUIStore((state) => state.clearNotifications);

  return {
    notifications,
    add,
    remove,
    clear,
    // Convenience methods
    info: (title: string, message?: string) => add({ type: 'info', title, message }),
    success: (title: string, message?: string) => add({ type: 'success', title, message }),
    warning: (title: string, message?: string) => add({ type: 'warning', title, message }),
    error: (title: string, message?: string) => add({ type: 'error', title, message }),
  };
}

/** Hook for responsive breakpoints */
export function useBreakpoint() {
  const breakpoint = useUIStore(selectBreakpoint);
  const isMobile = useUIStore((state) => state.isMobile);
  const isTablet = useUIStore((state) => state.isTablet);
  const isDesktop = useUIStore((state) => state.isDesktop);
  const setBreakpoint = useUIStore((state) => state.setBreakpoint);

  return {
    breakpoint,
    isMobile,
    isTablet,
    isDesktop,
    setBreakpoint,
  };
}
