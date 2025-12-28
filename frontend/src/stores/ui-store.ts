/**
 * UI Store (Zustand)
 * 
 * Purpose: Client-side UI state management for global application state
 * that doesn't belong in URL or server state
 * 
 * Features:
 * - Sidebar collapsed state
 * - Modal management
 * - Theme preferences
 * - Notification queue
 * - Command palette state
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

/** Theme options */
export type Theme = 'light' | 'dark' | 'system';

/** Modal state */
export interface ModalState {
  id: string;
  isOpen: boolean;
  data?: unknown;
}

/** Notification in queue */
export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message?: string | undefined;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
  createdAt: number;
}

/** UI Store state */
export interface UIState {
  // Sidebar
  sidebarCollapsed: boolean;
  sidebarMobileOpen: boolean;

  // Theme
  theme: Theme;
  resolvedTheme: 'light' | 'dark';

  // Modals
  modals: Record<string, ModalState>;

  // Notifications
  notifications: Notification[];
  maxNotifications: number;

  // Command Palette
  commandPaletteOpen: boolean;

  // Loading states
  globalLoading: boolean;
  loadingMessage?: string | undefined;

  // Breakpoint (updated by resize observer)
  breakpoint: 'mobile' | 'tablet' | 'desktop';
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
}

/** UI Store actions */
export interface UIActions {
  // Sidebar
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleMobileSidebar: () => void;
  setMobileSidebarOpen: (open: boolean) => void;

  // Theme
  setTheme: (theme: Theme) => void;
  setResolvedTheme: (theme: 'light' | 'dark') => void;

  // Modals
  openModal: (id: string, data?: unknown) => void;
  closeModal: (id: string) => void;
  toggleModal: (id: string) => void;
  closeAllModals: () => void;
  getModalData: <T = unknown>(id: string) => T | undefined;

  // Notifications
  addNotification: (notification: Omit<Notification, 'id' | 'createdAt'>) => string;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;

  // Command Palette
  openCommandPalette: () => void;
  closeCommandPalette: () => void;
  toggleCommandPalette: () => void;

  // Loading
  setGlobalLoading: (loading: boolean, message?: string) => void;

  // Breakpoint
  setBreakpoint: (breakpoint: 'mobile' | 'tablet' | 'desktop') => void;

  // Reset
  reset: () => void;
}

// ============================================================================
// Initial State
// ============================================================================

const initialState: UIState = {
  sidebarCollapsed: false,
  sidebarMobileOpen: false,
  theme: 'system',
  resolvedTheme: 'light',
  modals: {},
  notifications: [],
  maxNotifications: 5,
  commandPaletteOpen: false,
  globalLoading: false,
  breakpoint: 'desktop',
  isMobile: false,
  isTablet: false,
  isDesktop: true,
};

// ============================================================================
// Store
// ============================================================================

export const useUIStore = create<UIState & UIActions>()(
  persist(
    (set, get) => ({
      ...initialState,

      // Sidebar actions
      toggleSidebar: () =>
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

      setSidebarCollapsed: (collapsed) =>
        set({ sidebarCollapsed: collapsed }),

      toggleMobileSidebar: () =>
        set((state) => ({ sidebarMobileOpen: !state.sidebarMobileOpen })),

      setMobileSidebarOpen: (open) =>
        set({ sidebarMobileOpen: open }),

      // Theme actions
      setTheme: (theme) =>
        set({ theme }),

      setResolvedTheme: (resolved) =>
        set({ resolvedTheme: resolved }),

      // Modal actions
      openModal: (id, data) =>
        set((state) => {
          const next: ModalState =
            data === undefined ? { id, isOpen: true } : { id, isOpen: true, data };
          return { modals: { ...state.modals, [id]: next } };
        }),

      closeModal: (id) =>
        set((state) => {
          const current = state.modals[id];
          if (!current) return {};
          return { modals: { ...state.modals, [id]: { ...current, isOpen: false } } };
        }),

      toggleModal: (id) =>
        set((state) => {
          const current = state.modals[id];
          if (!current) {
            return { modals: { ...state.modals, [id]: { id, isOpen: true } } };
          }
          return { modals: { ...state.modals, [id]: { ...current, isOpen: !current.isOpen } } };
        }),

      closeAllModals: () =>
        set((state) => {
          const next: Record<string, ModalState> = {};
          for (const [id, modal] of Object.entries(state.modals)) {
            next[id] = { ...modal, isOpen: false };
          }
          return { modals: next };
        }),

      getModalData: <T = unknown>(id: string) => {
        return get().modals[id]?.data as T | undefined;
      },

      // Notification actions
      addNotification: (notification) => {
        const id = `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const createdAt = Date.now();

        set((state) => {
          const newNotification: Notification = { ...notification, id, createdAt };
          const next = [newNotification, ...state.notifications].slice(0, state.maxNotifications);
          return { notifications: next };
        });

        return id;
      },

      removeNotification: (id) =>
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
        })),

      clearNotifications: () =>
        set({ notifications: [] }),

      // Command palette actions
      openCommandPalette: () =>
        set({ commandPaletteOpen: true }),

      closeCommandPalette: () =>
        set({ commandPaletteOpen: false }),

      toggleCommandPalette: () =>
        set((state) => ({ commandPaletteOpen: !state.commandPaletteOpen })),

      // Loading actions
      setGlobalLoading: (loading, message) =>
        set({ globalLoading: loading, loadingMessage: message }),

      // Breakpoint actions
      setBreakpoint: (breakpoint) =>
        set({
          breakpoint,
          isMobile: breakpoint === 'mobile',
          isTablet: breakpoint === 'tablet',
          isDesktop: breakpoint === 'desktop',
        }),

      // Reset
      reset: () => set(initialState),
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
