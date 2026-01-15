import { StateCreator } from 'zustand';

export interface LayoutSlice {
  // Sidebar
  sidebarCollapsed: boolean;
  sidebarMobileOpen: boolean;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
  toggleMobileSidebar: () => void;
  setMobileSidebarOpen: (open: boolean) => void;

  // Command Palette
  commandPaletteOpen: boolean;
  openCommandPalette: () => void;
  closeCommandPalette: () => void;
  toggleCommandPalette: () => void;

  // Loading
  globalLoading: boolean;
  loadingMessage?: string | undefined;
  setGlobalLoading: (loading: boolean, message?: string) => void;

  // Breakpoint
  breakpoint: 'mobile' | 'tablet' | 'desktop';
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  setBreakpoint: (breakpoint: 'mobile' | 'tablet' | 'desktop') => void;
}

export const createLayoutSlice: StateCreator<LayoutSlice> = (set) => ({
  // Sidebar
  sidebarCollapsed: false,
  sidebarMobileOpen: false,
  toggleSidebar: () =>
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
  toggleMobileSidebar: () =>
    set((state) => ({ sidebarMobileOpen: !state.sidebarMobileOpen })),
  setMobileSidebarOpen: (open) => set({ sidebarMobileOpen: open }),

  // Command Palette
  commandPaletteOpen: false,
  openCommandPalette: () => set({ commandPaletteOpen: true }),
  closeCommandPalette: () => set({ commandPaletteOpen: false }),
  toggleCommandPalette: () =>
    set((state) => ({ commandPaletteOpen: !state.commandPaletteOpen })),

  // Loading
  globalLoading: false,
  loadingMessage: undefined,
  setGlobalLoading: (loading, message) =>
    set({ globalLoading: loading, loadingMessage: message }),

  // Breakpoint
  breakpoint: 'desktop',
  isMobile: false,
  isTablet: false,
  isDesktop: true,
  setBreakpoint: (breakpoint) =>
    set({
      breakpoint,
      isMobile: breakpoint === 'mobile',
      isTablet: breakpoint === 'tablet',
      isDesktop: breakpoint === 'desktop',
    }),
});
