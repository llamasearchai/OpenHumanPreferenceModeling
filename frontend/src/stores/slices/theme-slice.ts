import { StateCreator } from 'zustand';

export type Theme = 'light' | 'dark' | 'system';

export interface ThemeSlice {
  theme: Theme;
  resolvedTheme: 'light' | 'dark';
  setTheme: (theme: Theme) => void;
  setResolvedTheme: (theme: 'light' | 'dark') => void;
}

export const createThemeSlice: StateCreator<ThemeSlice> = (set) => ({
  theme: 'system',
  resolvedTheme: 'light',
  setTheme: (theme) => set({ theme }),
  setResolvedTheme: (resolved) => set({ resolvedTheme: resolved }),
});
