import * as React from 'react';

import { selectTheme, useUIStore } from '@/stores/ui-store';

interface ThemeProviderProps {
  children: React.ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  const theme = useUIStore(selectTheme);
  const setResolvedTheme = useUIStore((state) => state.setResolvedTheme);

  React.useEffect(() => {
    if (typeof window === 'undefined') return;

    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const root = document.documentElement;

    const applyTheme = () => {
      const resolved = theme === 'system' ? (media.matches ? 'dark' : 'light') : theme;
      root.classList.toggle('dark', resolved === 'dark');
      setResolvedTheme(resolved);
      // #region agent log
      fetch('http://127.0.0.1:7259/ingest/44e72182-20fc-4ac5-ace5-6d05735c6915',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ThemeProvider.tsx:applyTheme',message:'Theme resolved',data:{theme,systemDark:media.matches,resolved},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
      // #endregion
    };

    applyTheme();

    if (theme === 'system') {
      media.addEventListener('change', applyTheme);
      return () => media.removeEventListener('change', applyTheme);
    }

    return undefined;
  }, [theme, setResolvedTheme]);

  return <>{children}</>;
}
