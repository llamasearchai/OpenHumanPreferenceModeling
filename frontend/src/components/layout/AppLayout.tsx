/**
 * App Layout Component
 *
 * Purpose: Main application shell with sidebar and header
 * Responsive design: On mobile (<md), sidebar overlays content.
 * On tablet and desktop, sidebar pushes content.
 */

import * as React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { selectSidebarCollapsed, useUIStore } from '@/stores/ui-store';
import { cn } from '@/lib/utils';
import { AnimatePresence, motion } from 'framer-motion';

function getPageTitle(pathname: string): string {
  const routeTitles: Array<{ prefix: string; title: string }> = [
    { prefix: '/', title: 'Dashboard' },
    { prefix: '/annotations', title: 'Annotations' },
    { prefix: '/metrics', title: 'Metrics' },
    { prefix: '/alerts', title: 'Alerts' },
    { prefix: '/calibration', title: 'Calibration' },
    { prefix: '/active-learning', title: 'Active Learning' },
    { prefix: '/federated-learning', title: 'Federated Learning' },
    { prefix: '/quality-control', title: 'Quality Control' },
    { prefix: '/training', title: 'Training' },
    { prefix: '/playground', title: 'Playground' },
    { prefix: '/settings', title: 'Settings' },
  ];

  // Exact match for root; prefix match for nested pages.
  if (pathname === '/' || pathname === '') return 'Dashboard';
  const match = routeTitles.find((r) => r.prefix !== '/' && pathname.startsWith(r.prefix));
  return match?.title || 'OpenHuman Preference Modeling';
}

export function AppLayout() {
  const sidebarCollapsed = useUIStore(selectSidebarCollapsed);
  const location = useLocation();


  // UX: scroll to top on route change.
  React.useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  }, [location.pathname]);

  // UX: keep document title in sync with current route.
  React.useEffect(() => {
    const pageTitle = getPageTitle(location.pathname);
    document.title =
      pageTitle === 'OpenHuman Preference Modeling'
        ? pageTitle
        : `${pageTitle} • OpenHuman Preference Modeling`;
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-background">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-background focus:px-3 focus:py-2 focus:text-sm focus:font-medium focus:text-foreground focus:shadow"
      >
        Skip to main content
      </a>
      <Sidebar />
      {/*
        Responsive margins:
        - Mobile: No margin (sidebar overlays or is hidden)
        - Tablet (md): Reduced margin for collapsed sidebar
        - Desktop (lg): Full margin for expanded/collapsed sidebar
      */}
      <div
        className={cn(
          'transition-all duration-300',
          // Mobile: no margin, content takes full width
          'ml-0',
          // Tablet and up: apply sidebar margin
          sidebarCollapsed
            ? 'md:ml-16'
            : 'md:ml-16 lg:ml-64'
        )}
      >
        <Header />
        <main id="main-content" className="p-3 sm:p-4 md:p-6">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
