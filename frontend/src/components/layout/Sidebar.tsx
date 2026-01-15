/**
 * Sidebar Component
 *
 * Purpose: Main navigation sidebar
 */

import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  ClipboardList,
  LineChart,
  Bell,
  Settings,
  Gauge,
  ChevronLeft,
  ChevronRight,
  Brain,
  Network,
  ShieldCheck,
  GraduationCap,
  Gamepad2,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { selectSidebarCollapsed, useUIStore } from '@/stores/ui-store';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/annotations', icon: ClipboardList, label: 'Annotations' },
  { to: '/metrics', icon: LineChart, label: 'Metrics' },
  { to: '/alerts', icon: Bell, label: 'Alerts' },
  { to: '/calibration', icon: Gauge, label: 'Calibration' },
  { to: '/active-learning', icon: Brain, label: 'Active Learning' },
  { to: '/federated-learning', icon: Network, label: 'Federated Learning' },
  { to: '/quality-control', icon: ShieldCheck, label: 'Quality Control' },
  { to: '/training', icon: GraduationCap, label: 'Training' },
  { to: '/playground', icon: Gamepad2, label: 'Playground' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export function Sidebar() {
  const location = useLocation();
  const sidebarCollapsed = useUIStore(selectSidebarCollapsed);
  const toggleSidebar = useUIStore((state) => state.toggleSidebar);

  return (
    <TooltipProvider>
      {/* Mobile overlay backdrop */}
      <div
        className={cn(
          'fixed inset-0 z-30 bg-background/80 backdrop-blur-sm md:hidden transition-opacity duration-300',
          sidebarCollapsed ? 'opacity-0 pointer-events-none' : 'opacity-100'
        )}
        onClick={toggleSidebar}
        aria-hidden="true"
      />
      <aside
        className={cn(
          'fixed left-0 top-0 z-40 h-screen border-r bg-background/95 md:bg-background/60 backdrop-blur-xl shadow-lg transition-all duration-300',
          // Mobile: hidden when collapsed, full width when open
          sidebarCollapsed
            ? '-translate-x-full md:translate-x-0 w-64 md:w-16'
            : 'translate-x-0 w-64'
        )}
        role="navigation"
        aria-label="Main sidebar"
      >
        {/* Logo/Brand */}
        <div className={cn(
          'flex h-16 items-center border-b border-border/50 px-4',
          sidebarCollapsed ? 'justify-center' : 'justify-between'
        )}>
          {!sidebarCollapsed && (
            <span className="text-xl font-bold bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent truncate">
              OHPM
            </span>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleSidebar}
            className="h-8 w-8"
            aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex flex-col gap-1 p-2" role="navigation" aria-label="Main navigation">
          {navItems.map((item) => {
            // Fix: Use exact match for root, and ensure path segment match for others
            // This prevents /alerts from incorrectly matching /alert
            const isActive = location.pathname === item.to ||
              (item.to !== '/' && (
                location.pathname === item.to ||
                location.pathname.startsWith(`${item.to}/`)
              ));

            const link = (
              <NavLink
                key={item.to}
                to={item.to}
                aria-label={item.label}
                className={cn(
                  'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors',
                  'hover:bg-accent hover:text-accent-foreground',
                  isActive
                    ? 'bg-accent text-accent-foreground'
                    : 'text-muted-foreground',
                  sidebarCollapsed && 'justify-center'
                )}
              >
                <item.icon className="h-5 w-5 shrink-0" />
                {!sidebarCollapsed ? (
                  <span>{item.label}</span>
                ) : (
                  <span className="sr-only">{item.label}</span>
                )}
              </NavLink>
            );

            if (sidebarCollapsed) {
              return (
                <Tooltip key={item.to} delayDuration={0}>
                  <TooltipTrigger asChild>{link}</TooltipTrigger>
                  <TooltipContent side="right">{item.label}</TooltipContent>
                </Tooltip>
              );
            }

            return link;
          })}
        </nav>
      </aside>
    </TooltipProvider>
  );
}
