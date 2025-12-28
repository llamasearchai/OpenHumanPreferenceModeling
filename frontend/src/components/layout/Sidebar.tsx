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
import { useUIStore } from '@/stores/ui-store';
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
  const { sidebarCollapsed, toggleSidebar } = useUIStore();

  return (
    <TooltipProvider>
      <aside
        className={cn(
          'fixed left-0 top-0 z-40 h-screen border-r bg-card transition-all duration-300',
          sidebarCollapsed ? 'w-16' : 'w-64'
        )}
      >
        {/* Logo/Brand */}
        <div className={cn(
          'flex h-16 items-center border-b px-4',
          sidebarCollapsed ? 'justify-center' : 'justify-between'
        )}>
          {!sidebarCollapsed && (
            <span className="text-lg font-bold truncate">OHPM</span>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleSidebar}
            className="h-8 w-8"
          >
            {sidebarCollapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Navigation */}
        <nav className="flex flex-col gap-1 p-2">
          {navItems.map((item) => {
            const isActive = location.pathname === item.to ||
              (item.to !== '/' && location.pathname.startsWith(item.to));

            const link = (
              <NavLink
                key={item.to}
                to={item.to}
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
                {!sidebarCollapsed && <span>{item.label}</span>}
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
