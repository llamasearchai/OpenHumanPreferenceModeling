/**
 * Header Component
 *
 * Purpose: Top navigation bar with user menu
 */

import { Moon, Sun } from 'lucide-react';
import { selectTheme, useUIStore } from '@/stores/ui-store';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface HeaderProps {
  className?: string;
}

export function Header({ className }: HeaderProps) {
  const theme = useUIStore(selectTheme);
  const setTheme = useUIStore((state) => state.setTheme);

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  return (
    <header
      className={cn(
        'sticky top-0 z-30 flex h-16 items-center justify-between border-b bg-background px-4',
        className
      )}
    >
      {/* Page Title Area - can be filled by pages */}
      <div className="flex items-center gap-4">
        <div className="text-lg font-semibold">OpenHuman Preference Modeling</div>
      </div>

      {/* Right side actions */}
      <div className="flex items-center gap-2">
        {/* Theme Toggle */}
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleTheme}
          aria-label="Toggle theme"
        >
          {theme === 'light' ? (
            <Moon className="h-5 w-5" />
          ) : (
            <Sun className="h-5 w-5" />
          )}
        </Button>

      </div>
    </header>
  );
}
