/**
 * UI Store Tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import {
  useUIStore,
  useSidebar,
  useTheme,
  useModal,
  useNotifications,
  useBreakpoint,
  selectSidebarCollapsed,
  selectTheme,
  selectNotifications,
  selectModalOpen,
  selectModalData,
} from './ui-store';

describe('useUIStore', () => {
  beforeEach(() => {
    // Reset store before each test
    act(() => {
      useUIStore.getState().reset();
    });
  });

  describe('initial state', () => {
    it('has sidebar expanded', () => {
      const { result } = renderHook(() => useUIStore());
      expect(result.current.sidebarCollapsed).toBe(false);
    });

    it('has mobile sidebar closed', () => {
      const { result } = renderHook(() => useUIStore());
      expect(result.current.sidebarMobileOpen).toBe(false);
    });

    it('has system theme', () => {
      const { result } = renderHook(() => useUIStore());
      expect(result.current.theme).toBe('system');
    });

    it('has light resolved theme', () => {
      const { result } = renderHook(() => useUIStore());
      expect(result.current.resolvedTheme).toBe('light');
    });

    it('has empty modals', () => {
      const { result } = renderHook(() => useUIStore());
      expect(result.current.modals).toEqual({});
    });

    it('has empty notifications', () => {
      const { result } = renderHook(() => useUIStore());
      expect(result.current.notifications).toEqual([]);
    });

    it('has command palette closed', () => {
      const { result } = renderHook(() => useUIStore());
      expect(result.current.commandPaletteOpen).toBe(false);
    });

    it('is not globally loading', () => {
      const { result } = renderHook(() => useUIStore());
      expect(result.current.globalLoading).toBe(false);
    });

    it('defaults to desktop breakpoint', () => {
      const { result } = renderHook(() => useUIStore());
      expect(result.current.breakpoint).toBe('desktop');
      expect(result.current.isDesktop).toBe(true);
      expect(result.current.isMobile).toBe(false);
      expect(result.current.isTablet).toBe(false);
    });
  });

  describe('sidebar actions', () => {
    it('toggleSidebar toggles collapsed state', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.toggleSidebar();
      });

      expect(result.current.sidebarCollapsed).toBe(true);

      act(() => {
        result.current.toggleSidebar();
      });

      expect(result.current.sidebarCollapsed).toBe(false);
    });

    it('setSidebarCollapsed sets specific state', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setSidebarCollapsed(true);
      });

      expect(result.current.sidebarCollapsed).toBe(true);

      act(() => {
        result.current.setSidebarCollapsed(false);
      });

      expect(result.current.sidebarCollapsed).toBe(false);
    });

    it('toggleMobileSidebar toggles mobile open state', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.toggleMobileSidebar();
      });

      expect(result.current.sidebarMobileOpen).toBe(true);

      act(() => {
        result.current.toggleMobileSidebar();
      });

      expect(result.current.sidebarMobileOpen).toBe(false);
    });

    it('setMobileSidebarOpen sets specific state', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setMobileSidebarOpen(true);
      });

      expect(result.current.sidebarMobileOpen).toBe(true);
    });
  });

  describe('theme actions', () => {
    it('setTheme changes theme', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setTheme('dark');
      });

      expect(result.current.theme).toBe('dark');

      act(() => {
        result.current.setTheme('light');
      });

      expect(result.current.theme).toBe('light');
    });

    it('setResolvedTheme changes resolved theme', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setResolvedTheme('dark');
      });

      expect(result.current.resolvedTheme).toBe('dark');
    });
  });

  describe('modal actions', () => {
    it('openModal opens a modal', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.openModal('test-modal');
      });

      expect(result.current.modals['test-modal']).toEqual({
        id: 'test-modal',
        isOpen: true,
      });
    });

    it('openModal opens a modal with data', () => {
      const { result } = renderHook(() => useUIStore());
      const modalData = { userId: 123 };

      act(() => {
        result.current.openModal('test-modal', modalData);
      });

      expect(result.current.modals['test-modal']).toEqual({
        id: 'test-modal',
        isOpen: true,
        data: modalData,
      });
    });

    it('closeModal closes a modal', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.openModal('test-modal');
      });

      act(() => {
        result.current.closeModal('test-modal');
      });

      expect(result.current.modals['test-modal'].isOpen).toBe(false);
    });

    it('closeModal does nothing for non-existent modal', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.closeModal('non-existent');
      });

      expect(result.current.modals['non-existent']).toBeUndefined();
    });

    it('toggleModal toggles modal state', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.toggleModal('test-modal');
      });

      expect(result.current.modals['test-modal'].isOpen).toBe(true);

      act(() => {
        result.current.toggleModal('test-modal');
      });

      expect(result.current.modals['test-modal'].isOpen).toBe(false);
    });

    it('closeAllModals closes all open modals', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.openModal('modal1');
        result.current.openModal('modal2');
        result.current.openModal('modal3');
      });

      act(() => {
        result.current.closeAllModals();
      });

      expect(result.current.modals['modal1'].isOpen).toBe(false);
      expect(result.current.modals['modal2'].isOpen).toBe(false);
      expect(result.current.modals['modal3'].isOpen).toBe(false);
    });

    it('getModalData returns modal data', () => {
      const { result } = renderHook(() => useUIStore());
      const data = { test: 'data' };

      act(() => {
        result.current.openModal('test-modal', data);
      });

      expect(result.current.getModalData('test-modal')).toEqual(data);
    });

    it('getModalData returns undefined for non-existent modal', () => {
      const { result } = renderHook(() => useUIStore());

      expect(result.current.getModalData('non-existent')).toBeUndefined();
    });
  });

  describe('notification actions', () => {
    it('addNotification adds a notification', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.addNotification({
          type: 'info',
          title: 'Test',
        });
      });

      expect(result.current.notifications).toHaveLength(1);
      expect(result.current.notifications[0].title).toBe('Test');
      expect(result.current.notifications[0].type).toBe('info');
    });

    it('addNotification returns notification id', () => {
      const { result } = renderHook(() => useUIStore());
      let id: string | undefined;

      act(() => {
        id = result.current.addNotification({
          type: 'success',
          title: 'Success',
        });
      });

      expect(id).toBeDefined();
      expect(id).toContain('notification-');
    });

    it('addNotification with message', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.addNotification({
          type: 'warning',
          title: 'Warning',
          message: 'This is a warning message',
        });
      });

      expect(result.current.notifications[0].message).toBe('This is a warning message');
    });

    it('limits notifications to maxNotifications', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        for (let i = 0; i < 10; i++) {
          result.current.addNotification({
            type: 'info',
            title: `Notification ${i}`,
          });
        }
      });

      expect(result.current.notifications.length).toBeLessThanOrEqual(result.current.maxNotifications);
    });

    it('removeNotification removes specific notification', () => {
      const { result } = renderHook(() => useUIStore());
      let id: string | undefined;

      act(() => {
        id = result.current.addNotification({
          type: 'info',
          title: 'To Remove',
        });
      });

      act(() => {
        result.current.removeNotification(id!);
      });

      expect(result.current.notifications).toHaveLength(0);
    });

    it('clearNotifications removes all notifications', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.addNotification({ type: 'info', title: '1' });
        result.current.addNotification({ type: 'info', title: '2' });
        result.current.addNotification({ type: 'info', title: '3' });
      });

      act(() => {
        result.current.clearNotifications();
      });

      expect(result.current.notifications).toHaveLength(0);
    });
  });

  describe('command palette actions', () => {
    it('openCommandPalette opens palette', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.openCommandPalette();
      });

      expect(result.current.commandPaletteOpen).toBe(true);
    });

    it('closeCommandPalette closes palette', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.openCommandPalette();
      });

      act(() => {
        result.current.closeCommandPalette();
      });

      expect(result.current.commandPaletteOpen).toBe(false);
    });

    it('toggleCommandPalette toggles palette', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.toggleCommandPalette();
      });

      expect(result.current.commandPaletteOpen).toBe(true);

      act(() => {
        result.current.toggleCommandPalette();
      });

      expect(result.current.commandPaletteOpen).toBe(false);
    });
  });

  describe('loading actions', () => {
    it('setGlobalLoading sets loading state', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setGlobalLoading(true);
      });

      expect(result.current.globalLoading).toBe(true);
    });

    it('setGlobalLoading sets loading with message', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setGlobalLoading(true, 'Loading data...');
      });

      expect(result.current.globalLoading).toBe(true);
      expect(result.current.loadingMessage).toBe('Loading data...');
    });
  });

  describe('breakpoint actions', () => {
    it('setBreakpoint sets mobile', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setBreakpoint('mobile');
      });

      expect(result.current.breakpoint).toBe('mobile');
      expect(result.current.isMobile).toBe(true);
      expect(result.current.isTablet).toBe(false);
      expect(result.current.isDesktop).toBe(false);
    });

    it('setBreakpoint sets tablet', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setBreakpoint('tablet');
      });

      expect(result.current.breakpoint).toBe('tablet');
      expect(result.current.isMobile).toBe(false);
      expect(result.current.isTablet).toBe(true);
      expect(result.current.isDesktop).toBe(false);
    });

    it('setBreakpoint sets desktop', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setBreakpoint('desktop');
      });

      expect(result.current.breakpoint).toBe('desktop');
      expect(result.current.isMobile).toBe(false);
      expect(result.current.isTablet).toBe(false);
      expect(result.current.isDesktop).toBe(true);
    });
  });

  describe('reset', () => {
    it('resets all state to initial', () => {
      const { result } = renderHook(() => useUIStore());

      act(() => {
        result.current.setSidebarCollapsed(true);
        result.current.setTheme('dark');
        result.current.openModal('modal');
        result.current.addNotification({ type: 'info', title: 'Test' });
        result.current.openCommandPalette();
        result.current.setGlobalLoading(true);
        result.current.setBreakpoint('mobile');
      });

      act(() => {
        result.current.reset();
      });

      expect(result.current.sidebarCollapsed).toBe(false);
      expect(result.current.theme).toBe('system');
      expect(result.current.modals).toEqual({});
      expect(result.current.notifications).toEqual([]);
      expect(result.current.commandPaletteOpen).toBe(false);
      expect(result.current.globalLoading).toBe(false);
      expect(result.current.breakpoint).toBe('desktop');
    });
  });

  describe('selectors', () => {
    it('selectSidebarCollapsed returns collapsed state', () => {
      const state = useUIStore.getState();
      expect(selectSidebarCollapsed(state)).toBe(false);
    });

    it('selectTheme returns theme', () => {
      const state = useUIStore.getState();
      expect(selectTheme(state)).toBe('system');
    });

    it('selectNotifications returns notifications array', () => {
      const state = useUIStore.getState();
      expect(selectNotifications(state)).toEqual([]);
    });

    it('selectModalOpen returns modal open state', () => {
      act(() => {
        useUIStore.getState().openModal('test');
      });

      const state = useUIStore.getState();
      expect(selectModalOpen('test')(state)).toBe(true);
      expect(selectModalOpen('other')(state)).toBe(false);
    });

    it('selectModalData returns modal data', () => {
      act(() => {
        useUIStore.getState().openModal('test', { value: 42 });
      });

      const state = useUIStore.getState();
      expect(selectModalData('test')(state)).toEqual({ value: 42 });
    });
  });
});

describe('useSidebar hook', () => {
  beforeEach(() => {
    act(() => {
      useUIStore.getState().reset();
    });
  });

  it('returns sidebar state', () => {
    const { result } = renderHook(() => useSidebar());

    expect(result.current.collapsed).toBe(false);
    expect(result.current.mobileOpen).toBe(false);
  });

  it('provides toggle function', () => {
    const { result } = renderHook(() => useSidebar());

    act(() => {
      result.current.toggle();
    });

    expect(result.current.collapsed).toBe(true);
  });

  it('provides setCollapsed function', () => {
    const { result } = renderHook(() => useSidebar());

    act(() => {
      result.current.setCollapsed(true);
    });

    expect(result.current.collapsed).toBe(true);
  });
});

describe('useTheme hook', () => {
  beforeEach(() => {
    act(() => {
      useUIStore.getState().reset();
    });
  });

  it('returns theme state', () => {
    const { result } = renderHook(() => useTheme());

    expect(result.current.theme).toBe('system');
    expect(result.current.resolvedTheme).toBe('light');
  });

  it('provides isDark and isLight', () => {
    const { result } = renderHook(() => useTheme());

    expect(result.current.isDark).toBe(false);
    expect(result.current.isLight).toBe(true);
  });

  it('provides setTheme function', () => {
    const { result } = renderHook(() => useTheme());

    act(() => {
      result.current.setTheme('dark');
    });

    expect(result.current.theme).toBe('dark');
  });
});

describe('useModal hook', () => {
  beforeEach(() => {
    act(() => {
      useUIStore.getState().reset();
    });
  });

  it('returns modal state', () => {
    const { result } = renderHook(() => useModal('test-modal'));

    expect(result.current.isOpen).toBe(false);
    expect(result.current.data).toBeUndefined();
  });

  it('provides open function', () => {
    const { result } = renderHook(() => useModal('test-modal'));

    act(() => {
      result.current.open();
    });

    expect(result.current.isOpen).toBe(true);
  });

  it('provides open function with data', () => {
    const { result } = renderHook(() => useModal('test-modal'));

    act(() => {
      result.current.open({ userId: 1 });
    });

    expect(result.current.isOpen).toBe(true);
    expect(result.current.data).toEqual({ userId: 1 });
  });

  it('provides close function', () => {
    const { result } = renderHook(() => useModal('test-modal'));

    act(() => {
      result.current.open();
    });

    act(() => {
      result.current.close();
    });

    expect(result.current.isOpen).toBe(false);
  });

  it('provides toggle function', () => {
    const { result } = renderHook(() => useModal('test-modal'));

    act(() => {
      result.current.toggle();
    });

    expect(result.current.isOpen).toBe(true);

    act(() => {
      result.current.toggle();
    });

    expect(result.current.isOpen).toBe(false);
  });
});

describe('useNotifications hook', () => {
  beforeEach(() => {
    act(() => {
      useUIStore.getState().reset();
    });
  });

  it('returns notifications state', () => {
    const { result } = renderHook(() => useNotifications());

    expect(result.current.notifications).toEqual([]);
  });

  it('provides add function', () => {
    const { result } = renderHook(() => useNotifications());

    act(() => {
      result.current.add({ type: 'info', title: 'Test' });
    });

    expect(result.current.notifications).toHaveLength(1);
  });

  it('provides info convenience method', () => {
    const { result } = renderHook(() => useNotifications());

    act(() => {
      result.current.info('Info Title', 'Info message');
    });

    expect(result.current.notifications[0].type).toBe('info');
    expect(result.current.notifications[0].title).toBe('Info Title');
  });

  it('provides success convenience method', () => {
    const { result } = renderHook(() => useNotifications());

    act(() => {
      result.current.success('Success Title');
    });

    expect(result.current.notifications[0].type).toBe('success');
  });

  it('provides warning convenience method', () => {
    const { result } = renderHook(() => useNotifications());

    act(() => {
      result.current.warning('Warning Title');
    });

    expect(result.current.notifications[0].type).toBe('warning');
  });

  it('provides error convenience method', () => {
    const { result } = renderHook(() => useNotifications());

    act(() => {
      result.current.error('Error Title');
    });

    expect(result.current.notifications[0].type).toBe('error');
  });

  it('provides remove function', () => {
    const { result } = renderHook(() => useNotifications());
    let id: string | undefined;

    act(() => {
      id = result.current.add({ type: 'info', title: 'Test' });
    });

    act(() => {
      result.current.remove(id!);
    });

    expect(result.current.notifications).toHaveLength(0);
  });

  it('provides clear function', () => {
    const { result } = renderHook(() => useNotifications());

    act(() => {
      result.current.add({ type: 'info', title: '1' });
      result.current.add({ type: 'info', title: '2' });
    });

    act(() => {
      result.current.clear();
    });

    expect(result.current.notifications).toHaveLength(0);
  });
});

describe('useBreakpoint hook', () => {
  beforeEach(() => {
    act(() => {
      useUIStore.getState().reset();
    });
  });

  it('returns breakpoint state', () => {
    const { result } = renderHook(() => useBreakpoint());

    expect(result.current.breakpoint).toBe('desktop');
    expect(result.current.isDesktop).toBe(true);
  });

  it('provides setBreakpoint function', () => {
    const { result } = renderHook(() => useBreakpoint());

    act(() => {
      result.current.setBreakpoint('mobile');
    });

    expect(result.current.breakpoint).toBe('mobile');
    expect(result.current.isMobile).toBe(true);
  });
});
