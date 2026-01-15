/**
 * Tabs Component Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './tabs';

describe('Tabs', () => {
  const renderTabs = (props = {}) => {
    return render(
      <Tabs defaultValue="tab1" {...props}>
        <TabsList>
          <TabsTrigger value="tab1">Tab 1</TabsTrigger>
          <TabsTrigger value="tab2">Tab 2</TabsTrigger>
          <TabsTrigger value="tab3">Tab 3</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content 1</TabsContent>
        <TabsContent value="tab2">Content 2</TabsContent>
        <TabsContent value="tab3">Content 3</TabsContent>
      </Tabs>
    );
  };

  describe('rendering', () => {
    it('renders tabs list', () => {
      renderTabs();
      expect(screen.getByRole('tablist')).toBeInTheDocument();
    });

    it('renders all tab triggers', () => {
      renderTabs();
      expect(screen.getByRole('tab', { name: 'Tab 1' })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'Tab 2' })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: 'Tab 3' })).toBeInTheDocument();
    });

    it('renders default active content', () => {
      renderTabs();
      expect(screen.getByText('Content 1')).toBeInTheDocument();
    });

    it('shows only active content', () => {
      renderTabs();
      // Active content is in visible tabpanel
      const tabpanel = screen.getByRole('tabpanel');
      expect(tabpanel).toHaveTextContent('Content 1');
    });
  });

  describe('interaction', () => {
    it('switches content on tab click', async () => {
      const user = userEvent.setup();
      renderTabs();

      await user.click(screen.getByRole('tab', { name: 'Tab 2' }));
      expect(screen.getByText('Content 2')).toBeVisible();
    });

    it('updates active tab state', async () => {
      const user = userEvent.setup();
      renderTabs();

      const tab2 = screen.getByRole('tab', { name: 'Tab 2' });
      await user.click(tab2);
      expect(tab2).toHaveAttribute('data-state', 'active');
    });

    it('deactivates previous tab', async () => {
      const user = userEvent.setup();
      renderTabs();

      await user.click(screen.getByRole('tab', { name: 'Tab 2' }));
      expect(screen.getByRole('tab', { name: 'Tab 1' })).toHaveAttribute('data-state', 'inactive');
    });
  });

  describe('keyboard navigation', () => {
    it('moves focus with arrow keys', async () => {
      const user = userEvent.setup();
      renderTabs();

      await user.tab();
      expect(screen.getByRole('tab', { name: 'Tab 1' })).toHaveFocus();

      await user.keyboard('{ArrowRight}');
      expect(screen.getByRole('tab', { name: 'Tab 2' })).toHaveFocus();
    });

    it('wraps around with arrow keys', async () => {
      const user = userEvent.setup();
      renderTabs();

      await user.tab();
      await user.keyboard('{ArrowLeft}');
      expect(screen.getByRole('tab', { name: 'Tab 3' })).toHaveFocus();
    });

    it('activates tab on Enter', async () => {
      const user = userEvent.setup();
      renderTabs();

      await user.tab();
      await user.keyboard('{ArrowRight}');
      await user.keyboard('{Enter}');
      expect(screen.getByText('Content 2')).toBeVisible();
    });

    it('activates tab on Space', async () => {
      const user = userEvent.setup();
      renderTabs();

      await user.tab();
      await user.keyboard('{ArrowRight}');
      await user.keyboard(' ');
      expect(screen.getByText('Content 2')).toBeVisible();
    });
  });

  describe('controlled mode', () => {
    it('supports controlled value', async () => {
      const handleChange = vi.fn();
      render(
        <Tabs value="tab1" onValueChange={handleChange}>
          <TabsList>
            <TabsTrigger value="tab1">Tab 1</TabsTrigger>
            <TabsTrigger value="tab2">Tab 2</TabsTrigger>
          </TabsList>
          <TabsContent value="tab1">Content 1</TabsContent>
          <TabsContent value="tab2">Content 2</TabsContent>
        </Tabs>
      );

      const user = userEvent.setup();
      await user.click(screen.getByRole('tab', { name: 'Tab 2' }));
      expect(handleChange).toHaveBeenCalledWith('tab2');
    });
  });

  describe('disabled state', () => {
    it('disables individual tabs', async () => {
      const user = userEvent.setup();
      render(
        <Tabs defaultValue="tab1">
          <TabsList>
            <TabsTrigger value="tab1">Tab 1</TabsTrigger>
            <TabsTrigger value="tab2" disabled>Tab 2</TabsTrigger>
          </TabsList>
          <TabsContent value="tab1">Content 1</TabsContent>
          <TabsContent value="tab2">Content 2</TabsContent>
        </Tabs>
      );

      const disabledTab = screen.getByRole('tab', { name: 'Tab 2' });
      expect(disabledTab).toBeDisabled();

      await user.click(disabledTab);
      expect(screen.getByText('Content 1')).toBeVisible();
    });
  });

  describe('styling', () => {
    it('TabsList has correct base classes', () => {
      renderTabs();
      const tabsList = screen.getByRole('tablist');
      expect(tabsList).toHaveClass('inline-flex', 'items-center', 'rounded-md', 'bg-muted');
    });

    it('active tab has correct styling', () => {
      renderTabs();
      const activeTab = screen.getByRole('tab', { name: 'Tab 1' });
      expect(activeTab).toHaveAttribute('data-state', 'active');
    });

    it('TabsTrigger has focus ring', () => {
      renderTabs();
      const tab = screen.getByRole('tab', { name: 'Tab 1' });
      expect(tab).toHaveClass('focus-visible:ring-2');
    });

    it('supports custom className on TabsList', () => {
      render(
        <Tabs defaultValue="tab1">
          <TabsList className="custom-list">
            <TabsTrigger value="tab1">Tab 1</TabsTrigger>
          </TabsList>
          <TabsContent value="tab1">Content</TabsContent>
        </Tabs>
      );
      expect(screen.getByRole('tablist')).toHaveClass('custom-list');
    });

    it('supports custom className on TabsTrigger', () => {
      render(
        <Tabs defaultValue="tab1">
          <TabsList>
            <TabsTrigger value="tab1" className="custom-trigger">Tab 1</TabsTrigger>
          </TabsList>
          <TabsContent value="tab1">Content</TabsContent>
        </Tabs>
      );
      expect(screen.getByRole('tab')).toHaveClass('custom-trigger');
    });

    it('supports custom className on TabsContent', () => {
      render(
        <Tabs defaultValue="tab1">
          <TabsList>
            <TabsTrigger value="tab1">Tab 1</TabsTrigger>
          </TabsList>
          <TabsContent value="tab1" className="custom-content">Content</TabsContent>
        </Tabs>
      );
      expect(screen.getByRole('tabpanel')).toHaveClass('custom-content');
    });
  });

  describe('accessibility', () => {
    it('has correct ARIA attributes', () => {
      renderTabs();
      const tab = screen.getByRole('tab', { name: 'Tab 1' });
      expect(tab).toHaveAttribute('aria-selected', 'true');
    });

    it('tabpanel has correct role', () => {
      renderTabs();
      expect(screen.getByRole('tabpanel')).toBeInTheDocument();
    });

    it('tab controls tabpanel', () => {
      renderTabs();
      const tab = screen.getByRole('tab', { name: 'Tab 1' });
      const panel = screen.getByRole('tabpanel');
      expect(tab).toHaveAttribute('aria-controls', panel.id);
    });
  });

  describe('display names', () => {
    it('TabsList has correct display name', () => {
      expect(TabsList.displayName).toBe('TabsList');
    });

    it('TabsTrigger has correct display name', () => {
      expect(TabsTrigger.displayName).toBe('TabsTrigger');
    });

    it('TabsContent has correct display name', () => {
      expect(TabsContent.displayName).toBe('TabsContent');
    });
  });
});
