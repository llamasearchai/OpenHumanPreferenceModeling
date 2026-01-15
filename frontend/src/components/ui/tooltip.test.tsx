/**
 * Tooltip Component Tests
 *
 * Note: Radix UI Tooltip renders content in both visible div and hidden span.
 * These tests use getAllByText and select the first/visible element.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from './tooltip';

describe('Tooltip', () => {
  describe('rendering', () => {
    it('renders trigger element', () => {
      render(
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <button>Hover me</button>
            </TooltipTrigger>
            <TooltipContent>Tooltip content</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      expect(screen.getByRole('button', { name: 'Hover me' })).toBeInTheDocument();
    });

    it('tooltip content is hidden by default', () => {
      render(
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger>Trigger</TooltipTrigger>
            <TooltipContent>Hidden Content</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      expect(screen.queryByText('Hidden Content')).not.toBeInTheDocument();
    });

    it('accepts custom trigger text', () => {
      render(
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger>Custom Trigger</TooltipTrigger>
            <TooltipContent>Content</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      expect(screen.getByText('Custom Trigger')).toBeInTheDocument();
    });
  });

  describe('controlled mode', () => {
    it('shows content when open prop is true', () => {
      render(
        <TooltipProvider>
          <Tooltip open>
            <TooltipTrigger>Trigger</TooltipTrigger>
            <TooltipContent>Always visible</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      // Radix renders content in multiple places - check at least one exists
      const contents = screen.getAllByText('Always visible');
      expect(contents.length).toBeGreaterThan(0);
    });

    it('shows content when defaultOpen is true', () => {
      render(
        <TooltipProvider>
          <Tooltip defaultOpen>
            <TooltipTrigger>Trigger</TooltipTrigger>
            <TooltipContent>Open by default</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      const contents = screen.getAllByText('Open by default');
      expect(contents.length).toBeGreaterThan(0);
    });
  });

  describe('styling', () => {
    it('applies custom className to content when open', () => {
      render(
        <TooltipProvider>
          <Tooltip open>
            <TooltipTrigger>Trigger</TooltipTrigger>
            <TooltipContent className="custom-tooltip">Content</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      const contents = screen.getAllByText('Content');
      const visibleContent = contents.find(el => el.closest('.custom-tooltip'));
      expect(visibleContent).toBeTruthy();
    });

    it('has base styling classes when open', () => {
      render(
        <TooltipProvider>
          <Tooltip open>
            <TooltipTrigger>Trigger</TooltipTrigger>
            <TooltipContent>Styled Content</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      const contents = screen.getAllByText('Styled Content');
      const styledContent = contents.find(el => el.closest('.z-50'));
      expect(styledContent).toBeTruthy();
    });
  });

  describe('rich content', () => {
    it('renders complex content when open', () => {
      render(
        <TooltipProvider>
          <Tooltip open>
            <TooltipTrigger>Trigger</TooltipTrigger>
            <TooltipContent>
              <div data-testid="complex-content">
                <strong>Title</strong>
                <p>Description</p>
              </div>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      // Complex content should be rendered
      expect(screen.getAllByTestId('complex-content').length).toBeGreaterThan(0);
    });
  });

  describe('accessibility', () => {
    it('tooltip has role="tooltip" when open', () => {
      render(
        <TooltipProvider>
          <Tooltip open>
            <TooltipTrigger>Trigger</TooltipTrigger>
            <TooltipContent>Tooltip text</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      expect(screen.getByRole('tooltip')).toBeInTheDocument();
    });
  });

  describe('display names', () => {
    it('TooltipContent has correct display name', () => {
      expect(TooltipContent.displayName).toBe('TooltipContent');
    });
  });

  describe('ref forwarding', () => {
    it('forwards ref to TooltipContent', () => {
      const ref = { current: null as HTMLDivElement | null };
      render(
        <TooltipProvider>
          <Tooltip open>
            <TooltipTrigger>Trigger</TooltipTrigger>
            <TooltipContent ref={ref}>Content</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      expect(ref.current).toBeInstanceOf(HTMLDivElement);
    });
  });

  describe('props', () => {
    it('accepts sideOffset prop', () => {
      render(
        <TooltipProvider>
          <Tooltip open>
            <TooltipTrigger>Trigger</TooltipTrigger>
            <TooltipContent sideOffset={10}>Offset Content</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      const contents = screen.getAllByText('Offset Content');
      expect(contents.length).toBeGreaterThan(0);
    });

    it('accepts side prop', () => {
      render(
        <TooltipProvider>
          <Tooltip open>
            <TooltipTrigger>Trigger</TooltipTrigger>
            <TooltipContent side="right">Side Content</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
      // Find element with data-side attribute
      const sideContent = document.querySelector('[data-side="right"]');
      expect(sideContent).toBeInTheDocument();
    });
  });
});
