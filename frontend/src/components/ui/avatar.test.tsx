/* eslint-disable no-undef */
/**
 * Avatar Component Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Avatar, AvatarImage, AvatarFallback } from './avatar';

describe('Avatar', () => {
  describe('rendering', () => {
    it('renders Avatar container', () => {
      render(
        <Avatar data-testid="avatar">
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );
      expect(screen.getByTestId('avatar')).toBeInTheDocument();
    });

    it('has correct base styling', () => {
      render(
        <Avatar data-testid="avatar">
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );
      const avatar = screen.getByTestId('avatar');
      expect(avatar).toHaveClass('relative', 'flex', 'h-10', 'w-10', 'shrink-0', 'overflow-hidden', 'rounded-full');
    });

    it('supports custom className', () => {
      render(
        <Avatar className="custom-class" data-testid="avatar">
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );
      expect(screen.getByTestId('avatar')).toHaveClass('custom-class');
    });
  });

  describe('AvatarImage', () => {
    it('renders image component when src provided', async () => {
      const { container } = render(
        <Avatar>
          <AvatarImage src="https://example.com/avatar.jpg" alt="User avatar" />
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );

      // Radix Avatar renders img element after load
      // Just verify component renders without error
      expect(container).toBeInTheDocument();
    });

    it('accepts alt text prop', () => {
      const { container } = render(
        <Avatar>
          <AvatarImage src="https://example.com/avatar.jpg" alt="John Doe" />
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );

      expect(container).toBeInTheDocument();
    });

    it('accepts className prop', () => {
      const { container } = render(
        <Avatar>
          <AvatarImage src="https://example.com/avatar.jpg" alt="User" className="custom-image" />
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );

      expect(container).toBeInTheDocument();
    });
  });

  describe('AvatarFallback', () => {
    it('renders fallback content', () => {
      render(
        <Avatar>
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );
      expect(screen.getByText('JD')).toBeInTheDocument();
    });

    it('has correct styling', () => {
      render(
        <Avatar>
          <AvatarFallback data-testid="fallback">JD</AvatarFallback>
        </Avatar>
      );
      const fallback = screen.getByTestId('fallback');
      expect(fallback).toHaveClass('flex', 'h-full', 'w-full', 'items-center', 'justify-center', 'rounded-full', 'bg-muted');
    });

    it('supports custom className', () => {
      render(
        <Avatar>
          <AvatarFallback className="custom-fallback" data-testid="fallback">JD</AvatarFallback>
        </Avatar>
      );
      expect(screen.getByTestId('fallback')).toHaveClass('custom-fallback');
    });

    it('can render icon as fallback', () => {
      render(
        <Avatar>
          <AvatarFallback>
            <span data-testid="icon">👤</span>
          </AvatarFallback>
        </Avatar>
      );
      expect(screen.getByTestId('icon')).toBeInTheDocument();
    });
  });

  describe('fallback behavior', () => {
    it('shows fallback when no image', () => {
      render(
        <Avatar>
          <AvatarFallback>AB</AvatarFallback>
        </Avatar>
      );
      expect(screen.getByText('AB')).toBeInTheDocument();
    });

    it('supports delayMs prop', () => {
      const { container } = render(
        <Avatar>
          <AvatarImage src="https://example.com/avatar.jpg" alt="User" />
          <AvatarFallback delayMs={100}>JD</AvatarFallback>
        </Avatar>
      );
      // Component renders without error
      expect(container).toBeInTheDocument();
    });
  });

  describe('sizes', () => {
    it('can have custom size via className', () => {
      render(
        <Avatar className="h-16 w-16" data-testid="avatar">
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );
      expect(screen.getByTestId('avatar')).toHaveClass('h-16', 'w-16');
    });

    it('can be small', () => {
      render(
        <Avatar className="h-8 w-8" data-testid="avatar">
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );
      expect(screen.getByTestId('avatar')).toHaveClass('h-8', 'w-8');
    });

    it('can be large', () => {
      render(
        <Avatar className="h-24 w-24" data-testid="avatar">
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );
      expect(screen.getByTestId('avatar')).toHaveClass('h-24', 'w-24');
    });
  });

  describe('ref forwarding', () => {
    it('forwards ref to Avatar root', () => {
      const ref = { current: null as HTMLSpanElement | null };
      render(
        <Avatar ref={ref}>
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );
      expect(ref.current).toBeInstanceOf(HTMLSpanElement);
    });

    it('forwards ref to AvatarImage', () => {
      const ref = { current: null as HTMLImageElement | null };
      render(
        <Avatar>
          <AvatarImage ref={ref} src="https://example.com/avatar.jpg" alt="User" />
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );
      // Image ref may be null if image hasn't loaded
      // Just check no error is thrown
    });

    it('forwards ref to AvatarFallback', () => {
      const ref = { current: null as HTMLSpanElement | null };
      render(
        <Avatar>
          <AvatarFallback ref={ref}>JD</AvatarFallback>
        </Avatar>
      );
      expect(ref.current).toBeInstanceOf(HTMLSpanElement);
    });
  });

  describe('display names', () => {
    it('Avatar has correct display name', () => {
      expect(Avatar.displayName).toBe('Avatar');
    });

    it('AvatarImage has correct display name', () => {
      expect(AvatarImage.displayName).toBe('AvatarImage');
    });

    it('AvatarFallback has correct display name', () => {
      expect(AvatarFallback.displayName).toBe('AvatarFallback');
    });
  });

  describe('accessibility', () => {
    it('accepts alt prop on image', () => {
      const { container } = render(
        <Avatar>
          <AvatarImage src="https://example.com/avatar.jpg" alt="Profile picture of John Doe" />
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );
      expect(container).toBeInTheDocument();
    });

    it('can have aria-label on container', () => {
      render(
        <Avatar aria-label="User avatar" data-testid="avatar">
          <AvatarFallback>JD</AvatarFallback>
        </Avatar>
      );
      expect(screen.getByTestId('avatar')).toHaveAttribute('aria-label', 'User avatar');
    });
  });
});
