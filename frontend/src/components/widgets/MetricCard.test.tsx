/**
 * MetricCard Widget Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Activity } from 'lucide-react';
import { MetricCard } from './MetricCard';

describe('MetricCard', () => {
  describe('rendering', () => {
    it('renders title and value', () => {
      render(<MetricCard title="Total Requests" value={5000} />);

      expect(screen.getByText('Total Requests')).toBeInTheDocument();
      expect(screen.getByText('5000')).toBeInTheDocument();
    });

    it('renders string value', () => {
      render(<MetricCard title="Status" value="Healthy" />);

      expect(screen.getByText('Healthy')).toBeInTheDocument();
    });

    it('renders with icon', () => {
      render(<MetricCard title="Activity" value={100} icon={Activity} />);

      expect(screen.getByText('Activity')).toBeInTheDocument();
    });

    it('renders description when provided', () => {
      render(
        <MetricCard
          title="API Calls"
          value="10,000"
          description="Calls this hour"
        />
      );

      expect(screen.getByText('Calls this hour')).toBeInTheDocument();
    });
  });

  describe('loading state', () => {
    it('renders skeleton when loading', () => {
      const { container } = render(
        <MetricCard title="Loading" value={0} isLoading />
      );

      const skeletons = container.querySelectorAll('[class*="animate-pulse"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    it('does not render value when loading', () => {
      render(<MetricCard title="Loading" value={1000} isLoading />);

      expect(screen.queryByText('1000')).not.toBeInTheDocument();
    });

    it('renders skeleton for icon area when icon provided and loading', () => {
      const { container } = render(
        <MetricCard title="Loading" value={0} icon={Activity} isLoading />
      );

      const skeletons = container.querySelectorAll('[class*="animate-pulse"]');
      expect(skeletons.length).toBeGreaterThan(1);
    });
  });

  describe('trend indicator', () => {
    it('renders trend badge with positive value', () => {
      render(
        <MetricCard
          title="Users"
          value={100}
          trend={{ value: 15, label: 'from yesterday', isPositive: true }}
        />
      );

      expect(screen.getByText('+15%')).toBeInTheDocument();
      expect(screen.getByText('from yesterday')).toBeInTheDocument();
    });

    it('renders trend badge with negative value', () => {
      render(
        <MetricCard
          title="Users"
          value={100}
          trend={{ value: -8, label: 'from last week', isPositive: false }}
        />
      );

      expect(screen.getByText('-8%')).toBeInTheDocument();
    });

    it('handles zero trend value', () => {
      render(
        <MetricCard
          title="Users"
          value={100}
          trend={{ value: 0, label: 'no change', isPositive: true }}
        />
      );

      expect(screen.getByText('+0%')).toBeInTheDocument();
    });
  });

  describe('variants', () => {
    it('renders default variant', () => {
      const { container } = render(
        <MetricCard title="Default" value={100} variant="default" />
      );

      // Should not have variant-specific styling
      expect(container.querySelector('[class*="border-green"]')).not.toBeInTheDocument();
    });

    it('renders success variant', () => {
      const { container } = render(
        <MetricCard title="Success" value={100} variant="success" />
      );

      expect(container.querySelector('[class*="border-green"]')).toBeInTheDocument();
    });

    it('renders warning variant', () => {
      const { container } = render(
        <MetricCard title="Warning" value={100} variant="warning" />
      );

      expect(container.querySelector('[class*="border-yellow"]')).toBeInTheDocument();
    });

    it('renders destructive variant', () => {
      const { container } = render(
        <MetricCard title="Error" value={100} variant="destructive" />
      );

      expect(container.querySelector('[class*="border-red"]')).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <MetricCard title="Test" value={1} className="custom-metric" />
      );

      expect(container.querySelector('.custom-metric')).toBeInTheDocument();
    });
  });
});
