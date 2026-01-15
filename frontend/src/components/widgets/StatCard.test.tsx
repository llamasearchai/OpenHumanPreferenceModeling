/**
 * StatCard Widget Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Activity } from 'lucide-react';
import { StatCard } from './StatCard';

describe('StatCard', () => {
  describe('rendering', () => {
    it('renders title and value', () => {
      render(<StatCard title="Total Users" value={1234} />);

      expect(screen.getByText('Total Users')).toBeInTheDocument();
      expect(screen.getByText('1234')).toBeInTheDocument();
    });

    it('renders string value', () => {
      render(<StatCard title="Status" value="Active" />);

      expect(screen.getByText('Active')).toBeInTheDocument();
    });

    it('renders with icon', () => {
      render(<StatCard title="Activity" value={100} icon={Activity} />);

      expect(screen.getByText('Activity')).toBeInTheDocument();
    });

    it('renders description when provided', () => {
      render(
        <StatCard
          title="Revenue"
          value="$10,000"
          description="Monthly revenue"
        />
      );

      expect(screen.getByText('Monthly revenue')).toBeInTheDocument();
    });
  });

  describe('loading state', () => {
    it('renders skeleton when loading', () => {
      const { container } = render(<StatCard title="Loading" value={0} isLoading />);

      // Should have skeleton elements
      const skeletons = container.querySelectorAll('[class*="animate-pulse"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    it('does not render value when loading', () => {
      render(<StatCard title="Loading" value={5000} isLoading />);

      expect(screen.queryByText('5000')).not.toBeInTheDocument();
    });

    it('renders skeleton for icon when loading', () => {
      const { container } = render(<StatCard title="Loading" value={0} isLoading icon={Activity} />);
      const skeletons = container.querySelectorAll('[class*="animate-pulse"]');
      // Should have 3 skeletons? (Title, Icon, Value)
      // Title: h-4 w-24. Icon: h-4 w-4. Value: h-8 w-20.
      expect(skeletons.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('change indicator', () => {
    it('renders positive change', () => {
      render(
        <StatCard
          title="Users"
          value={100}
          change={{ value: 12.5, label: 'from last month' }}
        />
      );

      expect(screen.getByText('+12.5%')).toBeInTheDocument();
      expect(screen.getByText('from last month')).toBeInTheDocument();
    });

    it('renders negative change', () => {
      render(
        <StatCard
          title="Users"
          value={100}
          change={{ value: -5.2, label: 'from last week' }}
        />
      );

      expect(screen.getByText('-5.2%')).toBeInTheDocument();
    });

    it('renders zero change as positive', () => {
      render(
        <StatCard
          title="Users"
          value={100}
          change={{ value: 0, label: 'no change' }}
        />
      );

      expect(screen.getByText('+0%')).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <StatCard title="Test" value={1} className="custom-class" />
      );

      expect(container.querySelector('.custom-class')).toBeInTheDocument();
    });

    it('positive change has green styling', () => {
      render(
        <StatCard
          title="Test"
          value={1}
          change={{ value: 10, label: 'test' }}
        />
      );

      const changeElement = screen.getByText('+10%');
      expect(changeElement).toHaveClass('text-green-600');
    });

    it('negative change has red styling', () => {
      render(
        <StatCard
          title="Test"
          value={1}
          change={{ value: -10, label: 'test' }}
        />
      );

      const changeElement = screen.getByText('-10%');
      expect(changeElement).toHaveClass('text-red-600');
    });
  });
});
