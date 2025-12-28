/**
 * Alert Component Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Alert, AlertTitle, AlertDescription } from './alert';

describe('Alert', () => {
  describe('Rendering', () => {
    it('renders with default variant', () => {
      render(<Alert>Alert message</Alert>);
      expect(screen.getByRole('alert')).toBeInTheDocument();
      expect(screen.getByText('Alert message')).toBeInTheDocument();
    });

    it('renders with info variant', () => {
      render(<Alert variant="info">Info alert</Alert>);
      expect(screen.getByRole('alert')).toHaveClass('bg-blue-50');
    });

    it('renders with success variant', () => {
      render(<Alert variant="success">Success alert</Alert>);
      expect(screen.getByRole('alert')).toHaveClass('bg-green-50');
    });

    it('renders with warning variant', () => {
      render(<Alert variant="warning">Warning alert</Alert>);
      expect(screen.getByRole('alert')).toHaveClass('bg-yellow-50');
    });

    it('renders with destructive variant', () => {
      render(<Alert variant="destructive">Error alert</Alert>);
      expect(screen.getByRole('alert')).toHaveClass('text-destructive');
    });
  });

  describe('Accessibility', () => {
    it('has alert role', () => {
      render(<Alert>Alert</Alert>);
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });
  });
});

describe('AlertTitle', () => {
  it('renders title text', () => {
    render(<AlertTitle>Alert Title</AlertTitle>);
    expect(screen.getByText('Alert Title')).toBeInTheDocument();
  });

  it('has proper heading style', () => {
    render(<AlertTitle>Title</AlertTitle>);
    expect(screen.getByText('Title')).toHaveClass('font-medium');
  });
});

describe('AlertDescription', () => {
  it('renders description text', () => {
    render(<AlertDescription>Alert description</AlertDescription>);
    expect(screen.getByText('Alert description')).toBeInTheDocument();
  });
});

describe('Alert Composition', () => {
  it('renders complete alert structure', () => {
    render(
      <Alert variant="info">
        <AlertTitle>Heads up!</AlertTitle>
        <AlertDescription>
          You can add components to your app using the cli.
        </AlertDescription>
      </Alert>
    );

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('Heads up!')).toBeInTheDocument();
    expect(screen.getByText(/You can add components/)).toBeInTheDocument();
  });
});
