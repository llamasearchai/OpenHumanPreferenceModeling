/**
 * NotFound Page Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import NotFoundPage from './NotFound';

const renderWithRouter = (component: React.ReactNode) => {
  return render(<BrowserRouter>{component}</BrowserRouter>);
};

describe('NotFoundPage', () => {
  describe('rendering', () => {
    it('renders 404 heading', () => {
      renderWithRouter(<NotFoundPage />);
      expect(screen.getByText('404')).toBeInTheDocument();
    });

    it('renders Page Not Found text', () => {
      renderWithRouter(<NotFoundPage />);
      expect(screen.getByText('Page Not Found')).toBeInTheDocument();
    });

    it('renders description text', () => {
      renderWithRouter(<NotFoundPage />);
      expect(
        screen.getByText(/The page you're looking for doesn't exist/)
      ).toBeInTheDocument();
    });

    it('renders Go Home button', () => {
      renderWithRouter(<NotFoundPage />);
      expect(screen.getByRole('link', { name: /Go Home/i })).toBeInTheDocument();
    });
  });

  describe('navigation', () => {
    it('Go Home button links to home page', () => {
      renderWithRouter(<NotFoundPage />);
      const link = screen.getByRole('link', { name: /Go Home/i });
      expect(link).toHaveAttribute('href', '/');
    });
  });

  describe('accessibility', () => {
    it('has proper heading hierarchy', () => {
      renderWithRouter(<NotFoundPage />);
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('404');
      expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent('Page Not Found');
    });
  });
});
