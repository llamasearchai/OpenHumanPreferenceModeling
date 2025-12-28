/**
 * Export Button Widget
 *
 * Purpose: Reusable export functionality for data tables and charts
 */

import { Button } from '@/components/ui/button';
import { Download } from 'lucide-react';

interface ExportButtonProps {
  data: unknown[];
  filename: string;
  format?: 'csv' | 'json';
  transform?: (data: unknown[]) => string;
  disabled?: boolean;
  variant?: 'default' | 'outline' | 'ghost';
  size?: 'default' | 'sm' | 'lg' | 'icon';
}

export function ExportButton({
  data,
  filename,
  format = 'csv',
  transform,
  disabled,
  variant = 'outline',
  size = 'sm',
}: ExportButtonProps) {
  const handleExport = () => {
    let content: string;
    let mimeType: string;
    let extension: string;

    if (transform) {
      content = transform(data);
      mimeType = 'text/plain';
      extension = 'txt';
    } else if (format === 'csv') {
      if (data.length === 0) {
        return;
      }

      // Convert array of objects to CSV
      const headers = Object.keys(data[0] as Record<string, unknown>);
      const rows = data.map((item) => {
        const obj = item as Record<string, unknown>;
        return headers.map((header) => {
          const value = obj[header];
          if (value === null || value === undefined) return '';
          if (typeof value === 'object') return JSON.stringify(value);
          return String(value);
        });
      });

      content = [
        headers.join(','),
        ...rows.map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(',')),
      ].join('\n');
      mimeType = 'text/csv';
      extension = 'csv';
    } else {
      content = JSON.stringify(data, null, 2);
      mimeType = 'application/json';
      extension = 'json';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.${extension}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Button
      variant={variant}
      size={size}
      onClick={handleExport}
      disabled={disabled || data.length === 0}
    >
      <Download className="mr-2 h-4 w-4" />
      Export {format.toUpperCase()}
    </Button>
  );
}
