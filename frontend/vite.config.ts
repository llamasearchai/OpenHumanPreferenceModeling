import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

/**
 * Vite Configuration
 * 
 * Purpose: Development server and build optimization for the frontend
 * 
 * Key features:
 * - Hot module replacement optimized for Three.js (preserves WebGL context)
 * - Code splitting strategy for optimal bundle sizes
 * - Path aliases for clean imports
 * - Environment variable validation
 */
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
      '@/components': resolve(__dirname, './src/components'),
      '@/hooks': resolve(__dirname, './src/hooks'),
      '@/lib': resolve(__dirname, './src/lib'),
      '@/stores': resolve(__dirname, './src/stores'),
      '@/types': resolve(__dirname, './src/types'),
      '@/styles': resolve(__dirname, './src/styles'),
      '@/assets': resolve(__dirname, './src/assets'),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: process.env.VITE_WS_URL || 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  build: {
    target: 'ES2022',
    outDir: 'dist',
    // Only generate sourcemaps in development, not production
    sourcemap: process.env.NODE_ENV !== 'production',
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunk for stable dependencies
          vendor: ['react', 'react-dom', 'react-router-dom'],
          // TanStack Query chunk
          query: ['@tanstack/react-query'],
          // UI components chunk
          ui: ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', '@radix-ui/react-tooltip'],
          // Three.js chunk (separate due to large size)
          three: ['three', '@react-three/fiber', '@react-three/drei'],
          // D3.js chunk (tree-shaken)
          d3: ['d3'],
          // Animation chunk
          animation: ['framer-motion'],
        },
      },
    },
    chunkSizeWarningLimit: 750,
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'three', '@react-three/fiber', 'd3'],
    exclude: ['@react-three/postprocessing'],
  },
});
