import js from '@eslint/js';
import typescript from '@typescript-eslint/eslint-plugin';
import typescriptParser from '@typescript-eslint/parser';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import jsxA11y from 'eslint-plugin-jsx-a11y';

export default [
  js.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      parser: typescriptParser,
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        ecmaFeatures: {
          jsx: true,
        },
      },
      globals: {
        fetch: 'readonly',
        Headers: 'readonly',
        Request: 'readonly',
        Response: 'readonly',
        AbortController: 'readonly',
        AbortSignal: 'readonly',
        localStorage: 'readonly',
        console: 'readonly',
        setTimeout: 'readonly',
        clearTimeout: 'readonly',
        setInterval: 'readonly',
        clearInterval: 'readonly',
        URL: 'readonly',
        URLSearchParams: 'readonly',
        Math: 'readonly',
        JSON: 'readonly',
        Promise: 'readonly',
        document: 'readonly',
        window: 'readonly',
        Blob: 'readonly',
        FormData: 'readonly',
        navigator: 'readonly',
        PointerEvent: 'readonly',
        KeyboardEvent: 'readonly',
        MouseEvent: 'readonly',
        HTMLElement: 'readonly',
        HTMLDivElement: 'readonly',
        HTMLCanvasElement: 'readonly',
        sessionStorage: 'readonly',
        HTMLTableSectionElement: 'readonly',
        HTMLTableRowElement: 'readonly',
        HTMLTableCellElement: 'readonly',
        HTMLTableCaptionElement: 'readonly',
        HTMLTextAreaElement: 'readonly',
        MediaQueryListEvent: 'readonly',
        Navigator: 'readonly',
        requestAnimationFrame: 'readonly',
        performance: 'readonly',
        PerformanceObserver: 'readonly',
        PerformanceEntry: 'readonly',
        Performance: 'readonly',
        ResizeObserver: 'readonly',
        HTMLSpanElement: 'readonly',
        File: 'readonly',
        GeoJSON: 'readonly',
        React: 'readonly',
        HTMLParagraphElement: 'readonly',
        HTMLHeadingElement: 'readonly',
        HTMLButtonElement: 'readonly',
        HTMLInputElement: 'readonly',
        HTMLTableElement: 'readonly',
        alert: 'readonly',
      },
    },
    plugins: {
      '@typescript-eslint': typescript,
      'react': react,
      'react-hooks': reactHooks,
      'jsx-a11y': jsxA11y,
    },
    rules: {
      // TypeScript rules
      ...typescript.configs.recommended.rules,
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/no-explicit-any': 'warn',
      'no-redeclare': 'off',
      '@typescript-eslint/no-redeclare': 'error',

      // React rules
      'react/react-in-jsx-scope': 'off',
      'react/prop-types': 'off',

      // React hooks rules
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',

      // Accessibility rules
      'jsx-a11y/alt-text': 'error',
      'jsx-a11y/aria-props': 'error',
      'jsx-a11y/aria-role': 'error',

      // Security: Prevent localhost/127.0.0.1 fetch calls in production code
      'no-restricted-syntax': [
        'error',
        {
          selector: 'CallExpression[callee.name="fetch"][arguments.0.value=/localhost|127\\.0\\.0\\.1/]',
          message: 'Do not use localhost or 127.0.0.1 in fetch calls - this is likely debug code that should be removed.',
        },
        {
          selector: 'Literal[value=/http:\\/\\/localhost|http:\\/\\/127\\.0\\.0\\.1/]',
          message: 'Do not hardcode localhost or 127.0.0.1 URLs - use environment variables for API configuration.',
        },
        {
          selector: 'TemplateLiteral[quasis.0.value.raw=/localhost|127\\.0\\.0\\.1/]',
          message: 'Do not hardcode localhost or 127.0.0.1 URLs - use environment variables for API configuration.',
        },
      ],
    },
    settings: {
      react: {
        version: 'detect',
      },
    },
  },
  {
    ignores: [
      'dist/**',
      'node_modules/**',
      'coverage/**',
      'playwright-report/**',
      '*.config.js',
      '*.config.ts',
      '**/*.min.js',
      '.vite/**',
      '.cache/**',
    ],
  },
];
