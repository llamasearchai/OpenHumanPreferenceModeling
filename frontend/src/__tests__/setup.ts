/* eslint-disable no-undef, @typescript-eslint/no-explicit-any */

import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, vi } from 'vitest';

let consoleWarnSpy: ReturnType<typeof vi.spyOn> | undefined;
let consoleErrorSpy: ReturnType<typeof vi.spyOn> | undefined;

beforeAll(() => {
  const originalWarn = console.warn.bind(console);
  const originalError = console.error.bind(console);

  consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation((...args) => {
    const msg = String(args[0] ?? '');
    if (msg.includes('React Router Future Flag Warning')) {
      return;
    }
    originalWarn(...args);
  }) as any;

  consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation((...args) => {
    const msg = String(args[0] ?? '');
    if (msg.includes('Not implemented: navigation (except hash changes)')) {
      return;
    }
    originalError(...args);
  }) as any;
});

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
    get length() {
      return Object.keys(store).length;
    },
    key: vi.fn((index: number) => Object.keys(store)[index] ?? null),
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
  writable: true,
});

// Mock sessionStorage
Object.defineProperty(window, 'sessionStorage', {
  value: localStorageMock,
  writable: true,
});

// Mock ResizeObserver for Radix UI components
class ResizeObserverMock {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}

global.ResizeObserver = ResizeObserverMock as unknown as typeof ResizeObserver;

// Mock PointerEvent for Radix UI components
class PointerEventMock extends MouseEvent {
  constructor(type: string, params: PointerEventInit = {}) {
    super(type, params);
    Object.assign(this, {
      pointerId: params.pointerId ?? 0,
      width: params.width ?? 1,
      height: params.height ?? 1,
      pressure: params.pressure ?? 0,
      tangentialPressure: params.tangentialPressure ?? 0,
      tiltX: params.tiltX ?? 0,
      tiltY: params.tiltY ?? 0,
      twist: params.twist ?? 0,
      pointerType: params.pointerType ?? 'mouse',
      isPrimary: params.isPrimary ?? true,
    });
  }
}

if (typeof window !== 'undefined') {
  window.PointerEvent = PointerEventMock as unknown as typeof PointerEvent;
  // JSDOM doesn't implement scrollTo; some components use it for UX on navigation.
  window.scrollTo = vi.fn() as unknown as typeof window.scrollTo;
  window.HTMLElement.prototype.scrollIntoView = vi.fn();
  window.HTMLElement.prototype.releasePointerCapture = vi.fn();
  window.HTMLElement.prototype.setPointerCapture = vi.fn();
  window.HTMLElement.prototype.hasPointerCapture = vi.fn().mockReturnValue(false);
}

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

beforeEach(() => {
  // Clear localStorage mock before each test
  localStorageMock.clear();
  vi.clearAllMocks();
});

afterEach(() => {
  cleanup();
});

afterAll(() => {
  consoleWarnSpy?.mockRestore();
  consoleErrorSpy?.mockRestore();
});
