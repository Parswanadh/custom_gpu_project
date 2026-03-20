import '@testing-library/jest-dom';
import { vi } from 'vitest';

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
window.ResizeObserver = ResizeObserver;

class IntersectionObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
window.IntersectionObserver = IntersectionObserver as any;

// Mock Three.js Canvas to prevent WebGL context errors in JSDOM
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: any) => children,
  useFrame: vi.fn(),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => null,
  Environment: () => null,
  Float: ({ children }: any) => children,
  Box: ({ children }: any) => children,
  Plane: ({ children }: any) => children,
  Line: () => null,
  Sphere: ({ children }: any) => children,
  Text: ({ children }: any) => children,
}));

vi.mock('@react-three/postprocessing', () => ({
  EffectComposer: ({ children }: any) => children,
  Bloom: () => null,
  DepthOfField: () => null,
}));
