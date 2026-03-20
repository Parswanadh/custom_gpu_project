'use client';

import { useReducedMotion, motion } from 'framer-motion';
import { useMemo } from 'react';

type ColorVariant = 'copper' | 'cyan' | 'violet' | 'green' | 'red';
type DensityVariant = 'sparse' | 'medium' | 'dense';

interface WorkflowBackgroundProps {
  color: ColorVariant;
  density: DensityVariant;
}

const colorMap = {
  copper: '#B87333',
  cyan: '#00F5FF',
  violet: '#7C3AED',
  green: '#00FF88',
  red: '#FF3366',
};

export default function WorkflowBackground({ color, density }: WorkflowBackgroundProps) {
  const reducedMotion = useReducedMotion();
  const hexColor = colorMap[color];

  const lineCount = { sparse: 5, medium: 12, dense: 25 }[density];

  // Generate deterministic-looking random SVG trace patterns
  const paths = useMemo(() => {
    const arr = [];
    for (let i = 0; i < lineCount; i++) {
      const y = Math.random() * 100;
      const x1 = Math.random() * 20;
      const x2 = x1 + Math.random() * 30 + 10;
      const x3 = x2 + Math.random() * 10;
      const y2 = y + (Math.random() > 0.5 ? 20 : -20);
      
      const d = `M ${x1} ${y} L ${x2} ${y} L ${x3} ${y2} L 100 ${y2}`;
      arr.push(d);
    }
    return arr;
  }, [lineCount]);

  return (
    <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none opacity-30">
      <svg width="100%" height="100%" viewBox="0 0 100 100" className="w-full h-full" preserveAspectRatio="none">
        {paths.map((d, i) => (
          <motion.path
            key={i}
            d={d}
            fill="none"
            stroke={hexColor}
            style={{ willChange: "opacity" }}
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={{ opacity: 0.1 }}
            animate={
              reducedMotion
                ? { opacity: 0.2 }
                : { opacity: [0.1, 0.4, 0.1] }
            }
            transition={
              reducedMotion
                ? {}
                : {
                    duration: 3 + Math.random() * 4,
                    repeat: Infinity,
                    repeatType: 'mirror',
                    ease: 'easeInOut',
                    delay: Math.random() * 2,
                  }
            }
          />
        ))}
      </svg>
      {/* Dark gradient fade for edges */}
      <div className="absolute inset-0 bg-gradient-to-b from-silicon-black via-transparent to-silicon-black" />
    </div>
  );
}
