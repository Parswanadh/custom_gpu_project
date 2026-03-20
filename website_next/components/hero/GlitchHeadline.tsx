'use client';

import { useEffect, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';

const chars = '!<>-_\\/[]{}—=+*^?#_';

export default function GlitchHeadline({ text }: { text: string }) {
  const [displayText, setDisplayText] = useState('');
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    if (reducedMotion) {
      setDisplayText(text);
      return;
    }

    let iteration = 0;
    const interval = setInterval(() => {
      setDisplayText(
        text
          .split('')
          .map((letter, index) => {
            if (index < iteration) {
              return text[index];
            }
            return chars[Math.floor(Math.random() * chars.length)];
          })
          .join('')
      );
      
      // Step controls speed of reveal
      if (iteration >= text.length) {
        clearInterval(interval);
        setDisplayText(text);
      } else {
        iteration += 1 / 3; // 3 frames of scramble per character
      }
    }, 40);

    return () => clearInterval(interval);
  }, [text, reducedMotion]);

  return (
    <motion.h1 
      className="font-space font-bold text-5xl md:text-7xl lg:text-8xl tracking-tight text-white mb-6 relative z-10"
      initial={{ opacity: 0, y: 20 }}
      animate={{ 
        opacity: 1, 
        y: 0,
        textShadow: [
          "0 0 10px rgba(0,245,255,0.4)",
          "0 0 20px rgba(0,245,255,0.6)",
          "0 0 10px rgba(0,245,255,0.4)",
          "0 0 30px rgba(0,245,255,0.5)",
          "0 0 10px rgba(0,245,255,0.4)"
        ]
      }}
      transition={{ 
        opacity: { duration: 0.8 },
        y: { duration: 0.8 },
        textShadow: { duration: 2, repeat: Infinity, ease: "easeInOut" }
      }}
    >
      {displayText}
    </motion.h1>
  );
}
