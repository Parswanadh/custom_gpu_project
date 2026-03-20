'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Check } from 'lucide-react';

const lines = [
  { text: "Detected Exception: TypeError: Cannot read prop 'id' of null", type: "error" },
  { text: "Analyzing stack trace...", type: "system" },
  { text: "Retrieving semantic context from AST...", type: "system" },
  { text: "Diagnosis: Missing truthiness validation prior to Map access.", type: "system" },
  { text: "Formulating architectural rule...", type: "system" },
  { text: "LESSON_INJECTED: 'Always assert node existence in AST loops'", type: "success" },
  { text: "Updating global prompt context... OK", type: "success" },
  { text: "Restarting code generation pipeline...", type: "info" }
];

export default function LessonTerminal() {
  const [visibleLines, setVisibleLines] = useState<number>(0);

  useEffect(() => {
    let currentLine = 0;
    const interval = setInterval(() => {
      if (currentLine < lines.length) {
        setVisibleLines(prev => prev + 1);
        currentLine++;
      } else {
        clearInterval(interval);
      }
    }, 800);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-[#050812] border border-white/10 rounded-2xl overflow-hidden shadow-2xl font-mono text-xs sm:text-sm">
      {/* Terminal Header */}
      <div className="flex items-center px-4 py-2 bg-white/5 border-b border-white/10">
        <div className="flex gap-2">
          <div className="w-3 h-3 rounded-full bg-error-red/80" />
          <div className="w-3 h-3 rounded-full bg-trace-gold/80" />
          <div className="w-3 h-3 rounded-full bg-oxide-green/80" />
        </div>
        <div className="mx-auto text-metal-silver/70 font-semibold tracking-wide">
          memory_injector.sh
        </div>
      </div>

      {/* Terminal Body */}
      <div className="p-6 h-[300px] overflow-y-auto space-y-3">
        {lines.slice(0, visibleLines).map((line, i) => (
          <motion.div 
            key={i}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            className={`flex items-start gap-3 ${
              line.type === 'error' ? 'text-error-red' :
              line.type === 'success' ? 'text-oxide-green' :
              line.type === 'info' ? 'text-neon-cyan' :
              'text-metal-silver'
            }`}
          >
            <span className="opacity-50 select-none flex-shrink-0">
              {line.type === 'success' ? <Check className="w-4 h-4 mt-0.5" /> : '>'}
            </span>
            <span className={line.type === 'error' ? 'font-bold' : ''}>
              {line.text}
            </span>
          </motion.div>
        ))}
        {visibleLines < lines.length && (
          <div className="typing-cursor" />
        )}
      </div>
    </div>
  );
}
