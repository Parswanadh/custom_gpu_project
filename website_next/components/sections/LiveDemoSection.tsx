'use client';

import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Play, Square } from 'lucide-react';
import WorkflowBackground from '../shared/WorkflowBackground';

const demoLines: { text: string; delay: number; success?: boolean; error?: boolean }[] = [
  { text: "$ bitbybit-sim --input 'hello' --mode accelerated", delay: 500 },
  { text: "> Initializing BitbyBit GPU (Verilog-2005 RTL)...", delay: 800 },
  { text: "> Loading Imprinted Weights (Gemma 3 270M)...", delay: 1500 },
  { text: "[System] Tokens: [8, 5, 12, 12, 15] (h-e-l-l-o)", delay: 2500 },
  { text: "[Pipeline] Token 0: 130 cycles. Zero-skip active (28.7%)", delay: 3500 },
  { text: "[Pipeline] Token 1: 130 cycles. MSE vs Float32: 0.000450", delay: 4500 },
  { text: "[Pipeline] Token 2: 130 cycles. KV-Cache Read hit.", delay: 5500 },
  { text: "[Pipeline] Token 3: 130 cycles. Parallel Softmax complete.", delay: 6500 },
  { text: "[Pipeline] Token 4: 130 cycles. GELU Piecewise Approx verified.", delay: 7500 },
  { text: "> Summary: 650 cycles total. Latency: 6.5 us @ 100MHz", delay: 9000, success: true },
  { text: "> Speedup vs Cortex-M4: 38.5x (Hardware Accelerated)", delay: 10000, success: true },
  { text: "> Simulation DONE. 5/5 tokens match Q8.8 Reference.", delay: 11000, success: true }
];

export default function LiveDemoSection() {
  const [isRunning, setIsRunning] = useState(false);
  const [visibleLines, setVisibleLines] = useState<number>(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll terminal
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [visibleLines]);

  const runDemo = () => {
    if (isRunning) return;
    setIsRunning(true);
    setVisibleLines(0);

    const timeouts: number[] = [];
    demoLines.forEach((line, index) => {
      const id = window.setTimeout(() => {
        setVisibleLines(index + 1);
        if (index === demoLines.length - 1) {
          setIsRunning(false);
        }
      }, line.delay);
      timeouts.push(id);
    });
    
    // Store timeouts in a ref if we want to cancel them properly
    (window as unknown as { _demoTimeouts: number[] })._demoTimeouts = timeouts;
  };

  const stopDemo = () => {
    const timeouts = (window as unknown as { _demoTimeouts: number[] })._demoTimeouts || [];
    timeouts.forEach((id: number) => window.clearTimeout(id));
    setIsRunning(false);
  };

  return (
    <section id="demo" className="relative py-32 w-full bg-silicon-black min-h-screen border-t border-white/5 flex items-center overflow-hidden">
      <WorkflowBackground color="cyan" density="medium" />
      
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full text-center">
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-block px-4 py-1.5 rounded-full border border-neon-cyan/30 bg-neon-cyan/10 text-neon-cyan font-dm text-sm font-bold tracking-widest uppercase mb-6 glow-on-hover">
            Cycle-Accurate Sim
          </div>
          <h2 className="font-space font-bold text-4xl md:text-5xl text-white mb-6">
            Witness the <span className="text-neon-cyan drop-shadow-[0_0_20px_rgba(0,245,255,0.4)]">Hardware</span> Flow
          </h2>
          <p className="font-inter text-lg text-metal-silver max-w-2xl mx-auto mb-12">
            Real-time Icarus Verilog output from the BitbyBit GPU core processing a 5-token sequence. Zero-multipliers, SIMD ternary, and silicon-imprinted weights in action.
          </p>
        </motion.div>

        {/* Terminal Window */}
        <motion.div 
          className="relative text-left bg-[#020408] border border-white/10 rounded-2xl overflow-hidden shadow-2xl"
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          {/* Mac-style Window Controls & Action Buttons */}
          <div className="flex items-center justify-between px-4 py-3 bg-silicon-gray/50 border-b border-white/5">
            <div className="flex gap-2">
              <div className="w-3 h-3 rounded-full bg-error-red/80" />
              <div className="w-3 h-3 rounded-full bg-trace-gold/80" />
              <div className="w-3 h-3 rounded-full bg-oxide-green/80" />
            </div>
            <div className="flex gap-4">
              {isRunning ? (
                <button 
                  onClick={stopDemo}
                  className="flex items-center gap-2 px-3 py-1 rounded-md bg-error-red/10 border border-error-red/30 text-error-red font-mono text-xs hover:bg-error-red hover:text-white transition-colors"
                >
                  <Square className="w-3 h-3" /> STOP SIM
                </button>
              ) : (
                <button 
                  onClick={runDemo}
                  className="flex items-center gap-2 px-3 py-1 rounded-md bg-neon-cyan/10 border border-neon-cyan/30 text-neon-cyan font-mono text-xs hover:bg-neon-cyan hover:text-silicon-black transition-colors"
                >
                  <Play className="w-3 h-3" /> START SIMULATION
                </button>
              )}
            </div>
          </div>

          {/* Terminal Output Area */}
          <div 
            ref={scrollRef}
            className="p-6 h-[400px] overflow-y-auto font-mono text-sm space-y-3 bg-black/40 backdrop-blur-sm"
          >
            {visibleLines === 0 && !isRunning && (
              <div className="text-metal-silver/50 italic">
                Ready for cycle-accurate Verilog execution...
              </div>
            )}
            
            {demoLines.slice(0, visibleLines).map((line, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className={`
                  ${line.error ? 'text-error-red font-bold' : 
                    line.success ? 'text-oxide-green glow-on-hover' : 
                    line.text.startsWith('$') ? 'text-neon-cyan' : 
                    line.text.startsWith('[Pipeline]') ? 'text-trace-gold' :
                    'text-metal-silver'}
                `}
              >
                {line.text}
              </motion.div>
            ))}
            
            {isRunning && (
              <div className="typing-cursor inline-block w-2 h-4 bg-neon-cyan animate-pulse mt-2" />
            )}
          </div>
        </motion.div>

      </div>
    </section>
  );
}
