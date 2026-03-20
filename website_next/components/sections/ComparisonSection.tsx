'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import { Info, Check, X } from 'lucide-react';
import WorkflowBackground from '../shared/WorkflowBackground';

const rows = [
  { 
    metric: 'Model Parameter Loading', 
    human: 'Cloud API', 
    copilot: 'AXI4-Lite DDR Stalls', 
    bitbybit: 'Silicon-Imprinted ROM',
    tooltip: 'Critical weights (Gemma 3) burned into logic elements via .hex bitstreams.'
  },
  { 
    metric: 'Compute Math Engine', 
    human: 'FP32 Clusters', 
    copilot: '19-cycle FP16 MACs', 
    bitbybit: '4-cycle SIMD Ternary',
    tooltip: 'Zero multi-cycle multipliers required. 4.8x faster using -1, 0, 1 logic.'
  },
  { 
    metric: 'Softmax Latency', 
    human: 'TFLOPS GPU', 
    copilot: '25-cycle Naive Loop', 
    bitbybit: '4-cycle Arrayed HW',
    tooltip: 'Utilizing a parallel max-subtraction tree and simultaneous 256-entry Exp LUT lookups.'
  },
  { 
    metric: 'Pipeline Throughput', 
    human: 'Batch Processing', 
    copilot: '18-cycle FSM Stalls', 
    bitbybit: '6-cycle/token Flow',
    tooltip: 'Continuous 6-stage pipeline overlapping fetching and logic for 341-cycle inference.'
  }
];

export default function ComparisonSection() {
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);

  return (
    <section id="comparison" className="relative py-32 w-full bg-[#050B14] min-h-screen border-t border-white/5 flex flex-col justify-center">
      <WorkflowBackground color="cyan" density="sparse" />
      
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full">
        <motion.div 
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-block px-4 py-1.5 rounded-full border border-neon-cyan/30 bg-neon-cyan/10 text-neon-cyan font-dm text-sm font-bold tracking-widest uppercase mb-6 glow-on-hover">
            Unprecedented Improvement Cycle
          </div>
          <h2 className="font-space font-bold text-4xl md:text-5xl text-white mb-6">
            Before vs <span className="text-neon-cyan">After.</span>
          </h2>
          <p className="font-inter text-lg text-metal-silver max-w-2xl mx-auto">
            A precise architectural comparison highlighting the delta between the Original Implementation and the BitNet/Imprint Unification.
          </p>
        </motion.div>

        {/* Desktop Table View */}
        <div className="glass-surface rounded-3xl border border-white/10 overflow-hidden shadow-2xl backdrop-blur-xl">
          <div className="grid grid-cols-4 gap-4 p-6 border-b border-white/5 bg-silicon-black/50">
            <div className="font-dm text-sm font-bold text-metal-silver uppercase tracking-widest">Subsystem</div>
            <div className="font-dm text-sm font-bold text-metal-silver uppercase tracking-widest text-center">Legacy IP</div>
            <div className="font-dm text-sm font-bold text-metal-silver uppercase tracking-widest text-center">Original RTL</div>
            <div className="font-space text-lg font-bold text-neon-cyan uppercase tracking-widest text-center flex items-center justify-center gap-2">
              <div className="w-2 h-2 rounded-full bg-neon-cyan animate-pulse" />
              BitbyBit
            </div>
          </div>

          <div className="relative">
            {/* Highlight Column Backdrop for BitbyBit */}
            <div className="absolute top-0 bottom-0 right-0 w-1/4 bg-neon-cyan/5 border-l border-r border-neon-cyan/20 pointer-events-none" />

            {rows.map((row, i) => (
              <motion.div 
                key={i}
                className="grid grid-cols-4 gap-4 p-6 border-b border-white/5 items-center relative hover:bg-white/5 transition-colors group cursor-pointer"
                initial={{ opacity: 0, x: -50 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                onMouseEnter={() => setHoveredRow(i)}
                onMouseLeave={() => setHoveredRow(null)}
              >
                {/* Metric Name with Tooltip trigger */}
                <div className="font-inter text-white font-medium flex items-center gap-2 relative">
                  {row.metric}
                  <Info className="w-4 h-4 text-metal-silver group-hover:text-neon-cyan transition-colors" />
                  
                  {/* Hover Tooltip */}
                  <AnimatePresence>
                    {hoveredRow === i && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.9, x: 10 }}
                        animate={{ opacity: 1, scale: 1, x: 0 }}
                        exit={{ opacity: 0, scale: 0.9, x: 10 }}
                        className="absolute left-full ml-4 top-1/2 -translate-y-1/2 w-64 p-3 bg-silicon-black border border-white/10 rounded-lg shadow-xl z-50 pointer-events-none"
                      >
                        <p className="font-mono text-xs text-metal-silver leading-relaxed">
                          {row.tooltip}
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                <div className="font-mono text-sm text-metal-silver text-center flex items-center justify-center gap-2">
                  <X className="w-4 h-4 text-error-red/50" /> {row.human}
                </div>
                <div className="font-mono text-sm text-metal-silver text-center flex items-center justify-center gap-2">
                  <X className="w-4 h-4 text-error-red/50" /> {row.copilot}
                </div>
                <div className="font-space font-bold text-white text-center flex items-center justify-center gap-2 relative z-10">
                  <Check className="w-5 h-5 text-neon-cyan drop-shadow-[0_0_8px_#00F5FF]" /> {row.bitbybit}
                </div>
              </motion.div>
            ))}
          </div>
        </div>

      </div>
    </section>
  );
}
