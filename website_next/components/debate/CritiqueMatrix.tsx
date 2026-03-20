'use client';

import { motion } from 'framer-motion';

export default function CritiqueMatrix() {
  const experts = ['Architecture', 'Performance', 'Security'];

  return (
    <div className="relative bg-[#050812] border border-white/10 rounded-2xl p-6 overflow-hidden">
      <div className="flex justify-between items-center mb-6">
        <h4 className="font-space font-bold text-white tracking-widest text-sm uppercase">
          N² Cross-Critique Matrix
        </h4>
        <div className="text-xs font-mono text-plasma-violet border border-plasma-violet/30 bg-plasma-violet/10 px-2 py-1 rounded">
          LIVE DEMO
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 mb-6 text-xs font-dm font-semibold text-metal-silver text-center">
        <div></div>
        <div>Arch.</div>
        <div>Perf.</div>
        <div>Sec.</div>
      </div>

      <div className="space-y-2">
        {experts.map((reviewer, i) => (
          <div key={reviewer} className="grid grid-cols-4 gap-2 items-center">
            <div className="text-xs font-dm font-semibold text-metal-silver text-right pr-2">
              {reviewer.substring(0, 4)}.
            </div>
            {experts.map((reviewed, j) => {
              const isSelf = i === j;
              return (
                <div 
                  key={`${i}-${j}`} 
                  className={`h-10 rounded border flex items-center justify-center relative ${
                    isSelf 
                      ? 'bg-white/5 border-transparent' 
                      : 'bg-silicon-gray/50 border-white/5'
                  }`}
                >
                  {!isSelf && (
                    <motion.div
                      className="absolute w-2 h-2 rounded-full"
                      initial={{ backgroundColor: '#FF3366', scale: 0 }}
                      animate={{ 
                        backgroundColor: ['#FF3366', '#00FF88'],
                        scale: [0, 1, 0.5, 1],
                        filter: ['blur(4px)', 'blur(0px)']
                      }}
                      transition={{ 
                        duration: 3, 
                        repeat: Infinity, 
                        repeatType: 'reverse',
                        delay: (i * 3 + j) * 0.2
                      }}
                    />
                  )}
                  {isSelf && <span className="text-white/20">-</span>}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Consensus Animation */}
      <div className="mt-8">
        <div className="flex justify-between text-xs font-mono text-metal-silver mb-2">
          <span>Consensus Target: 0.85</span>
          <motion.span
            animate={{ color: ['#FF3366', '#00FF88'] }}
            transition={{ duration: 3, repeat: Infinity, repeatType: 'reverse' }}
          >
            0.58 → 0.83
          </motion.span>
        </div>
        <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden relative">
          <motion.div 
            className="absolute top-0 left-0 h-full rounded-full"
            initial={{ width: '58%', backgroundColor: '#FF3366' }}
            animate={{ width: '83%', backgroundColor: '#00FF88' }}
            transition={{ duration: 3, repeat: Infinity, repeatType: 'reverse', ease: "easeInOut" }}
          />
        </div>
      </div>
    </div>
  );
}
