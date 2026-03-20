'use client';

import { motion } from 'framer-motion';

const bugs = [
  { run: 1, type: 'Syntax', desc: 'Missing async keyword in resolver', typeColor: 'text-error-red', bg: 'bg-error-red/10', border: 'border-error-red/30' },
  { run: 4, type: 'Type', desc: 'Strict null check failure on AST payload', typeColor: 'text-die-copper', bg: 'bg-die-copper/10', border: 'border-die-copper/30' },
  { run: 12, type: 'Logic', desc: 'Infinite loop during N² critique consensus', typeColor: 'text-neon-cyan', bg: 'bg-neon-cyan/10', border: 'border-neon-cyan/30' },
  { run: 21, type: 'Security', desc: 'Regex DOS vulnerability in log parser', typeColor: 'text-plasma-violet', bg: 'bg-plasma-violet/10', border: 'border-plasma-violet/30' },
  { run: 27, type: 'Quality', desc: 'Zero defects. Final deployment.', typeColor: 'text-oxide-green', bg: 'bg-oxide-green/10', border: 'border-oxide-green/30' },
];

export default function ErrorTimeline() {
  return (
    <div className="relative pl-8 md:pl-0">
      {/* 
        Timeline Vertical Line 
        md: Left aligned on mobile, center on desktop
      */}
      <div className="absolute left-0 md:left-1/2 top-0 bottom-0 w-1 bg-gradient-to-b from-error-red via-neon-cyan to-oxide-green transform md:-translate-x-1/2 opacity-20 rounded-full" />

      <div className="space-y-12">
        {bugs.map((bug, i) => (
          <motion.div 
            key={i} 
            className={`relative flex items-center ${i % 2 === 0 ? 'md:flex-row-reverse' : ''}`}
            initial={{ opacity: 0, x: i % 2 === 0 ? 50 : -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, type: 'spring' }}
          >
            {/* Timeline Dot */}
            <div className="absolute left-0 md:left-1/2 w-4 h-4 rounded-full bg-silicon-black border-2 border-white transform -translate-x-1.5 md:-translate-x-1/2 z-10 shadow-[0_0_15px_rgba(255,255,255,0.5)]" />

            <div className={`w-full md:w-5/12 ml-6 md:ml-0 ${i % 2 === 0 ? 'md:pl-10' : 'md:pr-10 text-left md:text-right'}`}>
              <div className={`glass-surface p-5 rounded-2xl border ${bug.border} glow-on-hover`}>
                <div className={`flex items-center gap-3 mb-2 ${i % 2 === 0 ? '' : 'md:justify-end'}`}>
                  <span className="font-space font-bold text-white text-lg">Run #{bug.run}</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${bug.bg} ${bug.typeColor}`}>
                    {bug.type.toUpperCase()}
                  </span>
                </div>
                <p className="font-inter text-sm text-metal-silver leading-relaxed">
                  {bug.desc}
                </p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
