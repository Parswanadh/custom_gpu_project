'use client';

import { motion } from 'framer-motion';
import CountUp from '../shared/CountUp';

const stats = [
  { value: 51, label: "Hardware Modules", suffix: "" },
  { value: 112, label: "Imprint Latency (Cycles)", suffix: "" },
  { value: 341, label: "Dynamic Latency (Cycles)", suffix: "" },
  { value: 2.67, label: "Tok/s Effective Throughput", suffix: "M" }
];

export default function HeroStats() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, delay: 1.2 }}
      className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-12 mb-8 relative z-20"
    >
      {stats.map((stat, i) => (
        <div 
          key={i} 
          className="glass-surface glow-on-hover rounded-xl p-6 border-white/5 flex flex-col items-center justify-center text-center backdrop-blur-xl"
        >
          <div className="font-mono text-3xl font-bold bg-gradient-to-r from-neon-cyan to-trace-gold bg-clip-text text-transparent">
            <CountUp end={stat.value} duration={2.5} suffix={stat.suffix} />
          </div>
          <div className="font-dm text-sm text-metal-silver mt-2 tracking-wide uppercase">
            {stat.label}
          </div>
        </div>
      ))}
    </motion.div>
  );
}
