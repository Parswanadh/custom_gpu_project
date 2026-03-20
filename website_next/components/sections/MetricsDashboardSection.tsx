'use client';

import { motion } from 'framer-motion';
import WorkflowBackground from '../shared/WorkflowBackground';
import RunsChart from '../metrics/RunsChart';
import CountUp from '../shared/CountUp';
import { Activity, Clock, Database, TrendingDown } from 'lucide-react';

const stats = [
  { label: 'Layer Latency', value: 96, prefix: '', suffix: ' cy', icon: <Clock className="w-5 h-5 text-neon-cyan" />, color: 'border-neon-cyan/20' },
  { label: 'Embedding Extract', value: 10, suffix: ' ns', icon: <Database className="w-5 h-5 text-die-copper" />, color: 'border-die-copper/20' },
  { label: 'Effective Clock', value: 100, suffix: ' MHz', icon: <Activity className="w-5 h-5 text-oxide-green" />, color: 'border-oxide-green/20' },
  { label: 'Latency (µs)', value: 1.12, suffix: ' µs', icon: <TrendingDown className="w-5 h-5 text-error-red" />, color: 'border-error-red/20' }
];

export default function MetricsDashboardSection() {
  return (
    <section id="metrics" className="relative py-32 w-full bg-silicon-black min-h-screen border-t border-white/5 overflow-hidden flex items-center">
      <WorkflowBackground color="copper" density="medium" />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full">
        <motion.div 
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-block px-4 py-1.5 rounded-full border border-die-copper/30 bg-die-copper/10 text-die-copper font-dm text-sm font-bold tracking-widest uppercase mb-6 glow-on-hover">
            Cycle-Accurate Benchmarks
          </div>
          <h2 className="font-space font-bold text-4xl md:text-5xl text-white mb-6">
            Raw Simulation <span className="text-die-copper">Results</span>
          </h2>
          <p className="font-inter text-lg text-metal-silver max-w-3xl mx-auto leading-relaxed">
            These are NOT estimated metrics—they are extracted directly from Icarus Verilog `vvp` simulation logs targeting a 100MHz FPGA clock.
          </p>
        </motion.div>

        {/* 4 Metric Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {stats.map((stat, i) => (
            <motion.div
              key={i}
              className={`glass-surface p-6 rounded-2xl border ${stat.color} hover:bg-white/5 transition-colors`}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.1 }}
            >
              <div className="flex justify-between items-start mb-8">
                <div className="p-3 bg-silicon-black/50 rounded-lg border border-white/5">
                  {stat.icon}
                </div>
              </div>
              <div>
                <div className="font-space font-bold text-3xl text-white mb-1 tracking-tight">
                  <CountUp end={stat.value} duration={2} prefix={stat.prefix} suffix={stat.suffix} />
                </div>
                <div className="font-mono text-xs text-metal-silver uppercase tracking-wider">
                  {stat.label}
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Main Chart */}
        <motion.div
          className="glass-surface p-6 md:p-8 rounded-3xl border border-white/10 w-full"
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <div className="flex flex-col md:flex-row justify-between items-center mb-8">
            <div>
              <h3 className="font-space font-bold text-xl text-white mb-2">Icarus Verilog (vvp) Simulation Log</h3>
              <p className="font-mono text-xs text-metal-silver">112-CYCLE IMPRINT INFERENCE TRACE</p>
            </div>
            <div className="flex items-center gap-6 mt-4 md:mt-0 font-dm text-sm">
              <div className="flex items-center gap-2 text-error-red font-medium">
                <div className="w-3 h-3 rounded-full bg-error-red" />
                Imprint Path
              </div>
              <div className="flex items-center gap-2 text-oxide-green font-medium">
                <div className="w-3 h-3 rounded-full bg-oxide-green" />
                Dynamic Path
              </div>
            </div>
          </div>
          
          <RunsChart />
        </motion.div>

      </div>
    </section>
  );
}
