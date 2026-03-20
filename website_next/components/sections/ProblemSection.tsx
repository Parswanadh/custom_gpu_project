'use client';

import { motion } from 'framer-motion';
import WorkflowBackground from '../shared/WorkflowBackground';
import { Clock, DollarSign, Bug } from 'lucide-react';

const problems = [
  {
    icon: <Clock className="w-8 h-8 text-error-red mb-4" />,
    title: "Memory Wall Stalls",
    desc: "Standard AXI4-Lite DDR controllers cause massive compute stalls, wasting 80% of peak TFLOPS waiting for weights."
  },
  {
    icon: <DollarSign className="w-8 h-8 text-error-red mb-4" />,
    title: "MAC Unit Latency",
    desc: "Traditional FP32/FP16 MAC units require 19+ cycles per operation, bottlenecking transformer throughput at the edge."
  },
  {
    icon: <Bug className="w-8 h-8 text-error-red mb-4" />,
    title: "PCIe Transfer Overhead",
    desc: "Continuous weight streaming across the PCIe bus introduces multi-microsecond jitter, breaking real-time execution."
  }
];

export default function ProblemSection() {
  return (
    <section id="problem" className="relative py-32 w-full bg-silicon-black border-t border-white/5 overflow-hidden">
      <WorkflowBackground color="red" density="sparse" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <motion.div 
          className="text-center mb-16"
          initial={{ opacity: 0, scale: 0.95, filter: 'blur(10px)' }}
          whileInView={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <div className="inline-block px-4 py-1.5 rounded-full border border-error-red/30 bg-error-red/10 text-error-red font-dm text-sm font-bold tracking-widest uppercase mb-6 glow-on-hover">
            The Architecture Bottleneck
          </div>
          <h2 className="font-space font-bold text-4xl md:text-5xl text-white">
            Legacy IP Is <span className="text-error-red">Fraying</span>
          </h2>
          <p className="font-inter text-lg text-metal-silver max-w-2xl mx-auto mt-6">
            Traditional GPU/NPU architectures were never designed for the sparse, high-throughput requirements of modern Transformer inference.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {problems.map((problem, i) => (
            <motion.div
              key={i}
              className="glass-surface p-8 rounded-2xl border border-error-red/10 transition-colors duration-300 hover:border-error-red group relative overflow-hidden"
              initial={{ opacity: 0, y: 30, filter: 'contrast(150%) brightness(50%)' }} // static glitch entrance
              whileInView={{ opacity: 1, y: 0, filter: 'contrast(100%) brightness(100%)' }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.15 }}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-error-red/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
              <div className="relative z-10">
                {/* Lottie fallback using lucide since lottie JSON files require asset loading setup */}
                <motion.div
                  animate={{ rotate: [0, -10, 10, 0] }}
                  transition={{ duration: 0.5, delay: 1, repeat: Infinity, repeatDelay: 5 }}
                >
                  {problem.icon}
                </motion.div>
                
                <h3 className="font-space text-2xl font-bold text-white mb-3 tracking-tight">
                  {problem.title}
                </h3>
                <p className="font-inter text-metal-silver leading-relaxed">
                  {problem.desc}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
