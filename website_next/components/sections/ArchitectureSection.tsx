'use client';

import { motion } from 'framer-motion';
import WorkflowBackground from '../shared/WorkflowBackground';
import dynamic from 'next/dynamic';
const ThreeDSystemGraph = dynamic(() => import('../architecture/3DSystemGraph'), { ssr: false });

export default function ArchitectureSection() {
  return (
    <section id="architecture" className="relative py-32 w-full bg-[#020408] min-h-screen overflow-hidden">
      <WorkflowBackground color="violet" density="sparse" />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full">
        
        <motion.div 
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-block px-4 py-1.5 rounded-full border border-plasma-violet/30 bg-plasma-violet/10 text-plasma-violet font-dm text-sm font-bold tracking-widest uppercase mb-6 glow-on-hover">
            Silicon Imprinting
          </div>
          <h2 className="font-space font-bold text-4xl md:text-6xl text-white mb-6">
            Hard-Burned Into <span className="text-plasma-violet">Silicon.</span>
          </h2>
          <p className="font-inter text-lg text-metal-silver max-w-3xl mx-auto leading-relaxed">
            Why fetch weights from slow DDR4? Our architecture supports <span className="text-white font-medium">Hardware-Imprinted Models</span>. We extract exact weights from Google&apos;s Gemma 3 270M, compress them to Q8.8 fixed-point, and hard-burn them directly into the Verilog compiled ROM for unyielding 8-cycle latency.
          </p>
        </motion.div>

        <motion.div
           initial={{ opacity: 0, scale: 0.95, filter: 'blur(10px)' }}
           whileInView={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
           viewport={{ once: true }}
           transition={{ duration: 0.8, delay: 0.2 }}
        >
          <ThreeDSystemGraph />
        </motion.div>

      </div>
    </section>
  );
}
