'use client';

import dynamic from 'next/dynamic';
const SiliconDieCanvas = dynamic(() => import('../hero/SiliconDieCanvas'), { ssr: false });
import GlitchHeadline from '../hero/GlitchHeadline';
import HeroStats from '../hero/HeroStats';
import { motion } from 'framer-motion';

export default function HeroSection() {
  return (
    <section className="relative w-full h-screen min-h-[800px] flex items-center justify-center overflow-hidden">
      {/* 3D WebGL Background Layers */}
      <SiliconDieCanvas />
      
      {/* Post-Canvas absolute CSS Scanline overlay */}
      <div className="scanline" />

      {/* Foreground Container */}
      <div className="relative z-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center flex flex-col items-center pt-24">
        
        {/* SVG Trace Line leading up to the headline (animates on load) */}
        <motion.svg 
          className="w-1 absolute top-0 h-24 mb-6" 
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 1.5, ease: "easeInOut" }}
        >
          <line x1="2" y1="0" x2="2" y2="96" stroke="#00F5FF" strokeWidth="4" strokeLinecap="round" />
        </motion.svg>

        <GlitchHeadline text="BitbyBit Custom Silicon" />

        <motion.p 
          className="max-w-3xl font-inter text-lg md:text-xl text-metal-silver font-light leading-relaxed"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
        >
          A ground-up, cycle-accurate Verilog-2005 architecture explicitly engineered for Transformer inference. <span className="text-trace-gold font-medium">No off-the-shelf IPs.</span> Pure, measurable RTL execution featuring zero-multiplier ternary logic, Compute-in-SRAM, and true <span className="text-neon-cyan font-bold">Silicon Imprinting</span> for hardwired LLMs.
        </motion.p>
        
        <HeroStats />
      </div>
    </section>
  );
}
