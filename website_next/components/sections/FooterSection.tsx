'use client';

import { Github, Twitter, Linkedin, Terminal } from 'lucide-react';
import { motion } from 'framer-motion';

export default function FooterSection() {
  return (
    <footer className="relative w-full bg-[#020408] border-t border-white/10 pt-24 pb-12 overflow-hidden">
      {/* Silicon Motif Background SVG */}
      <div className="absolute inset-0 opacity-[0.03] pointer-events-none">
        <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
          <pattern id="silicon" width="60" height="60" patternUnits="userSpaceOnUse">
            <path d="M 0 60 L 60 0 M 30 60 L 60 30 M 0 30 L 30 0" stroke="#FFFFFF" strokeWidth="1" fill="none" strokeDasharray="5,5"/>
            <rect x="25" y="25" width="10" height="10" fill="#FFFFFF" />
          </pattern>
          <rect width="100%" height="100%" fill="url(#silicon)" />
        </svg>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 flex flex-col md:flex-row justify-between items-center md:items-start gap-12">
        
        {/* Brand */}
        <div className="text-center md:text-left">
          <div className="flex items-center justify-center md:justify-start gap-2 text-white font-space font-bold text-2xl mb-4">
            <Terminal className="w-8 h-8 text-neon-cyan" />
            <span>BITBYBIT</span>
          </div>
          <p className="font-mono text-[10px] text-metal-silver max-w-sm leading-relaxed uppercase tracking-widest">
            Cycle-accurate Verilog-2005 hardware architectures explicitly engineered for Transformer inference.
          </p>
        </div>

        {/* Links Array with animated trace hovering */}
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-8 text-center md:text-left font-dm text-sm">
          <div className="space-y-4 flex flex-col items-center md:items-start">
            <h4 className="font-space font-bold tracking-widest text-white uppercase text-xs mb-2">Platform</h4>
            {['Architecture', 'Performance', 'Documentation'].map(link => (
              <a key={link} href={`#${link.toLowerCase()}`} className="relative group text-metal-silver hover:text-white transition-colors uppercase text-[10px] tracking-widest font-mono">
                {link}
                <motion.span className="absolute -bottom-1 left-0 w-0 h-[1px] bg-neon-cyan transition-all duration-300 group-hover:w-full" />
              </a>
            ))}
          </div>
          <div className="space-y-4 flex flex-col items-center md:items-start">
            <h4 className="font-space font-bold tracking-widest text-white uppercase text-xs mb-2">Company</h4>
            {['About', 'Careers', 'Contact'].map(link => (
              <a key={link} href="#" className="relative group text-metal-silver hover:text-white transition-colors">
                {link}
                <motion.span className="absolute -bottom-1 left-0 w-0 h-[1px] bg-neon-cyan transition-all duration-300 group-hover:w-full" />
              </a>
            ))}
          </div>
          <div className="col-span-2 lg:col-span-1 space-y-4 flex flex-col items-center md:items-start">
            <h4 className="font-space font-bold tracking-widest text-white uppercase text-xs mb-2">Socials</h4>
            <div className="flex gap-4">
              <a href="#" className="w-10 h-10 rounded-full border border-white/10 flex items-center justify-center hover:bg-white/5 hover:border-white/30 hover:text-white text-metal-silver transition-all duration-300">
                <Github className="w-4 h-4" />
              </a>
              <a href="#" className="w-10 h-10 rounded-full border border-white/10 flex items-center justify-center hover:bg-white/5 hover:border-white/30 hover:text-neon-cyan text-metal-silver transition-all duration-300">
                <Twitter className="w-4 h-4" />
              </a>
              <a href="#" className="w-10 h-10 rounded-full border border-white/10 flex items-center justify-center hover:bg-white/5 hover:border-white/30 hover:text-[#0a66c2] text-metal-silver transition-all duration-300">
                <Linkedin className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-16 pt-8 border-t border-white/5 flex flex-col md:flex-row items-center justify-between text-xs font-mono text-metal-silver/50">
        <p>© 2026 BitbyBit Custom Silicon. Cycle-Accurate Execution.</p>
        <div className="flex gap-4 mt-4 md:mt-0">
          <a href="#" className="hover:text-white transition-colors">Privacy</a>
          <a href="#" className="hover:text-white transition-colors">Terms</a>
        </div>
      </div>
    </footer>
  );
}
