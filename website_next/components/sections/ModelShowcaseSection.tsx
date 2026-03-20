'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import WorkflowBackground from '../shared/WorkflowBackground';
import TierStack, { ProfileType } from '../models/TierStack';
import { Zap, Scale, Cpu, Search, BrainCircuit } from 'lucide-react';
import clsx from 'clsx';

const profiles = [
  { id: 'fast', label: 'Fast Extracts', icon: <Zap className="w-5 h-5" /> },
  { id: 'balanced', label: 'Balanced Default', icon: <Scale className="w-5 h-5" /> },
  { id: 'powerful', label: 'Powerful Code Gen', icon: <Cpu className="w-5 h-5" /> },
  { id: 'reasoning', label: 'Deep Reasoning', icon: <BrainCircuit className="w-5 h-5" /> },
  { id: 'research', label: 'Research Agent', icon: <Search className="w-5 h-5" /> },
];

export default function ModelShowcaseSection() {
  const [activeProfile, setActiveProfile] = useState<ProfileType | null>(null);

  const toggleProfile = (id: string) => {
    setActiveProfile(prev => prev === id ? null : id as ProfileType);
  };

  return (
    <section id="models" className="relative py-32 w-full bg-[#050B14] min-h-screen border-t border-white/5 overflow-hidden">
      <WorkflowBackground color="cyan" density="dense" />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full flex flex-col lg:flex-row gap-16 items-center">
        
        {/* Left: Text & Profiles */}
        <motion.div 
          className="lg:w-1/2"
          initial={{ opacity: 0, filter: 'blur(8px)' }}
          whileInView={{ opacity: 1, filter: 'blur(0px)' }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-block px-4 py-1.5 rounded-full border border-neon-cyan/30 bg-neon-cyan/10 text-neon-cyan font-dm text-sm font-bold tracking-widest uppercase mb-6 glow-on-hover">
            Resilient Architecture
          </div>
          <h2 className="font-space font-bold text-4xl md:text-6xl text-white mb-6 tracking-tight">
            The 5-Tier <br/><span className="text-neon-cyan drop-shadow-[0_0_20px_rgba(0,245,255,0.4)]">Cascade</span>
          </h2>
          <p className="font-inter text-lg text-metal-silver mb-8 leading-relaxed">
            API outages happen. Auto-GIT dynamically routes requests through five escalating tiers depending on the task&apos;s cognitive profile, minimizing cost while guaranteeing 99.99% operational uptime.
          </p>

          <h3 className="font-mono text-sm text-white mb-4">INTERACTIVE PROFILE MAPPING:</h3>
          <div className="flex flex-wrap gap-3">
            {profiles.map(p => (
              <button
                key={p.id}
                onClick={() => toggleProfile(p.id)}
                className={clsx(
                  "flex items-center gap-2 px-4 py-2 rounded-xl font-dm text-sm transition-all duration-300 border backdrop-blur-md",
                  activeProfile === p.id 
                    ? "bg-neon-cyan/20 border-neon-cyan text-white shadow-[0_0_15px_rgba(0,245,255,0.3)] scale-105" 
                    : "bg-silicon-gray/50 border-white/10 text-metal-silver hover:bg-white/5 hover:border-white/30"
                )}
              >
                {p.icon} {p.label}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Right: 3D Isometric Stack */}
        <div className="lg:w-1/2 w-full">
          <TierStack activeProfile={activeProfile} />
        </div>

      </div>
    </section>
  );
}
