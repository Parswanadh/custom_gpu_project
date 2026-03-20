'use client';

import { motion } from 'framer-motion';
import ErrorTimeline from '../error-memory/ErrorTimeline';
import LessonTerminal from '../error-memory/LessonTerminal';
import CountUp from '../shared/CountUp';

// Scrolling Hex Background Array
const hexSource = "0101101000101101";

function HexBackground() {
  return (
    <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none opacity-5">
      <div className="absolute top-0 right-10 bottom-0 w-32 font-mono text-xs text-die-copper break-all tracking-widest leading-loose"
           style={{
             WebkitMaskImage: 'linear-gradient(to bottom, transparent, black 10%, black 90%, transparent)'
           }}
      >
        <motion.div
           animate={{ y: [0, -1000] }}
           transition={{ ease: "linear", duration: 20, repeat: Infinity }}
        >
          {Array(200).fill(hexSource).join(" ")}
        </motion.div>
      </div>
    </div>
  );
}

export default function ErrorMemorySection() {
  return (
    <section id="errors" className="relative py-32 w-full bg-silicon-black min-h-screen border-t border-white/5 overflow-hidden">
      <HexBackground />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full">
        <div className="text-center mb-16">
          <div className="inline-block px-4 py-1.5 rounded-full border border-oxide-green/30 bg-oxide-green/10 text-oxide-green font-dm text-sm font-bold tracking-widest uppercase mb-6 glow-on-hover">
            Self-Healing Memory
          </div>
          <h2 className="font-space font-bold text-4xl md:text-5xl text-white mb-6">
            <span className="text-error-red">233 Errors.</span> <span className="text-oxide-green">0 Manual Fixes.</span>
          </h2>
          <p className="font-inter text-lg text-metal-silver max-w-3xl mx-auto leading-relaxed">
            Every compiler failure, linter warning, and runtime crash is parsed, reasoned over, and abstracted into a permanent system prompt vector. Auto-GIT doesn&apos;t just fix the code—it learns never to make that mistake again.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-16 items-start mt-16">
          
          {/* Left: Progression Timeline */}
          <div className="relative">
            <h3 className="font-space font-bold text-2xl text-white mb-8 bg-silicon-gray/80 inline-block px-4 py-2 rounded-lg border border-white/10 backdrop-blur-sm">
              Evolution of Run #27
            </h3>
            <ErrorTimeline />
          </div>

          {/* Right: Lesson Injection Terminal & Stats */}
          <motion.div 
            className="sticky top-32 space-y-8"
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <div className="glass-surface p-6 rounded-2xl border border-white/5">
              <h3 className="font-space font-bold text-xl text-white mb-4">Real-time Memory Injection</h3>
              <LessonTerminal />
            </div>

            <div className="glass-surface p-8 rounded-2xl border border-oxide-green/20 flex items-center justify-between glow-on-hover backdrop-blur-xl">
              <div>
                <div className="text-sm font-dm text-metal-silver uppercase font-bold tracking-widest mb-1">
                  Cumulative Error Vectors
                </div>
                <div className="font-space font-bold text-4xl text-white">
                  <CountUp end={233} duration={2} /> stored
                </div>
              </div>
              <div className="w-16 h-16 rounded-full border-4 border-oxide-green border-t-transparent animate-spin flex items-center justify-center">
                <div className="w-8 h-8 rounded-full bg-oxide-green/20" />
              </div>
            </div>
          </motion.div>

        </div>
      </div>
    </section>
  );
}
