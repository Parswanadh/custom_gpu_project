'use client';

import { motion } from 'framer-motion';
import WorkflowBackground from '../shared/WorkflowBackground';
import CircuitGate from '../quality/CircuitGate';

export default function QualityGateSection() {
  return (
    <section id="validation" className="relative py-32 w-full bg-[#03070A] min-h-screen border-t border-white/5 flex items-center">
      <WorkflowBackground color="green" density="medium" />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          
          <motion.div 
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-block px-4 py-1.5 rounded-full border border-oxide-green/30 bg-oxide-green/10 text-oxide-green font-dm text-sm font-bold tracking-widest uppercase mb-6 glow-on-hover">
              Zero-Defect Guarantee
            </div>
            <h2 className="font-space font-bold text-4xl md:text-6xl text-white mb-6">
              5-Stage <br/><span className="text-oxide-green drop-shadow-[0_0_20px_rgba(0,255,136,0.4)]">Quality Gate</span>
            </h2>
            <p className="font-inter text-lg text-metal-silver mb-8 leading-relaxed">
              Before a single line of code reaches your main branch, it must unlock five rigorous sequential quality gates. Failure at any node triggers an immediate loop back to the multi-agent debate stage.
            </p>

            <ul className="space-y-6">
              <li className="flex gap-4">
                <div className="w-8 h-8 rounded-full bg-silicon-gray border border-white/10 flex items-center justify-center font-mono text-neon-cyan flex-shrink-0">1</div>
                <div>
                  <h4 className="text-white font-bold font-space">Syntactical Correctness</h4>
                  <p className="text-metal-silver font-inter text-sm">Validating AST and strict typings across all generated modules.</p>
                </div>
              </li>
              <li className="flex gap-4">
                <div className="w-8 h-8 rounded-full bg-silicon-gray border border-white/10 flex items-center justify-center font-mono text-trace-gold flex-shrink-0">2</div>
                <div>
                  <h4 className="text-white font-bold font-space">Performance Bounding</h4>
                  <p className="text-metal-silver font-inter text-sm">Ensuring algorithmic complexity limits O(n) are not exceeded for critical loops.</p>
                </div>
              </li>
              <li className="flex gap-4">
                <div className="w-8 h-8 rounded-full bg-silicon-gray border border-white/10 flex items-center justify-center font-mono text-plasma-violet flex-shrink-0">3</div>
                <div>
                  <h4 className="text-white font-bold font-space">Security Fuzzing</h4>
                  <p className="text-metal-silver font-inter text-sm">Symbolic execution checking for common injection and logic vulnerabilities.</p>
                </div>
              </li>
            </ul>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <CircuitGate />
          </motion.div>

        </div>
      </div>
    </section>
  );
}
