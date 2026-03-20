'use client';

import { motion } from 'framer-motion';
import WorkflowBackground from '../shared/WorkflowBackground';
import PipelineDAG from '../pipeline/PipelineDAG';

export default function PipelineSection() {
  return (
    <section id="pipeline" className="relative py-32 w-full bg-[#03060C] min-h-screen flex items-center">
      <WorkflowBackground color="copper" density="medium" />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full">
        <div className="grid lg:grid-cols-12 gap-12 items-center">
          
          <motion.div 
            className="lg:col-span-5"
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-block px-4 py-1.5 rounded-full border border-die-copper/30 bg-die-copper/10 text-die-copper font-dm text-sm font-bold tracking-widest uppercase mb-6">
              The 6-Stage Dynamic Flow
            </div>
            <h2 className="font-space font-bold text-4xl md:text-5xl text-white mb-6">
              Hardware Transformer <br/><span className="text-die-copper">Pipeline</span>
            </h2>
            <p className="font-inter text-lg text-metal-silver mb-8 leading-relaxed">
              Tokens traverse our custom hardware pipeline end-to-end without host intervention, completing a 12-layer model in just 341 cycles.
            </p>

            <ul className="space-y-4 font-mono text-sm text-metal-silver">
              <li className="flex items-center gap-3">
                <span className="text-neon-cyan">01</span> Hardware-Native RoPE & GQA
              </li>
              <li className="flex items-center gap-3">
                <span className="text-neon-cyan">02</span> Parallelized HW Softmax & Poly GELU
              </li>
              <li className="flex items-center gap-3">
                <span className="text-neon-cyan">03</span> INT4 KV Quantization On-The-Fly
              </li>
            </ul>
          </motion.div>

          {/* Interactive DAG Graph */}
          <motion.div 
            className="lg:col-span-7"
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <PipelineDAG />
          </motion.div>

        </div>
      </div>
    </section>
  );
}
