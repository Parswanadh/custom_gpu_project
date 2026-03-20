'use client';

import { motion } from 'framer-motion';
import WorkflowBackground from '../shared/WorkflowBackground';
import { Target, GitBranch, TerminalSquare, Rocket, Cpu, Activity } from 'lucide-react';

const roadmap = [
  { quarter: 'Epoch 1', title: 'Base Primitives', icon: <TerminalSquare className="w-5 h-5" />, status: 'complete', desc: 'Designing initial Q8.8 ALUs, hardware multipliers, block scaling, and standard GPT-2 inference.' },
  { quarter: 'Epoch 2', title: 'The GPU Subsystem', icon: <Cpu className="w-5 h-5" />, status: 'complete', desc: 'Bridging math cores into a standalone system with AXI4-Lite arrays and an 8-opcode processor.' },
  { quarter: 'Epoch 3', title: 'SOTA In-Hardware', icon: <GitBranch className="w-5 h-5" />, status: 'complete', desc: 'Writing RTL for Mixture-of-Experts routing, PagedAttention MMU concepts, and NVIDIA 2:4 sparsity.' },
  { quarter: 'Epoch 4', title: 'The BitNet Revolution', icon: <Target className="w-5 h-5" />, status: 'complete', desc: 'Tearing out legacy multipliers for BitNet 1.58b ternary engines and Compute-in-SRAM.' },
  { quarter: 'Epoch 5', title: 'Pipeline Unification', icon: <Activity className="w-5 h-5" />, status: 'complete', desc: 'Wiring the dynamic 6-stage data flow (Embed → RoPE → GQA → Softmax → GELU → KV Quant).' },
  { quarter: 'Epoch 6', title: 'Silicon Imprinting', icon: <Rocket className="w-5 h-5" />, status: 'active', desc: 'Crossing the software-hardware boundary. Burning HuggingFace .safetensors directly into Verilog ROM.' }
];

export default function RoadmapSection() {
  return (
    <section id="roadmap" className="relative py-32 w-full bg-[#050B14] border-t border-white/5 overflow-hidden">
      <WorkflowBackground color="copper" density="medium" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full text-center">
        <motion.div 
          className="mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-block px-4 py-1.5 rounded-full border border-die-copper/30 bg-die-copper/10 text-die-copper font-dm text-sm font-bold tracking-widest uppercase mb-6 glow-on-hover">
            The Engineering Journey
          </div>
          <h2 className="font-space font-bold text-4xl md:text-5xl text-white mb-6">
            The <span className="text-die-copper">Evolution</span>
          </h2>
        </motion.div>

        {/* Animated Gantt / Timeline Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 text-left">
          {roadmap.map((item, i) => (
            <motion.div
              key={i}
              className="relative glass-surface p-8 rounded-3xl border border-white/10 hover:border-die-copper/50 transition-all duration-300 group overflow-hidden"
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: i * 0.15 }}
            >
              {item.status === 'active' && (
                <div className="absolute top-0 right-0 w-32 h-32 bg-die-copper opacity-20 blur-[50px] mix-blend-screen pointer-events-none" />
              )}
              
              <div className="flex justify-between items-center mb-6">
                <div className="font-mono text-sm tracking-widest text-metal-silver uppercase flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${item.status === 'active' ? 'bg-die-copper animate-pulse' : 'bg-oxide-green'}`} />
                  {item.quarter}
                </div>
                <div className="p-2 glass-surface border border-white/10 rounded-lg text-white">
                  {item.icon}
                </div>
              </div>
              
              <h3 className="font-space font-bold text-xl text-white mb-3">
                {item.title}
              </h3>
              
              <div className="h-1 w-12 bg-die-copper/30 rounded-full mb-4 group-hover:w-full transition-all duration-500" />
              
              <p className="font-inter text-sm text-metal-silver/90 leading-relaxed">
                {item.desc}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
