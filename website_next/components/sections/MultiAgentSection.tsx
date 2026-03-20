'use client';

import { motion } from 'framer-motion';
import { Canvas } from '@react-three/fiber';
import { useReducedMotion } from 'framer-motion';
import CritiqueMatrix from '../debate/CritiqueMatrix';

// Minimal Neural Mesh Background Fallback
function NeuralMesh() {
  const reducedMotion = useReducedMotion();
  
  if (reducedMotion) {
    return <div className="absolute inset-0 bg-silicon-black" />;
  }

  return (
    <div className="absolute inset-0 z-0 opacity-20 pointer-events-none">
      <Canvas camera={{ position: [0, 0, 5] }}>
        <ambientLight intensity={0.5} />
        <mesh rotation={[1, 1, 1]}>
          <icosahedronGeometry args={[4, 1]} />
          <meshBasicMaterial color="#7C3AED" wireframe transparent opacity={0.3} />
        </mesh>
      </Canvas>
    </div>
  );
}

const experts = [
  { role: 'Architecture Lead', color: 'border-plasma-violet', bg: 'bg-plasma-violet/10', traits: ['Scalability', 'Patterns'] },
  { role: 'Security Auditor', color: 'border-neon-cyan', bg: 'bg-neon-cyan/10', traits: ['Vuln Check', 'Auth'] },
  { role: 'Performance Engineer', color: 'border-trace-gold', bg: 'bg-trace-gold/10', traits: ['O(n) Eval', 'Memory'] }
];

export default function MultiAgentSection() {
  return (
    <section id="debate" className="relative py-32 w-full bg-silicon-black overflow-hidden border-t border-white/5">
      <NeuralMesh />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 w-full">
        
        {/* Header */}
        <motion.div 
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-flex items-center gap-3 px-4 py-1.5 rounded-full border border-plasma-violet/30 bg-plasma-violet/10 text-plasma-violet font-dm text-sm font-bold tracking-widest uppercase mb-6 glow-on-hover cursor-pointer">
            Inspired by Stanford STORM <span>↗</span>
          </div>
          <h2 className="font-space font-bold text-4xl md:text-5xl text-white">
            Adversarial <span className="text-plasma-violet">Multi-Agent</span> Debate
          </h2>
          <p className="mt-4 font-inter text-lg text-metal-silver max-w-2xl mx-auto">
            Code generation isn&apos;t single-shot. We instantiate specialized domain personas to forcibly critique, rip apart, and rebuild proposed implementations until convergence.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 items-center">
          
          {/* Left: Materializing Cards */}
          <div className="space-y-4">
            {experts.map((e, i) => (
              <motion.div
                key={i}
                className={`glass-surface p-6 rounded-2xl border ${e.color} transition-all duration-500 hover:scale-105 hover:shadow-[0_0_30px_rgba(124,58,237,0.3)]`}
                initial={{ opacity: 0, scale: 0.8, filter: 'blur(10px)' }}
                whileInView={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ duration: 0.6, delay: i * 0.2 }}
              >
                <div className="flex justify-between items-start mb-4">
                  <h3 className="font-space font-bold text-xl text-white">{e.role}</h3>
                  <div className={`w-3 h-3 rounded-full ${e.bg} animate-pulse border border-white/50`} />
                </div>
                <div className="flex gap-2">
                  {e.traits.map(t => (
                    <span key={t} className="text-xs font-mono px-2 py-1 bg-white/5 rounded text-metal-silver">
                      {t}
                    </span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>

          {/* Right: Critique Matrix */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <CritiqueMatrix />
          </motion.div>

        </div>
      </div>
    </section>
  );
}
