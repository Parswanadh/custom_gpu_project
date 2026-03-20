'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, Server } from 'lucide-react';
import clsx from 'clsx';

export type ProfileType = 'fast' | 'balanced' | 'powerful' | 'reasoning' | 'research';

const tiers = [
  { id: 1, name: 'Primary Premium (OpenRouter)', color: '#00FF88', models: ['deepseek-chat', 'gemini-1.5-pro'], profiles: ['powerful', 'balanced'] },
  { id: 2, name: 'Primary Free (OpenRouter)', color: '#00F5FF', models: ['llama-3-70b', 'qwen2.5-coder'], profiles: ['fast', 'balanced'] },
  { id: 3, name: 'Multi-Key Pool (Groq)', color: '#7C3AED', models: ['mixtral-8x7b', 'llama-3-8b'], profiles: ['fast'] },
  { id: 4, name: 'Last Resort (OpenAI)', color: '#FFD700', models: ['gpt-4o-mini'], profiles: ['balanced', 'powerful'] },
  { id: 5, name: 'Offline Fallback (Ollama)', color: '#1A1F2E', models: ['phi4-mini', 'gemma2:2b'], profiles: ['reasoning', 'research', 'fast'] }
];

interface TierStackProps {
  activeProfile: ProfileType | null;
}

export default function TierStack({ activeProfile }: TierStackProps) {
  const [activeTierIndex, setActiveTierIndex] = useState(0);
  const [isSimulating, setIsSimulating] = useState(false);
  const [rateLimitedTiers, setRateLimitedTiers] = useState<number[]>([]);

  // Simulation loop for the API request falling down the stack
  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    if (isSimulating) {
      if (activeTierIndex < tiers.length - 1) {
        timeoutId = setTimeout(() => {
          // 40% chance to rate limit unless we're exactly matching a profile that forces a success
          const is429 = Math.random() > 0.6;
          if (is429) {
            setRateLimitedTiers(prev => [...prev, activeTierIndex]);
            setActiveTierIndex(prev => prev + 1);
          } else {
            setIsSimulating(false);
          }
        }, 1200);
      } else {
        // Reached bottom
        setIsSimulating(false);
      }
    }
    return () => clearTimeout(timeoutId);
  }, [isSimulating, activeTierIndex]);

  const triggerSimulation = () => {
    setIsSimulating(true);
    setActiveTierIndex(0);
    setRateLimitedTiers([]);
  };

  return (
    <div className="relative w-full h-[600px] flex items-center justify-center [perspective:1500px]">
      
      {/* Simulation Button */}
      <button 
        onClick={triggerSimulation}
        disabled={isSimulating}
        className="absolute top-0 right-0 z-50 px-4 py-2 bg-silicon-gray border border-white/10 rounded-full text-sm font-space font-bold hover:bg-white/5 transition-colors disabled:opacity-50 flex items-center gap-2"
      >
        <Server className="w-4 h-4" /> SIMULATE 429 FALLBACK
      </button>

      <div className="relative w-full max-w-lg transition-transform duration-700 [transform-style:preserve-3d] [transform:rotateX(55deg)_rotateZ(-45deg)]">
        {tiers.map((tier, index) => {
          const isRateLimited = rateLimitedTiers.includes(index);
          const isActive = index === activeTierIndex && isSimulating;
          const isFound = index === activeTierIndex && !isSimulating && rateLimitedTiers.length > 0;
          const strZ = 100 - index * 60; // Spread layers out
          
          let opacity = 0.5;
          let highlightClass = 'border-white/10';
          let bgClass = 'bg-silicon-gray/60';

          if (isRateLimited) {
            opacity = 0.3;
            highlightClass = 'border-error-red/50 shadow-[0_0_30px_rgba(255,51,102,0.6)]';
            bgClass = 'bg-error-red/20';
          } else if (isActive) {
            opacity = 1;
            highlightClass = 'border-neon-cyan/50 shadow-[0_0_40px_rgba(0,245,255,0.7)]';
            bgClass = 'bg-neon-cyan/10';
          } else if (isFound) {
            opacity = 1;
            highlightClass = 'border-oxide-green shadow-[0_0_40px_rgba(0,255,136,0.6)]';
            bgClass = 'bg-oxide-green/20';
          } else if (activeProfile) {
            opacity = tier.profiles.includes(activeProfile) ? 1 : 0.2;
            if (tier.profiles.includes(activeProfile)) {
              highlightClass = 'border-white/30';
              bgClass = 'bg-white/10';
            }
          }

          return (
            <motion.div
              key={tier.id}
              className={clsx(
                "absolute inset-0 w-full aspect-[4/3] rounded-2xl border-2 backdrop-blur-md transition-all duration-500",
                highlightClass,
                bgClass
              )}
              style={{
                transform: `translateZ(${strZ}px)`,
                backgroundColor: isActive ? 'transparent' : undefined, // let bgClass handle it
              }}
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              {/* Inner PCB styling */}
              <div 
                className="absolute inset-0 opacity-20 pointer-events-none rounded-xl"
                style={{
                  backgroundImage: `url("data:image/svg+xml,%3Csvg width='20' height='20' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 20L20 0M10 20L20 10M0 10L10 0' stroke='${tier.color.replace('#', '%23')}' stroke-width='1' fill='none'/%3E%3C/svg%3E")`
                }}
              />
              
              <div className="absolute inset-4 flex flex-col justify-between">
                <div className="flex justify-between items-start">
                  <div className="font-mono text-xs font-bold px-2 py-1 rounded bg-silicon-black/50 border border-white/10" style={{ color: tier.color }}>
                    TIER {tier.id}
                  </div>
                  {isRateLimited && (
                    <motion.div 
                      initial={{ scale: 0 }} 
                      animate={{ scale: [1, 1.2, 1] }} 
                      transition={{ repeat: Infinity }}
                      className="text-error-red flex items-center gap-1 font-mono text-sm bg-silicon-black/80 px-2 py-1 rounded border border-error-red/50"
                    >
                      <AlertTriangle className="w-4 h-4" /> 429
                    </motion.div>
                  )}
                  {isFound && (
                    <motion.div 
                      initial={{ scale: 0 }} 
                      animate={{ scale: 1 }} 
                      className="text-oxide-green font-mono text-sm bg-silicon-black/80 px-2 py-1 rounded border border-oxide-green/50"
                    >
                      SUCCESS 200
                    </motion.div>
                  )}
                </div>
                
                <div>
                  <h3 className="font-space font-bold text-xl text-white mb-2" style={{ textShadow: '0 2px 10px rgba(0,0,0,0.8)' }}>
                    {tier.name}
                  </h3>
                  <div className="flex gap-2 font-mono text-xs text-metal-silver">
                    {tier.models.join(', ')}
                  </div>
                </div>
              </div>

              {/* API Request Orb passing through */}
              <AnimatePresence>
                {isActive && (
                  <motion.div
                    className="absolute left-1/2 top-1/2 w-8 h-8 -ml-4 -mt-4 bg-neon-cyan rounded-full shadow-[0_0_30px_#00F5FF_inset]"
                    initial={{ translateZ: 100, opacity: 0 }}
                    animate={{ translateZ: 0, opacity: 1 }}
                    exit={{ translateZ: -100, opacity: 0 }}
                    transition={{ duration: 0.5 }}
                  />
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
