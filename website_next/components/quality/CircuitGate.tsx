'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Lock, Unlock, ShieldCheck, Zap, Code2, TextSelect, ScanLine } from 'lucide-react';
import clsx from 'clsx';

const stages = [
  { id: 1, name: 'Syntax Parsing', icon: <Code2 />, target: 100, color: '#00F5FF' },
  { id: 2, name: 'Type Validation', icon: <TextSelect />, target: 100, color: '#B87333' },
  { id: 3, name: 'Static Analysis', icon: <ScanLine />, target: 100, color: '#7C3AED' },
  { id: 4, name: 'O(n) Perf Bound', icon: <Zap />, target: 94, color: '#FFD700' },
  { id: 5, name: 'Sym. Execution Sec', icon: <ShieldCheck />, target: 100, color: '#00FF88' }
];

export default function CircuitGate() {
  const [activeStage, setActiveStage] = useState(0);
  const [isAuthorizing, setIsAuthorizing] = useState(false);
  const [authorized, setAuthorized] = useState(false);

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    if (isAuthorizing) {
      if (activeStage < stages.length) {
        timeoutId = setTimeout(() => {
          setActiveStage(prev => prev + 1);
        }, 800);
      } else {
        setAuthorized(true);
        setIsAuthorizing(false);
      }
    }
    return () => clearTimeout(timeoutId);
  }, [isAuthorizing, activeStage]);

  const triggerValidation = () => {
    setActiveStage(0);
    setAuthorized(false);
    setIsAuthorizing(true);
  };

  return (
    <div className="relative w-full max-w-4xl mx-auto bg-[#070B14] border border-white/10 p-8 rounded-3xl shadow-2xl backdrop-blur-xl">
      
      {/* Circuit Board Trace Top Decorator */}
      <svg className="absolute top-0 left-10 w-24 h-4 opacity-30" viewBox="0 0 100 20">
        <path d="M 0 0 L 20 20 L 100 20" fill="none" stroke="#00FF88" strokeWidth="2" />
      </svg>

      <div className="flex justify-between items-center mb-10">
        <div>
          <h3 className="font-space font-bold text-2xl text-white">Validation Gates</h3>
          <p className="font-mono text-xs text-metal-silver mt-1">STRICT MODE ENFORCED</p>
        </div>
        {!authorized && !isAuthorizing && (
          <button 
            onClick={triggerValidation}
            className="px-6 py-2 bg-oxide-green/10 border border-oxide-green/50 text-oxide-green font-space font-bold text-sm rounded-full hover:bg-oxide-green hover:text-silicon-black transition-all glow-on-hover"
          >
            INITIATE SCAN
          </button>
        )}
      </div>

      <div className="relative space-y-6">
        {/* Continuous background trace connecting the logic gates */}
        <div className="absolute left-[38px] top-6 bottom-6 w-0.5 bg-silicon-gray z-0" />

        {stages.map((stage, index) => {
          const isPassed = activeStage > index;
          const isCurrent = activeStage === index && isAuthorizing;
          
          return (
            <div key={stage.id} className="relative z-10 flex items-center gap-6">
              
              {/* Gate Node */}
              <motion.div 
                className={clsx(
                  "w-20 h-20 rounded-xl border flex items-center justify-center bg-silicon-black transition-all duration-300",
                  isPassed ? "border-oxide-green shadow-[0_0_20px_rgba(0,255,136,0.4)]" :
                  isCurrent ? `border-[${stage.color}] shadow-[0_0_20px_${stage.color}60]` :
                  "border-silicon-gray"
                )}
                animate={isCurrent ? { scale: [1, 1.05, 1] } : {}}
                transition={{ duration: 0.8, repeat: Infinity }}
              >
                {isPassed ? (
                  <Unlock className="w-8 h-8 text-oxide-green" />
                ) : (
                  <Lock className={clsx("w-8 h-8", isCurrent ? "text-white" : "text-metal-silver/30")} />
                )}
              </motion.div>

              {/* Progress & Info */}
              <div className="flex-1 glass-surface p-4 rounded-xl border border-white/5 relative overflow-hidden">
                <div className="flex justify-between items-center mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-white/70">{stage.icon}</span>
                    <span className="font-space font-bold text-white text-lg">{stage.name}</span>
                  </div>
                  <span className="font-mono text-xs font-bold text-metal-silver">
                    {isPassed ? `${stage.target}%` : isCurrent ? 'SCANNING...' : 'PENDING'}
                  </span>
                </div>
                
                {/* Progress Bar inside */}
                <div className="h-1.5 w-full bg-silicon-black rounded-full overflow-hidden">
                  <motion.div 
                    className="h-full rounded-full"
                    style={{ backgroundColor: stage.color }}
                    initial={{ width: 0 }}
                    animate={{ width: isPassed ? `${stage.target}%` : isCurrent ? ['0%', '100%'] : '0%' }}
                    transition={isCurrent ? { duration: 0.8, ease: "linear" } : { duration: 0.3 }}
                  />
                </div>
              </div>

            </div>
          );
        })}
      </div>

      {/* Final Deployment Stamp */}
      {authorized && (
        <motion.div 
          className="absolute inset-0 z-50 flex items-center justify-center bg-silicon-black/60 backdrop-blur-sm rounded-3xl"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <motion.div
            className="border-4 border-oxide-green text-oxide-green font-space font-bold text-4xl md:text-6xl px-8 py-4 uppercase tracking-widest shadow-[0_0_50px_rgba(0,255,136,0.3)] transform -rotate-12 bg-silicon-black/90"
            initial={{ scale: 3, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 10 }}
          >
            Authorized
          </motion.div>
        </motion.div>
      )}

    </div>
  );
}
