'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Play, Check, Cpu } from 'lucide-react';
import clsx from 'clsx';

type NodeState = 'idle' | 'active' | 'complete' | 'error';

interface PipelineNode {
  id: string;
  label: string;
  x: number;
  y: number;
  preview: string;
}

const nodes: PipelineNode[] = [
  { id: 'rope', label: 'RoPE Encoder', x: 10, y: 50, preview: 'lut.fetch(pos_id); return embeddings;' },
  { id: 'gqa', label: 'GQA Matrix', x: 26, y: 50, preview: 'fetch(K, V); attn_score(Q, K);' },
  { id: 'softmax', label: 'HW Softmax', x: 42, y: 50, preview: 'max_sub_tree(); exp_lut(); sum();' },
  { id: 'gelu', label: 'Poly GELU', x: 58, y: 50, preview: 'approx_poly_lut(hidden_states);' },
  { id: 'quant', label: 'INT4 Quant', x: 74, y: 50, preview: 'q8_8.to(int4_group); KV_cache.write();' },
  { id: 'compress', label: 'Compression', x: 90, y: 50, preview: 'compress_residual(8bit_dynamic);' },
];

const edges = [
  { from: 'rope', to: 'gqa' },
  { from: 'gqa', to: 'softmax' },
  { from: 'softmax', to: 'gelu' },
  { from: 'gelu', to: 'quant' },
  { from: 'quant', to: 'compress' },
];

export default function PipelineDAG() {
  const [activeNodes, setActiveNodes] = useState<Record<string, NodeState>>(() => 
    nodes.reduce((acc, n) => ({ ...acc, [n.id]: 'idle' }), {})
  );
  const [isPlaying, setIsPlaying] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 1024);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const resetPipeline = () => {
    setActiveNodes(nodes.reduce((acc, n) => ({ ...acc, [n.id]: 'idle' }), {}));
    setIsPlaying(false);
  };

  const playPipeline = async () => {
    if (isPlaying) return;
    setIsPlaying(true);
    resetPipeline();

    const sequence = [
      ['rope'],
      ['gqa'],
      ['softmax'],
      ['gelu'],
      ['quant'],
      ['compress']
    ];

    for (const step of sequence) {
      // Set to active
      setActiveNodes(prev => {
        const next = { ...prev };
        step.forEach(id => { next[id] = 'active'; });
        return next;
      });

      await new Promise(r => setTimeout(r, 800)); // Faster for demo

      // Set to complete
      setActiveNodes(prev => {
        const next = { ...prev };
        step.forEach(id => { next[id] = 'complete'; });
        return next;
      });
    }
    
    setIsPlaying(false);
  };

  return (
    <div className={clsx(
      "relative w-full bg-silicon-black/50 border border-white/5 rounded-3xl p-6 backdrop-blur-md overflow-hidden transition-all duration-500",
      isMobile ? "h-auto py-20 flex flex-col items-center gap-16" : "h-[500px]"
    )}>
      
      {/* Play Button */}
      <button 
        onClick={playPipeline}
        disabled={isPlaying}
        className="absolute top-6 right-6 z-30 flex items-center gap-2 px-4 py-2 bg-silicon-gray border border-white/10 rounded-full text-metal-silver hover:text-white hover:border-neon-cyan transition-all disabled:opacity-50"
      >
        <Play className="w-4 h-4" />
        <span className="font-space font-bold text-sm tracking-wide">
          {isPlaying ? 'EXECUTING...' : 'PLAY PIPELINE'}
        </span>
      </button>

      {/* SVG Canvas for Edges - ONLY SHOW ON DESKTOP */}
      {!isMobile && (
        <svg className="absolute inset-0 w-full h-full pointer-events-none z-10">
          <defs>
            <linearGradient id="edge-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#1A1F2E" />
              <stop offset="50%" stopColor="#8892A4" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#1A1F2E" />
            </linearGradient>
          </defs>

          {edges.map((edge, i) => {
            const fromNode = nodes.find(n => n.id === edge.from)!;
            const toNode = nodes.find(n => n.id === edge.to)!;
            
            const isActive = activeNodes[edge.from] === 'complete' && activeNodes[edge.to] === 'active';
            
            return (
              <g key={i}>
                <line 
                  x1={`${fromNode.x}%`} y1={`${fromNode.y}%`} 
                  x2={`${toNode.x}%`} y2={`${toNode.y}%`}
                  stroke="url(#edge-grad)"
                  strokeWidth="2"
                />
                
                {isActive && (
                  <motion.circle
                    r="4"
                    fill="#00F5FF"
                    filter="drop-shadow(0 0 8px #00F5FF)"
                    initial={{ cx: `${fromNode.x}%`, cy: `${fromNode.y}%` }}
                    animate={{ cx: `${toNode.x}%`, cy: `${toNode.y}%` }}
                    transition={{ duration: 1, ease: 'linear', repeat: Infinity }}
                  />
                )}
              </g>
            );
          })}
        </svg>
      )}

      {/* Nodes Render */}
      {nodes.map((node, index) => {
        const state = activeNodes[node.id];
        
        return (
          <div
            key={node.id}
            className={clsx(
              "z-20",
              isMobile ? "relative" : "absolute transform -translate-x-1/2 -translate-y-1/2"
            )}
            style={!isMobile ? { left: `${node.x}%`, top: `${node.y}%` } : {}}
            onMouseEnter={() => setHoveredNode(node.id)}
            onMouseLeave={() => setHoveredNode(null)}
          >
            {/* Mobile Connecting Line */}
            {isMobile && index < nodes.length - 1 && (
              <div className="absolute top-full left-1/2 h-16 w-px bg-white/10 -translate-x-1/2" />
            )}

            {/* Silicon Die Node Representation */}
            <div className={clsx(
              "relative w-16 h-16 bg-[#11141D] rounded-lg border-2 flex items-center justify-center transition-all duration-300 shadow cursor-help",
              state === 'idle' && "border-silicon-gray shadow-none",
              state === 'active' && "border-neon-cyan shadow-[0_0_20px_rgba(0,245,255,0.6)] scale-110",
              state === 'complete' && "border-oxide-green shadow-[0_0_15px_rgba(0,255,136,0.3)]",
              state === 'error' && "border-error-red shadow-[0_0_20px_rgba(255,51,102,0.6)]"
            )}>
              <div className="absolute inset-1 border border-dashed border-white/10 rounded-sm pointer-events-none" />
              
              {state === 'complete' ? (
                <Check className="w-6 h-6 text-oxide-green" />
              ) : (
                <Cpu className={clsx(
                  "w-6 h-6",
                  state === 'active' ? "text-neon-cyan" : "text-metal-silver"
                )} />
              )}
            </div>

            <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-3 font-space font-bold text-xs tracking-wider text-metal-silver whitespace-nowrap text-center">
              {node.label.toUpperCase()}
              {isMobile && <p className="font-mono text-[10px] text-neon-cyan opacity-60 mt-1">{node.preview}</p>}
            </div>

            {/* Hover Tooltip Preview (Desktop only) */}
            {!isMobile && (
              <AnimatePresence>
                {hoveredNode === node.id && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9, y: -10 }}
                    animate={{ opacity: 1, scale: 1, y: -20 }}
                    exit={{ opacity: 0, scale: 0.9, y: -10 }}
                    className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-4 bg-silicon-black/90 backdrop-blur-xl border border-white/10 p-3 rounded-lg whitespace-nowrap z-50 shadow-[0_10px_30px_rgba(0,0,0,0.8)]"
                  >
                    <p className="font-mono text-xs text-neon-cyan">{node.preview}</p>
                  </motion.div>
                )}
              </AnimatePresence>
            )}
          </div>
        );
      })}
    </div>
  );
}
