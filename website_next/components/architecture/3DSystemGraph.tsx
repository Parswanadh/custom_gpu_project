'use client';

import { Suspense, useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Float, Line, Sphere, Box } from '@react-three/drei';
import * as THREE from 'three';
import { useReducedMotion } from 'framer-motion';

// Architecture Nodes
const nodes = [
  { id: 'pcie', label: 'PCIE Gen4\nController', pos: [-6, 2, 0] as [number,number,number], color: '#B87333', type: 'box' },
  { id: 'cmd', label: 'Command\nProcessor', pos: [-2, 2, 0] as [number,number,number], color: '#00F5FF', type: 'sphere' },
  { id: 'rom', label: 'Weight ROM\n(Imprinted)', pos: [2, 2, 0] as [number,number,number], color: '#7C3AED', type: 'sphere' },
  { id: 'sram', label: 'Scratchpad\nSRAM', pos: [5, 4, -2] as [number,number,number], color: '#FFD700', type: 'box' },
  { id: 'layer', label: 'Transformer\nLayer (x12)', pos: [6, 2, 0] as [number,number,number], color: '#FF3366', type: 'box' },
  { id: 'units', label: 'RoPE/GQA\nUnits', pos: [5, 0, 2] as [number,number,number], color: '#00FF88', type: 'box' },
  { id: 'kvquant', label: 'KV-Cache\nQuantizer', pos: [2, -2, 0] as [number,number,number], color: '#8892A4', type: 'box' }
];

// Architecture Edges
const edges = [
  ['pcie', 'cmd'],
  ['cmd', 'rom'],
  ['rom', 'layer'],
  ['layer', 'sram'],
  ['layer', 'units'],
  ['layer', 'kvquant'],
  ['units', 'sram']
];

function SystemGraph({ reducedMotion }: { reducedMotion: boolean }) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state, delta) => {
    if (reducedMotion || !groupRef.current) return;
    groupRef.current.rotation.y += delta * 0.1;
  });

  const lines = useMemo(() => {
    return edges.map(edge => {
      const fromNode = nodes.find(n => n.id === edge[0]);
      const toNode = nodes.find(n => n.id === edge[1]);
      if (!fromNode || !toNode) return null;
      return { p1: new THREE.Vector3(...fromNode.pos), p2: new THREE.Vector3(...toNode.pos) };
    }).filter(Boolean) as { p1: THREE.Vector3, p2: THREE.Vector3 }[];
  }, []);

  return (
    <group ref={groupRef}>
      {/* Draw Nodes */}
      {nodes.map(node => (
        <Float key={node.id} speed={reducedMotion ? 0 : 2} rotationIntensity={0.5} floatIntensity={0.5} position={node.pos}>
          {node.type === 'box' ? (
            <Box args={[1.5, 1, 1.5]}>
              <meshStandardMaterial color={node.color} wireframe transparent opacity={0.6} />
              <Box args={[1.4, 0.9, 1.4]} position={[0,0,0]}>
                <meshStandardMaterial color={node.color} metalness={0.8} roughness={0.2} transparent opacity={0.4} />
              </Box>
            </Box>
          ) : (
            <Sphere args={[0.8, 32, 32]}>
              <meshStandardMaterial color={node.color} metalness={0.8} roughness={0.2} wireframe transparent opacity={0.6}/>
            </Sphere>
          )}

          {/* Node Label - Using a more reliable font source or falling back to standard */}
          <Text 
            position={[0, 1.4, 0]} 
            fontSize={0.35} 
            color="#FFFFFF" 
            anchorX="center" 
            anchorY="middle"
            maxWidth={2}
            textAlign="center"
          >
            {node.label}
          </Text>
        </Float>
      ))}

      {/* Draw Edges */}
      {lines.map((line, i) => (
        <Line 
          key={i} 
          points={[line.p1, line.p2]} 
          color="#8892A4" 
          lineWidth={1.5} 
          transparent 
          opacity={0.3} 
          dashed={!reducedMotion}
          dashScale={20}
          dashSize={1}
          dashOffset={0}
        />
      ))}

      {/* Animated Packets traversing edges */}
      {!reducedMotion && lines.map((line, i) => (
        <DataPacket key={`packet-${i}`} p1={line.p1} p2={line.p2} />
      ))}
    </group>
  );
}

function DataPacket({ p1, p2 }: { p1: THREE.Vector3, p2: THREE.Vector3 }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const offset = useMemo(() => Math.random() * 2, []); // Random stagger

  useFrame((state) => {
    if (!meshRef.current) return;
    const time = (state.clock.elapsedTime + offset) % 2; // loops every 2 seconds
    const progress = time / 2;
    meshRef.current.position.lerpVectors(p1, p2, progress);
  });

  return (
    <Sphere ref={meshRef} args={[0.1, 8, 8]}>
      <meshBasicMaterial color="#00F5FF" />
    </Sphere>
  );
}

function LoadingSpinner() {
  return (
    <group>
      <Sphere args={[0.5, 16, 16]}>
        <meshBasicMaterial color="#00F5FF" wireframe />
      </Sphere>
    </group>
  );
}

export default function ThreeDSystemGraph() {
  const reducedMotion = useReducedMotion() ?? false;
  const [isMobile, setIsMobile] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);
  
  if (!isMounted) return <div className="w-full h-[400px] md:h-[600px] bg-silicon-black/20 animate-pulse rounded-3xl" />;

  return (
    <div className="w-full h-[400px] md:h-[600px] bg-silicon-black/50 border border-white/5 rounded-3xl overflow-hidden shadow-2xl backdrop-blur-md relative">
      <Canvas 
        camera={{ position: isMobile ? [0, 8, 20] : [0, 5, 15], fov: 45 }} 
        gl={{ alpha: true, antialias: true }}
      >
        <ambientLight intensity={0.6} />
        <pointLight position={[10, 10, 10]} intensity={1.5} color="#FFD700" />
        <pointLight position={[-10, -10, -10]} intensity={1.5} color="#7C3AED" />
        
        <Suspense fallback={<LoadingSpinner />}>
          <SystemGraph reducedMotion={reducedMotion} />
        </Suspense>
        
        <OrbitControls 
          enableZoom={false} 
          enablePan={false} 
          autoRotate={!reducedMotion} 
          autoRotateSpeed={0.5}
          maxPolarAngle={Math.PI / 1.5}
          minPolarAngle={Math.PI / 4}
        />
      </Canvas>
      <div className="absolute bottom-4 left-0 right-0 text-center pointer-events-none">
        <p className="font-mono text-[10px] md:text-xs text-metal-silver uppercase tracking-widest">Interactive Graph: Drag to Rotate</p>
      </div>
    </div>
  );
}
