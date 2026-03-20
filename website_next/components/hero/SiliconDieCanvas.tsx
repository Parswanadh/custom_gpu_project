'use client';

import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Float, Box, Plane, Line, Sphere } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { useReducedMotion } from 'framer-motion';

// --- ASSEMBLY SUB-COMPONENTS --- //

const SubstrateQuads = ({ assemblyProgress }: { assemblyProgress: number }) => {
  // 4 pieces of the silicon substrate that slide in from corners
  const quads = [
    { pos: [-2, 0, -2], start: [-15, 5, -15] }, // Top Left
    { pos: [2, 0, -2], start: [15, 5, -15] },  // Top Right
    { pos: [-2, 0, 2], start: [-15, 5, 15] },  // Bottom Left
    { pos: [2, 0, 2], start: [15, 5, 15] },   // Bottom Right
  ];

  return (
    <group>
      {quads.map((q, i) => {
        const currentPos = new THREE.Vector3().fromArray(q.start).lerp(new THREE.Vector3().fromArray(q.pos), Math.min(1, assemblyProgress * 1.5));
        return (
          <Box key={i} args={[4, 0.15, 4]} position={currentPos}>
            <meshStandardMaterial 
              color="#020408" 
              metalness={1} 
              roughness={0.1} 
              transparent 
              opacity={assemblyProgress > 0.1 ? 1 : 0} 
            />
          </Box>
        );
      })}
    </group>
  );
};

const AssemblyCore = ({ assemblyProgress }: { assemblyProgress: number }) => {
  const coreRef = useRef<THREE.Group>(null);
  
  useFrame((state) => {
    if (!coreRef.current) return;
    const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.1 + 0.3;
    if (assemblyProgress > 0.8) {
      coreRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 4) * 0.02);
    }
  });

  // Core drops from top
  const coreY = assemblyProgress < 0.5 ? 10 : THREE.MathUtils.lerp(10, 0.06, (assemblyProgress - 0.5) * 2);
  const coreOpacity = assemblyProgress < 0.5 ? 0 : (assemblyProgress - 0.5) * 2;

  return (
    <group ref={coreRef} position={[0, coreY, 0]}>
      <Plane args={[4.5, 4.5]} rotation={[-Math.PI / 2, 0, 0]}>
        <meshBasicMaterial color="#00F5FF" transparent opacity={coreOpacity * 0.3} blending={THREE.AdditiveBlending} />
      </Plane>
      <Plane args={[2, 2]} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
        <meshBasicMaterial color="#7C3AED" transparent opacity={coreOpacity * 0.5} blending={THREE.AdditiveBlending} />
      </Plane>
    </group>
  );
};

const FloatingALU = ({ index, assemblyProgress }: { index: number, assemblyProgress: number }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const startPos = useMemo(() => [
    (Math.random() - 0.5) * 40,
    Math.random() * 20 + 10,
    (Math.random() - 0.5) * 40
  ], []);
  
  const targetPos = useMemo(() => [
    (index % 4 - 1.5) * 1.5,
    0.1,
    (Math.floor(index / 4) - 1.5) * 1.5
  ], [index]);

  useFrame((state) => {
    if (!meshRef.current || assemblyProgress < 0.3) return;
    const p = Math.min(1, (assemblyProgress - 0.3) * 2);
    meshRef.current.position.x = THREE.MathUtils.lerp(startPos[0], targetPos[0], p);
    meshRef.current.position.y = THREE.MathUtils.lerp(startPos[1], targetPos[1], p);
    meshRef.current.position.z = THREE.MathUtils.lerp(startPos[2], targetPos[2], p);
    meshRef.current.rotation.x += 0.01;
    meshRef.current.rotation.y += 0.01;
  });

  return (
    <Box ref={meshRef} args={[0.4, 0.4, 0.4]} position={startPos as [number, number, number]}>
      <meshStandardMaterial color="#1A1F2E" metalness={0.8} roughness={0.2} transparent opacity={assemblyProgress > 0.3 ? 1 : 0} />
    </Box>
  );
};

const AssemblyTraces = ({ assemblyProgress }: { assemblyProgress: number }) => {
  const lines = useMemo(() => {
    const l = [];
    for (let i = 0; i < 30; i++) {
      const points = [];
      let x = (Math.random() - 0.5) * 7;
      let z = (Math.random() - 0.5) * 7;
      points.push(new THREE.Vector3(x, 0.04, z));
      for (let j = 0; j < 3; j++) {
        if (Math.random() > 0.5) x += (Math.random() - 0.5) * 2;
        else z += (Math.random() - 0.5) * 2;
        points.push(new THREE.Vector3(Math.max(-3.8, Math.min(3.8, x)), 0.04, Math.max(-3.8, Math.min(3.8, z))));
      }
      l.push(points);
    }
    return l;
  }, []);

  const opacity = assemblyProgress < 0.7 ? 0 : (assemblyProgress - 0.7) * 3.3;

  return (
    <group>
      {lines.map((pts, i) => (
        <Line key={i} points={pts} color="#FFD700" lineWidth={1.5} transparent opacity={opacity} />
      ))}
    </group>
  );
};

// Main Scene
export default function SiliconDieCanvas() {
  const [assemblyProgress, setAssemblyProgress] = useState(0);
  const reducedMotion = useReducedMotion() ?? false;
  const [isMobile, setIsMobile] = useState(false);
  const dieGroupRef = useRef<THREE.Group>(null);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  useFrame((state, delta) => {
    if (reducedMotion) {
      setAssemblyProgress(1);
      return;
    }
    // Slowly advance assembly over 4 seconds
    if (assemblyProgress < 1) {
      setAssemblyProgress(prev => Math.min(1, prev + delta * 0.25));
    }

    if (dieGroupRef.current && assemblyProgress >= 1) {
      dieGroupRef.current.rotation.y += delta * 0.2;
    }
  });

  return (
    <div className="absolute inset-0 z-0 bg-silicon-black overflow-hidden">
      <Canvas 
        shadows 
        camera={{ position: [12, 10, 20], fov: 45 }}
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
        dpr={isMobile ? [1, 1] : [1, 1.5]}
      >
        <color attach="background" args={['#020408']} />
        <fog attach="fog" args={['#020408', 15, 35]} />
        
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 20, 5]} intensity={1.5} color="#FFD700" castShadow />
        <pointLight position={[-10, 10, -10]} intensity={2} color="#00F5FF" />

        <group ref={dieGroupRef}>
          <Float speed={assemblyProgress >= 1 ? 2 : 0} rotationIntensity={0.5} floatIntensity={1}>
            <SubstrateQuads assemblyProgress={assemblyProgress} />
            <AssemblyCore assemblyProgress={assemblyProgress} />
            <AssemblyTraces assemblyProgress={assemblyProgress} />
            
            {/* ALU Clusters assembling */}
            {Array.from({ length: 16 }).map((_, i) => (
              <FloatingALU key={i} index={i} assemblyProgress={assemblyProgress} />
            ))}
          </Float>
        </group>

        <OrbitControls enableZoom={false} enablePan={false} maxPolarAngle={Math.PI / 2.1} />
        
        {!reducedMotion && !isMobile && (
          <EffectComposer multisampling={0}>
            <Bloom luminanceThreshold={0.2} mipmapBlur intensity={assemblyProgress >= 1 ? 1.5 : 0.5} />
          </EffectComposer>
        )}
      </Canvas>
      
      {/* Background assembly status text overlay */}
      {assemblyProgress < 1 && (
        <div className="absolute bottom-12 left-1/2 -translate-x-1/2 z-30 flex flex-col items-center">
          <div className="w-64 h-1 bg-white/5 rounded-full overflow-hidden">
            <div 
              className="h-full bg-neon-cyan transition-all duration-300 ease-out" 
              style={{ width: `${assemblyProgress * 100}%` }}
            />
          </div>
          <p className="mt-4 font-mono text-[10px] text-neon-cyan uppercase tracking-[0.2em] animate-pulse">
            Initializing RTL Subsystems: {(assemblyProgress * 100).toFixed(0)}%
          </p>
        </div>
      )}

      <div className="absolute inset-0 grain-overlay z-10 pointer-events-none"></div>
    </div>
  );
}
