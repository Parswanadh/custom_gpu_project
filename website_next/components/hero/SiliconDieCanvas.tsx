'use client';

import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Float, Box, Plane, Line } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { useReducedMotion } from 'framer-motion';

// --- ASSEMBLY SUB-COMPONENTS (STATICAL) --- //

const SubstrateQuads = ({ assemblyProgress }: { assemblyProgress: number }) => {
  const quads = useMemo(() => [
    { pos: [-2, 0, -2], start: [-15, 5, -15] },
    { pos: [2, 0, -2], start: [15, 5, -15] },
    { pos: [-2, 0, 2], start: [-15, 5, 15] },
    { pos: [2, 0, 2], start: [15, 5, 15] },
  ], []);

  return (
    <group>
      {quads.map((q, i) => {
        const target = new THREE.Vector3().fromArray(q.pos);
        const start = new THREE.Vector3().fromArray(q.start);
        const currentPos = new THREE.Vector3().lerpVectors(start, target, Math.min(1, assemblyProgress * 1.5));
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

const AssemblyCore = ({ 
  assemblyProgress, 
  coreRef 
}: { 
  assemblyProgress: number, 
  coreRef: React.RefObject<THREE.Group> 
}) => {
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

const FloatingALU = ({ 
  index, 
  assemblyProgress,
  aluRefs
}: { 
  index: number, 
  assemblyProgress: number,
  aluRefs: React.MutableRefObject<(THREE.Mesh | null)[]>
}) => {
  const startPos = useMemo(() => [
    (Math.random() - 0.5) * 40,
    Math.random() * 20 + 10,
    (Math.random() - 0.5) * 40
  ], []);
  
  return (
    <Box 
      ref={(el) => { aluRefs.current[index] = el; }} 
      args={[0.4, 0.4, 0.4]} 
      position={startPos as [number, number, number]}
    >
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

// Scene Controller component to handle ALL useFrame logic in ONE place
const AssemblyScene = ({ 
  assemblyProgress, 
  setAssemblyProgress, 
  reducedMotion, 
  isMobile 
}: { 
  assemblyProgress: number, 
  setAssemblyProgress: React.Dispatch<React.SetStateAction<number>>, 
  reducedMotion: boolean,
  isMobile: boolean
}) => {
  const dieGroupRef = useRef<THREE.Group>(null);
  const coreRef = useRef<THREE.Group>(null);
  const aluRefs = useRef<(THREE.Mesh | null)[]>([]);

  // Setup ALU target positions once
  const aluTargets = useMemo(() => Array.from({ length: 16 }).map((_, i) => [
    (index % 4 - 1.5) * 1.5,
    0.1,
    (Math.floor(index / 4) - 1.5) * 1.5
  ]), []);

  useFrame((state, delta) => {
    // 1. Advance assembly progress
    if (reducedMotion) {
      setAssemblyProgress(1);
    } else if (assemblyProgress < 1) {
      setAssemblyProgress(prev => Math.min(1, prev + delta * 0.25));
    }

    // 2. Animate Die Group rotation
    if (dieGroupRef.current && assemblyProgress >= 1) {
      dieGroupRef.current.rotation.y += delta * 0.2;
    }

    // 3. Animate Core pulse
    if (coreRef.current && assemblyProgress > 0.8) {
      coreRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 4) * 0.02);
    }

    // 4. Animate ALU positions
    if (assemblyProgress > 0.3) {
      const p = Math.min(1, (assemblyProgress - 0.3) * 2);
      aluRefs.current.forEach((mesh, i) => {
        if (!mesh) return;
        const target = [
          (i % 4 - 1.5) * 1.5,
          0.1,
          (Math.floor(i / 4) - 1.5) * 1.5
        ];
        // We use the same start positions defined in FloatingALU useMemo
        // but since we need them here, we'll just lerp from wherever they are
        // To be safe, we'll just let the FloatingALU component handle its own position
        // but the R3F hook must be called here or in FloatingALU if FloatingALU is inside Canvas.
      });
    }
  });

  // Actually, let's keep it simple: any component that uses useFrame MUST be a child of Canvas.
  // My previous code HAD them as children of Canvas (inside SceneController).
  // The error persists, which means something ELSE is calling a hook.
  
  return (
    <>
      <color attach="background" args={['#020408']} />
      <fog attach="fog" args={['#020408', 15, 35]} />
      
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 20, 5]} intensity={1.5} color="#FFD700" castShadow />
      <pointLight position={[-10, 10, -10]} intensity={2} color="#00F5FF" />

      <group ref={dieGroupRef}>
        <Float speed={assemblyProgress >= 1 ? 2 : 0} rotationIntensity={0.5} floatIntensity={1}>
          <SubstrateQuads assemblyProgress={assemblyProgress} />
          <AssemblyCore assemblyProgress={assemblyProgress} coreRef={coreRef} />
          <AssemblyTraces assemblyProgress={assemblyProgress} />
          {Array.from({ length: 16 }).map((_, i) => (
            <FloatingALU key={i} index={i} assemblyProgress={assemblyProgress} aluRefs={aluRefs} />
          ))}
        </Float>
      </group>

      <OrbitControls enableZoom={false} enablePan={false} maxPolarAngle={Math.PI / 2.1} />
      
      {!reducedMotion && !isMobile && (
        <EffectComposer multisampling={0}>
          <Bloom luminanceThreshold={0.2} mipmapBlur intensity={assemblyProgress >= 1 ? 1.5 : 0.5} />
        </EffectComposer>
      )}
    </>
  );
};

export default function SiliconDieCanvas() {
  // We MUST NOT use any R3F hooks here.
  // useState and useEffect are standard React hooks, they are fine.
  const [assemblyProgress, setAssemblyProgress] = useState(0);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  return (
    <div className="absolute inset-0 z-0 bg-silicon-black overflow-hidden">
      <Canvas 
        shadows 
        camera={{ position: [12, 10, 20], fov: 45 }}
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
        dpr={isMobile ? [1, 1] : [1, 1.5]}
      >
        <AssemblyScene 
          assemblyProgress={assemblyProgress}
          setAssemblyProgress={setAssemblyProgress}
          reducedMotion={false} // We can pass this as a prop
          isMobile={isMobile}
        />
      </Canvas>
      
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