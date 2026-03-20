'use client';

import React, { useRef, useMemo, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Float, Box, Plane, Line } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import * as THREE from 'three';
import { useReducedMotion } from 'framer-motion';

// --- SUB-COMPONENTS --- //

const TransistorGrid = () => {
  const instances = useRef<THREE.InstancedMesh>(null);
  const gridSize = 30;
  
  useMemo(() => {
    if (!instances.current) return;
    const dummy = new THREE.Object3D();
    let i = 0;
    for (let x = -gridSize / 2; x < gridSize / 2; x++) {
      for (let z = -gridSize / 2; z < gridSize / 2; z++) {
        dummy.position.set(x * 0.2, 0.05, z * 0.2);
        // Add random slight variation to scale to look like complex logic gates
        dummy.scale.set(0.8, Math.random() * 0.5 + 0.1, 0.8);
        dummy.updateMatrix();
        instances.current.setMatrixAt(i++, dummy.matrix);
      }
    }
    instances.current.instanceMatrix.needsUpdate = true;
  }, []);

  return (
    <instancedMesh ref={instances} args={[undefined, undefined, gridSize * gridSize]}>
      <boxGeometry args={[0.1, 0.1, 0.1]} />
      <meshStandardMaterial color="#1a1f2e" metalness={0.8} roughness={0.2} />
    </instancedMesh>
  );
};

const BondPads = () => {
  const pads = [];
  const edge = 3.8;
  const count = 12;
  const step = (edge * 2) / count;
  
  for (let i = 0; i <= count; i++) {
    const pos = -edge + i * step;
    pads.push([-edge, 0.02, pos]); // Left Edge
    pads.push([edge, 0.02, pos]);  // Right Edge
    pads.push([pos, 0.02, -edge]); // Top Edge
    pads.push([pos, 0.02, edge]);  // Bottom Edge
  }

  return (
    <group>
      {pads.map((pos, i) => (
        <Box key={i} args={[0.2, 0.05, 0.2]} position={pos as [number, number, number]}>
          <meshStandardMaterial color="#B87333" metalness={1} roughness={0.1} />
        </Box>
      ))}
    </group>
  );
};

const Traces = () => {
  const lines = useMemo(() => {
    const l = [];
    for (let i = 0; i < 20; i++) {
      const points = [];
      let x = (Math.random() - 0.5) * 6;
      let z = (Math.random() - 0.5) * 6;
      points.push(new THREE.Vector3(x, 0.03, z));
      
      for (let j = 0; j < 3; j++) {
        // Orthogonal routing typical of silicon
        if (Math.random() > 0.5) x += (Math.random() - 0.5) * 2;
        else z += (Math.random() - 0.5) * 2;
        points.push(new THREE.Vector3(x, 0.03, z));
      }
      l.push(points);
    }
    return l;
  }, []);

  return (
    <group>
      {lines.map((pts, i) => (
        <Line key={i} points={pts} color="#FFD700" lineWidth={1} />
      ))}
    </group>
  );
};

const CopperParticles = ({ reducedMotion, count = 200 }: { reducedMotion: boolean, count?: number }) => {
  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 10;
      arr[i * 3 + 1] = Math.random() * 5;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 10;
    }
    return arr;
  }, [count]);

  const pointsRef = useRef<THREE.Points>(null);

  useFrame(() => {
    if (reducedMotion || !pointsRef.current) return;
    const positions = pointsRef.current.geometry.attributes.position.array as Float32Array;
    for (let i = 0; i < count; i++) {
      positions[i * 3 + 1] += 0.01; // float up
      if (positions[i * 3 + 1] > 5) {
        positions[i * 3 + 1] = 0;
      }
    }
    pointsRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial size={0.05} color="#00F5FF" transparent opacity={0.6} sizeAttenuation />
    </points>
  );
};

// Main Die Object with enhanced visual effects
const SiliconDie = ({ isHovered, reducedMotion }: { isHovered: boolean, reducedMotion: boolean }) => {
  const dieRef = useRef<THREE.Group>(null);
  const coreRef = useRef<THREE.Mesh>(null);
  
  useFrame((state, delta) => {
    if (reducedMotion || !dieRef.current) return;
    
    // Smooth adaptive rotation
    const targetSpeed = isHovered ? 0.05 : 0.3;
    dieRef.current.rotation.y += delta * targetSpeed;
    
    // Pulsing central compute core logic
    if (coreRef.current) {
      const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.1 + 0.25;
      (coreRef.current.material as THREE.MeshBasicMaterial).opacity = pulse;
      coreRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 3) * 0.015);
    }
  });

  return (
    <group ref={dieRef}>
      {/* Base Silicon Substrate - Ultra Metallic */}
      <Box args={[8, 0.15, 8]} position={[0, 0, 0]} castShadow receiveShadow>
        <meshStandardMaterial 
          color="#020408" 
          metalness={1} 
          roughness={0.1} 
          envMapIntensity={2}
        />
      </Box>

      {/* Surface Pattern (Transistors) */}
      <TransistorGrid />
      
      {/* Bond Pads - Copper Highlights */}
      <BondPads />
      
      {/* Circuit Traces - Golden Highways */}
      <Traces />

      {/* Active Compute Core Glow (Neon Cyan) */}
      <Plane ref={coreRef} args={[4.5, 4.5]} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
        <meshBasicMaterial 
          color="#00F5FF" 
          transparent 
          opacity={0.3} 
          blending={THREE.AdditiveBlending} 
        />
      </Plane>
      
      {/* Inner Processing Core (Plasma Violet) */}
      <Plane args={[2, 2]} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.06, 0]}>
        <meshBasicMaterial 
          color="#7C3AED" 
          transparent 
          opacity={0.5} 
          blending={THREE.AdditiveBlending} 
        />
      </Plane>
    </group>
  );
};

export default function SiliconDieCanvas() {
  const [isHovered, setIsHovered] = useState(false);
  const reducedMotion = useReducedMotion() ?? false;
  const [webGLAvailable, setWebGLAvailable] = useState(true);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      if (!gl) setWebGLAvailable(false);
    } catch {
      setWebGLAvailable(false);
    }
    
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  if (!webGLAvailable) {
    return (
      <div className="absolute inset-0 bg-silicon-black flex items-center justify-center">
        <div className="w-full h-full bg-[radial-gradient(ellipse_at_center,_var(--neon-cyan)_0%,_var(--silicon-black)_100%)] opacity-20"></div>
        <div className="absolute inset-0 grain-overlay"></div>
      </div>
    );
  }

  return (
    <div 
      className="absolute inset-0 z-0 bg-silicon-black overflow-hidden"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <Canvas 
        shadows 
        camera={{ position: isMobile ? [8, 6, 12] : [5, 4, 8], fov: 45 }}
        gl={{ antialias: false, alpha: false, powerPreference: 'high-performance' }}
        dpr={isMobile ? [1, 1] : [1, 1.5]}
      >
        <color attach="background" args={['#020408']} />
        <fog attach="fog" args={['#020408', 10, 20]} />
        
        <ambientLight intensity={0.4} />
        <directionalLight position={[10, 10, 5]} intensity={1.5} color="#FFD700" castShadow />
        <pointLight position={[-5, 5, -5]} intensity={2} color="#00F5FF" />
        <pointLight position={[0, 2, 0]} intensity={1} color="#7C3AED" />

        <Float speed={reducedMotion ? 0 : 2.5} rotationIntensity={0.8} floatIntensity={2}>
          <SiliconDie isHovered={isHovered} reducedMotion={reducedMotion} />
        </Float>

        <CopperParticles reducedMotion={reducedMotion} count={isMobile ? 50 : 200} />

        <OrbitControls 
          enableZoom={false} 
          enablePan={false} 
          maxPolarAngle={Math.PI / 2.2} 
          minDistance={3} 
          maxDistance={20}
        />
        
        {!reducedMotion && !isMobile && (
          <EffectComposer multisampling={0}>
            <Bloom luminanceThreshold={0.2} mipmapBlur intensity={1.5} />
          </EffectComposer>
        )}
      </Canvas>
      <div className="absolute inset-0 grain-overlay z-10 pointer-events-none"></div>
    </div>
  );
}
