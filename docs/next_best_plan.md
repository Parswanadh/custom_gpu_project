# The Next Best Plan: Gemma-3 Hardware & UI Integration Roadmap

Based on the multi-agent Swarm Analysis and Alignment Debate, the following is the optimal, sequenced execution plan to advance the Tri-Fold Prototype without sacrificing the SOTA 2.67M Tokens/sec throughput.

## Track 1: High-Performance Memory Subsystem (Hardware Foundation)
*Before adding complex Gemma-3 math blocks, the memory subsystem must be upgraded to handle parallel data access.*
- **Task:** Implement Dual-Ported Block RAMs for Sine/Cosine LUTs to eliminate structural hazards during RoPE calculation.
- **Task:** Introduce wide-bus asynchronous FIFOs (skid buffers) at the boundaries of the `gate` and `up` projection pipelines to prevent MLP resource contention.

## Track 2: Gemma-3 Verilog Block Implementation
*With the memory foundation secure, build the Gemma-3 specific math units using Q8.8 fixed-point precision.*
- **Task:** Implement `rmsnorm.v` using a heavily pipelined reduction tree for sum-of-squares.
- **Task:** Implement `rope_unit.v` utilizing the newly created Dual-Ported LUTs for complex rotations.
- **Task:** Implement `gated_mlp.v` featuring parallel `gate` (with `gelu_pytorch_tanh` LUT) and `up` linear projections.

## Track 3: Full-Stack Integration & Visualization
*Wire the simulated hardware metrics of the new blocks into the Next.js dashboard.*
- **Task:** Create Three.js meshes: `<RMSNormNode>`, `<RoPENode>`, and `<GatedMLPNode>` inside `GemmaExecutionFlow`.
- **Task:** Drill the `useSimulatedMetrics` props down to these components.
- **Task:** Animate the nodes (e.g., pulsing emissive materials, particle flow speed) based on the simulated `throughput` and `bottleneck` states.