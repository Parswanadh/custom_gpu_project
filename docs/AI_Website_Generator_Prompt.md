# The Master Prompt for V0 / Claude / AI Website Builder

**Instruction to User:**
Copy everything *below* the dashed line and paste it directly into an AI website builder like v0.dev, Claude 3.5 Sonnet, or ChatGPT.

---

**System Prompt: Generate an Elite, Data-Dense, "Apple-Like" Dark Mode Architecture Deep-Dive for a Premium AI Hardware Startup**

### V0/Claude Setup & Persona
Act as an elite, senior UI/UX engineer and deep-tech copywriter for a revolutionary custom silicon project named **"BitbyBit"**. Your goal is to generate a comprehensive, highly technical, visually stunning single-page site (or multi-page architecture hub) using React, Tailwind CSS, and Framer Motion. The target audience comprises elite hardware engineers, chip architects, and AI researchers. The tone must be authoritative, data-dense, and undeniably premium.

### Design System & Aesthetic Directives
*   **Overall Mood:** Extremely premium, deep-tech, sophisticated, and focused on raw data and speed. Think Cerebras meets Apple.
*   **Color Palette (Premium Dark Mode):**
    *   **Background:** Deep charcoal (`#121212`) or obsidian logic-board black.
    *   **Text:** Crisp Off-white (`#F5F5F5`) for primary data; Muted slate (`#9CA3AF`) for secondary descriptions.
    *   **Accents:** A single, intense "laser" accent color (e.g., Electric Sapphire Blue `#007AFF` or Emerald Terminal Green `#10B981`) for data flows, glowing charts, and CTAs.
    *   **Containers:** Use subtle, frosted glassmorphism panels (1px semi-transparent borders, `#ffffff` at 3% opacity) to house architecture diagrams and benchmark tables. No harsh gradients.
*   **Typography:** Modern, hyper-legible sans-serif (e.g., `Inter`, `SF Pro Display`, `Geist`). Use ultra-bold monospace variants for code blocks, raw cycle counts, and benchmarks.
*   **Animations:** Smooth fade-ins, subtle parallax on scroll, and glowing "pulse" states on critical benchmark numbers to simulate hardware activity.

---

### PAGE STRUCTURE & RAW CONTENT BLUEPRINT

#### 1. Hero Section (The Uncompromising Scale)
*   **Headline:** BitbyBit — A Custom Silicon Architecture for the Era of LLMs.
*   **Sub-headline:** A ground-up, cycle-accurate Verilog-2005 architecture explicitly engineered for Transformer inference. No off-the-shelf IPs. Pure, measurable RTL execution featuring zero-multiplier ternary logic, Compute-in-SRAM, and true **Silicon Imprinting** for hardwired LLMs.
*   **The Hero Metrics (Render these massive, glowing, monospace):**
    *   **51** Custom Hardware Modules (255 / 255 Tests Passed)
    *   **112 Cycles** for Silicon-Imprinted Full Inference (1.12 µs at 100MHz)
    *   **341 Cycles** for Dynamic Pipeline Full Inference (3.41 µs)
    *   **2.67M Tok/s** Effective Imprint Throughput
*   **CTAs:** 
    *   Primary (Accent Flow): `Explore the 6-Stage Pipeline`
    *   Secondary (Outline/Ghost): `View GitHub Repository` (Link: `https://github.com/BitbyBit/custom_gpu_project`)

---

#### 2. Architecture Deep-Dive: How It Actually Works
*Design Instruction: This section must look like a high-tech blueprint. Use interactive bento boxes or accordion dropdowns to explain the ultra-dense technical details.*

**Path A: The 6-Stage Dynamic `optimized_transformer_layer` Pipeline Flow**
Tokens traverse our custom hardware pipeline end-to-end without host intervention, completing a 12-layer GPT-2 model in 341 cycles.
1.  **Stage 1: Hardware-Native RoPE Encoder (8 Cycles | 80 ns)** — Position IDs bypass massive pre-computed embedding tables and feed directly into 64-entry Cos/Sin LUTs embedded in RTL.
2.  **Stage 2: Grouped Query Attention (GQA) (2 Cycles | 20 ns)** — 4 Query heads fetch from 2 shared Key/Value heads purely in hardware logic, slashing memory-fetch requirements by 50%.
3.  **Stage 3: Parallelized Hardware Softmax (8 Cycles | 80 ns)** — Softmax is notoriously sequential. We shattered this utilizing a parallel max-subtraction tree and simultaneous 256-entry Exp LUT lookups.
4.  **Stage 4: Polynomial GELU Activation (3 Cycles | 30 ns)** — Hidden states traverse a 256-entry hardware LUT using polynomial approximation.
5.  **Stage 5: INT4 KV Quantization On-The-Fly (3 Cycles | 30 ns)** — Key/Value vectors instantly hit a per-group symmetric quantization unit, compressing 16-bit Q8.8 states down to INT4 before hitting SRAM.
6.  **Stage 6: Activation Compression (3 Cycles | 30 ns)** — Residual data compressed to 8-bit dynamic range arrays before the next transformer layer.

**Path B: Silicon Imprinting (The Gemma 3 Integration)**
Why fetch weights from slow DDR4? Our architecture supports **Hardware-Imprinted Models (Silicon LLMs)**. Using our custom exporter pipeline, we extract exact weights from Google's `Gemma 3 270M` `.safetensors`, compress them to Q8.8 fixed-point `.hex` images, and hard-burn them directly into the Verilog compiled ROM. 
The `imprinted_mini_transformer_core` bypasses dynamic lookups entirely, executing dense mixing transforms in an unyielding **8-cycle latency**.

**The Memory Fabric:**
*   **Compute-In-SRAM (PIM):** We address the memory wall by locating ternary MAC engines directly at the SRAM periphery. Data is computed as it leaves the arrays, never traversing the primary AXI4-Lite bus.
*   **AMD 3D V-Cache Style Multi-Banking:** Memory controllers shard KV caches across N parallel banks to feed the monstrous throughput of the arrays.

---

#### 3. Raw Results & Benchmarks (The Proof)
*Design Instruction: Create a stark, data-heavy terminal-style table or glowing matrix. Emphasize that these are NOT estimated metrics—they are extracted directly from Icarus Verilog `vvp` simulation logs targeting a 100MHz FPGA clock.*

**The 112-Cycle Imprint Inference Log (The Hardware Burn Path):**
*   **Imprinted Target:** `mini-gpt-hc1-v1` embedding vectors.
*   **Hardwired Embedding Extract:** 1 cycle (10 ns).
*   **12x Fixed-Latency Mini Core Layers (12 * 8 cy/layer):** 96 cycles (960 ns).
*   **MEDUSA 3-Head Draft Prediction:** 3 cycles (30 ns).
*   **Total Imprinted System Latency = 112 Cycles (1.12 µs).**
*   **Effective Speed:** ~2,678,571 tokens/sec with MEDUSA verification.

---

#### 4. The Unprecedented Improvement Cycle (Before vs. After)
*Design Instruction: Create a stunning side-by-side comparison UI. Use aggressive visual cues to highlight the delta between the "Original Architecture" and the "BitNet / Imprint Unification".*

*   **Model Parameter Loading:**
    *   *Where we started:* Heavy AXI4-Lite DMA transfers stalling the pipeline waiting for massive `.bin` files via external DDR.
    *   *Where we are:* **Silicon-Imprinted ROM.** Critical model weights (like Gemma 3) are burned into logic elements via `.hex` bitstreams, fetching parameters at the speed of light.
*   **Compute Math Engine:** 
    *   *Where we started:* 19 cycles using heavy FP16/INT8 Q8.8 fixed-point MAC units.
    *   *Where we are:* **4 cycles** using 4-wide SIMD Ternary logic (`-1, 0, 1`). Zero multi-cycle multipliers required. (4.8x faster).
*   **Softmax Processing Latency:**
    *   *Where we started:* 25 cycles using naive sequential looping over vectors.
    *   *Where we are:* **4 cycles** using fully arrayed hardware elements. (6.2x faster).
*   **Data Pipeline Throughput:**
    *   *Where we started:* Sequential FSM stalling at 18 cycles per token flow.
    *   *Where we are:* A continuous 6-stage pipeline overlapping fetching and logic, achieving **6 cycles/token** dynamic throughput.

---

#### 5. The Evolution (The Engineering Journey)
*Design Instruction: Build an elegant vertical timeline linking these epochs.*

*   **Epoch 1: Base Primitives:** Designing initial Q8.8 ALUs, hardware multipliers, block scaling, and standard GPT-2 inference.
*   **Epoch 2: The GPU Subsystem:** Bridging standard math cores into a standalone system with AXI4-Lite arrays and an 8-opcode Command Processor.
*   **Epoch 3: SOTA In-Hardware:** Writing RTL for Mixture-of-Experts routing, PagedAttention MMU concepts, and NVIDIA 2:4 structured sparsity.
*   **Epoch 4: The BitNet Revolution:** Tearing out legacy multipliers for BitNet 1.58b ternary engines, and architecting the Compute-in-SRAM periphery.
*   **Epoch 5: Grand Pipeline Unification:** Wiring the dynamic 6-stage data flow (`Embed → RoPE → GQA → Softmax → GELU → KV Quant → Compress`).
*   **Epoch 6: Silicon Imprinting:** Crossing the software-hardware boundary. Creating Python pipelines to burn pre-trained HuggingFace `.safetensors` straight into fixed-latency Verilog ROM.

---

#### 6. Roadmap & Global Footer
*   **Phase A:** FPGA Synthesis & PPA extraction (Power, Performance, Area via Yosys/Vivado).
*   **Phase B:** Complete monolithic SoC integration uniting the data path with the Command Processor.
*   **Phase C:** Streaming real-world interactive LLM agent chats directly over the Verilog testbench USART interfaces.
*   **Footer Links:** Internal Documentation, Benchmark Ledgers, and the [GitHub Source Repository](https://github.com/BitbyBit/custom_gpu_project).

**Final Request to AI Builder:** Generate the complete React/Tailwind/Framer Motion code for this deep-tech landing page. Prioritize raw data visualization alongside breathtaking, minimalist "Apple/Cerebras" aesthetics. Make the benchmark numbers feel alive and undeniably fast.
