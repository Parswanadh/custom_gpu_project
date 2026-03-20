# BitbyBit Custom Silicon — Website Build Reference
> **For AI Models:** This is a complete, exhaustive handoff document for the Next.js landing page located at `d:\Projects\BitbyBit\custom_gpu_project\website_next`. Read every section before writing any code.

---

## 1. Project Overview

**Project:** BitbyBit Custom GPU — a ground-up, cycle-accurate Verilog-2005 architecture for Transformer inference.  
**Website Goal:** A premium "Apple-meets-Deep-Tech" landing page that explains the hardware to both engineers and investors in a stunning, interactive format.  
**Live Deployment:** `https://bitbybit-sandy.vercel.app`  
**Local Dev Path:** `d:\Projects\BitbyBit\custom_gpu_project\website_next`  
**Framework:** Next.js 14 (App Router)  
**Language:** TypeScript (strict mode)  
**Styling:** Tailwind CSS v3  
**3D:** Three.js via `@react-three/fiber` + `@react-three/drei`  
**Animation:** Framer Motion v12  

---

## 2. Tech Stack & All Dependencies

### Runtime Dependencies (`package.json`)
| Package | Version | Purpose |
|---|---|---|
| `next` | 14.2.15 | App Router SSR framework |
| `react` + `react-dom` | ^18 | UI library |
| `framer-motion` | ^12.38.0 | Scroll animations, presence, hover effects |
| `three` | ^0.163.0 | WebGL 3D engine |
| `@react-three/fiber` | ^8.16.2 | React declarative Three.js renderer |
| `@react-three/drei` | ^9.105.6 | Three.js helpers (OrbitControls, Float, Text, Line, etc.) |
| `@react-three/postprocessing` | ^2.16.2 | Bloom, post-processing passes |
| `recharts` | ^2.12.7 | SVG charts for metrics section |
| `lucide-react` | ^0.577.0 | Icon set |
| `tailwind-merge` | ^3.5.0 | Utilities merging |
| `tailwindcss-animate` | ^1.0.7 | CSS keyframe animation plugin |
| `tw-animate-css` | ^1.4.0 | Additional animations |
| `shadcn` | ^4.0.8 | Headless UI primitives |
| `clsx` | ^2.1.1 | Conditional className utility |
| `lottie-react` | ^2.4.1 | Lottie animations (currently unused) |
| `react-countup` | ^6.5.3 | (Installed but custom `CountUp.tsx` is used instead) |

### Dev Dependencies
| Package | Purpose |
|---|---|
| `typescript ^5` | Strict type checking |
| `tailwindcss ^3.4.1` | CSS framework |
| `vitest ^1.5.0` | Unit testing |
| `@playwright/test ^1.43.1` | E2E testing |
| `@testing-library/react` | Component testing |

---

## 3. Design System

### Color Palette (Tailwind tokens & CSS variables)
All colors are defined in both `tailwind.config.ts` and `app/globals.css`:

| Token | Hex | Tailwind Class | Use Case |
|---|---|---|---|
| Silicon Black | `#020408` | `bg-silicon-black` | Primary background, die substrate |
| Silicon Gray | `#1A1F2E` | `bg-silicon-gray` | Card backgrounds |
| Neon Cyan | `#00F5FF` | `text-neon-cyan`, `border-neon-cyan` | Primary accent, BitbyBit highlight, glow |
| Die Copper | `#B87333` | `text-die-copper` | Secondary accent (pipeline, roadmap sections) |
| Trace Gold | `#FFD700` | `text-trace-gold` | Circuit trace highlights |
| Plasma Violet | `#7C3AED` | `text-plasma-violet` | Architecture section accent |
| Oxide Green | `#00FF88` | `text-oxide-green` | Success/complete states |
| Error Red | `#FF3366` | `text-error-red` | Error states, contrast elements |
| Metal Silver | `#8892A4` | `text-metal-silver` | Body copy, muted text |

### Typography (Google Fonts via `next/font`)
| Font | CSS Variable | Tailwind Class | Use |
|---|---|---|---|
| Space Grotesk | `--font-space-grotesk` | `font-space` | Headlines, brand name |
| Inter | `--font-inter` | `font-inter` | Body copy, descriptions |
| JetBrains Mono | `--font-jetbrains-mono` | `font-mono` | Technical data, code strings |
| DM Sans | `--font-dm-sans` | `font-dm` | UI labels, badges |

### Global CSS Utilities (`app/globals.css`)
| Class | Description |
|---|---|
| `.grain-overlay` | Fixed film grain overlay, `opacity: 0.03`, `mix-blend-mode: overlay` |
| `.scanline` | Animated neon cyan scan line traversing hero section |
| `.glass-surface` | Glassmorphic card: `backdrop-blur-24px`, `rgba(26,31,46,0.6)` bg |
| `.glow-on-hover` | `box-shadow: 0 0 20px var(--neon-cyan)` on hover |
| `.typing-cursor` | Blinking neon-cyan cursor for terminal effects |

---

## 4. Repository Structure

```
website_next/
├── app/
│   ├── layout.tsx          # Root layout: fonts, metadata, body class
│   ├── page.tsx            # Main page: assembles all 8 sections
│   └── globals.css         # Global tokens, utilities, keyframes
│
├── components/
│   ├── navigation/
│   │   └── NavigationBar.tsx       # [CLIENT] Fixed top nav with logo + anchor links
│   ├── hero/
│   │   ├── SiliconDieCanvas.tsx    # [CLIENT] Three.js GPU die, the CENTERPIECE visual
│   │   ├── GlitchHeadline.tsx      # [CLIENT] Scramble-to-reveal headline animation
│   │   └── HeroStats.tsx           # [CLIENT] 4 animated stat counters (CountUp-powered)
│   ├── architecture/
│   │   └── 3DSystemGraph.tsx       # [CLIENT] Orbital 3D graph with data packets
│   ├── pipeline/
│   │   └── PipelineDAG.tsx         # [CLIENT] Interactive 6-stage hardware pipeline DAG
│   ├── metrics/
│   │   └── RunsChart.tsx           # [CLIENT] Recharts area chart (sim log data)
│   ├── shared/
│   │   ├── CountUp.tsx             # [CLIENT] IntersectionObserver + framer animate() counter
│   │   └── WorkflowBackground.tsx  # [CLIENT] Animated SVG circuit trace background
│   ├── sections/                   # [CLIENT] Page section layout wrappers
│   │   ├── HeroSection.tsx
│   │   ├── ArchitectureSection.tsx
│   │   ├── PipelineSection.tsx
│   │   ├── MetricsDashboardSection.tsx
│   │   ├── ComparisonSection.tsx
│   │   ├── RoadmapSection.tsx
│   │   └── FooterSection.tsx
│   └── ui/                        # Shadcn UI primitives
│       ├── badge.tsx
│       ├── button.tsx
│       └── card.tsx
│
├── tailwind.config.ts      # Design tokens, font family, color palette
├── next.config.mjs         # Next.js configuration (default minimal)
├── tsconfig.json           # TypeScript strict mode
└── vitest.config.ts        # Unit test setup
```

---

## 5. Page Sections — Full Specification

The page composition in `app/page.tsx` renders 8 sections in this order:

### 5.1 `<NavigationBar />`
- **File:** `components/navigation/NavigationBar.tsx`
- **Type:** Fixed top, `z-50`, glassmorphic backdrop
- **Logo:** `BITBYBIT` + `Cpu` icon from lucide-react
- **Links:** Architecture, Raw Results, Evolution, Comparison, Roadmap
- **Behavior:** Hides/reveals on scroll (Framer Motion)

### 5.2 `<HeroSection />`
- **File:** `components/sections/HeroSection.tsx`
- **Height:** `100vh`, `min-h[800px]`
- **Background:** `SiliconDieCanvas` (WebGL, dynamically imported `ssr: false`)
- **Headline:** `GlitchHeadline text="BitbyBit Custom Silicon"` — scrambles random chars then resolves to real text character-by-character
- **Subtext:** Verilog-2005, zero-multiplier ternary logic, Compute-in-SRAM, Silicon Imprinting
- **Stats (`HeroStats`):** 4 CountUp cards:
  - 51 Hardware Modules
  - 112 Imprint Latency (cycles)
  - 341 Dynamic Latency (cycles)
  - 2M Tok/s Effective Throughput

### 5.3 `<ArchitectureSection />`
- **File:** `components/sections/ArchitectureSection.tsx`
- **Background:** `WorkflowBackground color="violet" density="sparse"`
- **Headline:** "Hard-Burned Into Silicon."
- **Description:** Explains silicon imprinting — extracting Gemma 3 270M weights, Q8.8 fixed-point compression, burning into Verilog ROM for 8-cycle latency
- **3D Visual:** `ThreeDSystemGraph` (dynamically imported `ssr: false`) — an orbital Three.js scene with 7 glowing nodes and data packets traversing edges

### 5.4 `<PipelineSection />`
- **File:** `components/sections/PipelineSection.tsx`
- **Background:** `WorkflowBackground color="copper" density="medium"`
- **Layout:** 12-column grid: text left (5 cols) + DAG visualization right (7 cols)
- **Headline:** "Hardware Transformer Pipeline"
- **Key stats:** 341-cycle 12-layer inference, hardware-native RoPE & GQA, Parallelized Softmax & Poly GELU, INT4 KV Quantization
- **Interactive Visual:** `PipelineDAG` — 6 animated nodes:
  1. Embed (Copper)
  2. RoPE Encoder (Cyan)
  3. GQA Matrix (Violet)
  4. Softmax (Green)
  5. GELU (Gold)
  6. KV Quant (Red)

### 5.5 `<MetricsDashboardSection />`
- **File:** `components/sections/MetricsDashboardSection.tsx`
- **Background:** `WorkflowBackground color="copper" density="medium"`
- **Data Cards (4, CountUp animated on scroll-in):**
  - Layer Latency: `96 cy`
  - Embedding Extract: `10 ns`
  - Effective Clock: `100 MHz`
  - Latency: `1.12 µs`
- **Chart:** `RunsChart` — Recharts area chart labelled "Icarus Verilog (vvp) Simulation Log" with `112-CYCLE IMPRINT` and `341-CYCLE DYNAMIC` series

### 5.6 `<ComparisonSection />`
- **File:** `components/sections/ComparisonSection.tsx`
- **Background:** `WorkflowBackground color="cyan" density="sparse"`
- **Headline:** "Before vs After." 
- **Table (4 rows, 4 columns):**
  - Columns: Subsystem | Legacy IP | Original RTL | **BitbyBit**
  - Row 1: Model Parameter Loading → Cloud API → AXI4-Lite DDR Stalls → **Silicon-Imprinted ROM**
  - Row 2: Compute Math Engine → FP32 Clusters → 19-cycle FP16 MACs → **4-cycle SIMD Ternary**
  - Row 3: Softmax Latency → TFLOPS GPU → 25-cycle Naive Loop → **4-cycle Arrayed HW**
  - Row 4: Pipeline Throughput → Batch Processing → 18-cycle FSM Stalls → **6-cycle/token Flow**
- **Interactive:** Hover on any row reveals a tooltip with engineering explanation

### 5.7 `<RoadmapSection />`
- **File:** `components/sections/RoadmapSection.tsx`
- **Background:** `WorkflowBackground color="copper" density="medium"`
- **Headline:** "The Evolution"
- **6 Epoch Cards (grid, staggered reveal):**
  - Epoch 1 — Base Primitives (complete ✓)
  - Epoch 2 — The GPU Subsystem (complete ✓)
  - Epoch 3 — SOTA In-Hardware (complete ✓)
  - Epoch 4 — The BitNet Revolution (complete ✓)
  - Epoch 5 — Pipeline Unification (complete ✓)
  - Epoch 6 — Silicon Imprinting (active, pulsing copper dot)

### 5.8 `<FooterSection />`
- **File:** `components/sections/FooterSection.tsx`
- **Logo:** `BITBYBIT` + Terminal icon
- **Copyright:** `© 2026 BitbyBit Custom Silicon. Cycle-Accurate Execution.`
- **Links:** Platform (Architecture, Pricing, Docs), Company (About, Careers, Contact), Socials (GitHub, Twitter, LinkedIn)

---

## 6. Key Component Deep-Dives

### 6.1 `SiliconDieCanvas.tsx` — THE HERO CENTERPIECE
This is the most critical visual. It is a fullscreen `absolute inset-0` WebGL canvas rendering a custom GPU die.

**Architecture:**
```
SiliconDieCanvas (exported, dynamically imported)
├── Canvas (r3f) — gl={{ antialias: false, alpha: false, powerPreference: 'high-performance' }}, dpr={[1, 1.5]}
│   ├── color attach="background" — #020408
│   ├── fog — #020408, near 10, far 20
│   ├── Lights — ambientLight, directionalLight (gold), 2x pointLights (cyan, violet)
│   ├── Float (speed=2.5, rotationIntensity=0.8, floatIntensity=2) [IMPORTANT: aggressive bobbing]
│   │   └── SiliconDie (isHovered, reducedMotion)
│   │       ├── Box 8x0.1x8 — Silicon substrate (#020408)
│   │       ├── TransistorGrid — 30x30 instanced tiny boxes, metallic
│   │       ├── BondPads — copper (#B87333) pads around die perimeter
│   │       ├── Traces — 20 orthogonally-routed gold (#FFD700) lines
│   │       └── Plane 4x4 — cyan heat zone (#00F5FF) additive blending
│   ├── CopperParticles — 200 floating cyan dots rising upward
│   ├── OrbitControls — enableZoom=FALSE (critical!), enablePan=false, drag-to-rotate only
│   └── EffectComposer — Bloom (luminanceThreshold=0.2, mipmapBlur, intensity=1.5)
└── grain-overlay div (CSS, pointer-events: none)
```

**Critical Rules:**
- `enableZoom={false}` is MANDATORY — zoom=true traps the mousewheel inside the canvas and breaks page scrolling
- The die rotates at `delta * 0.3` (fast) normally, `delta * 0.05` (slow) when hovered
- WebGL context uses `alpha: false` — background is not transparent at canvas level, instead set via `<color attach="background">`
- `dpr={[1, 1.5]}` prevents retina displays from rendering 4x pixels and lagging

### 6.2 `GlitchHeadline.tsx`
Scrambles through random chars from `'!<>-_\\/[]{}—=+*^?#_'` and progressively reveals the real text character-by-character, 3 frames of scramble per character at 40ms intervals.

**Fix applied:** When the interval clears (`iteration >= text.length`), `setDisplayText(text)` is explicitly called to guarantee the full resolved string is shown. This prevents stale state where the last few characters stay as random glyphs.

### 6.3 `CountUp.tsx`
- Uses `useInView` from framer-motion (with `margin: "-50px"`, `once: true`)
- Triggers `animate(0, end, { duration, ease: "easeOut", onUpdate })` from framer-motion
- **Critical:** Do NOT use `requestAnimationFrame` directly — it races against React's strict mode double-invoke and produces stale closures. Use `framer-motion`'s `animate()` function instead.

### 6.4 `WorkflowBackground.tsx`
- Renders animated SVG circuit traces on a `0 0 100 100` viewBox (unitless, NOT percentage)
- Uses `useMemo` to generate paths deterministically
- Paths mutate `opacity: [0.1, 0.4, 0.1]` on a looping Framer Motion animation
- **Critical Fix:** Path `d` attribute values MUST be unitless numbers (matching the viewBox=`0 0 100 100`) — NOT percentage strings like `50%`. Writing `50%` causes SVG parser errors and invisible lines.

### 6.5 `3DSystemGraph.tsx`
- 7 nodes (Next.js Client, API Gateway, Orchestrator, 3x Agents, Vector DB)
- Nodes alternate Box and Sphere geometries, connected by dashed Line elements
- `DataPacket` spheres travel along edges using `lerpVectors`
- Wrapped in `<Suspense fallback={null}>` to avoid WebGL freezing while async Three.js `Text` fonts load
- Canvas uses `gl={{ alpha: true }}` so the container background shows through

---

## 7. SSR / Client Component Architecture

**Critical Next.js 14 Rule:** Heavy WebGL/3D components MUST be dynamically imported with `ssr: false` to prevent server-side rendering crashes and bundle bloat.

| Component | Import Strategy | Reason |
|---|---|---|
| `SiliconDieCanvas` | `next/dynamic` + `ssr: false` | Three.js requires `window`, `WebGLRenderingContext` |
| `ThreeDSystemGraph` | `next/dynamic` + `ssr: false` | Same — Three.js is browser-only |
| `PipelineDAG` | Static import (client component) | Pure React/SVG, no WebGL |
| `WorkflowBackground` | Static import (client component) | Pure SVG/Framer Motion |
| `CountUp` | Static import (client component) | Requires `IntersectionObserver` |
| All `sections/*` | Static import | `'use client'` directive at top |

**Bundle Stats (last clean build):**
```
Route (app)           Size     First Load JS
/ (main page)         159 kB    246 kB total
Shared chunks                   87.6 kB
```

---

## 8. Animation System

All animations use **Framer Motion**. Common patterns:

### Scroll-Triggered Reveal (used on every section)
```tsx
<motion.div
  initial={{ opacity: 0, y: 30 }}
  whileInView={{ opacity: 1, y: 0 }}
  viewport={{ once: true }}
  transition={{ duration: 0.6 }}
>
```

### Staggered Children (used in grids/lists)
```tsx
transition={{ duration: 0.5, delay: i * 0.15 }}
```

### Tooltip AnimatePresence (used in ComparisonSection)
```tsx
<AnimatePresence>
  {hovered && (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
    />
  )}
</AnimatePresence>
```

### Reduced Motion
All animation components check `useReducedMotion()` from framer-motion. When true, animations are disabled or replaced with static states. The global CSS `@media (prefers-reduced-motion: reduce)` also disables all transitions.

---

## 9. Known Issues & Bugs (as of March 2026)

| Issue | Severity | Root Cause | Status |
|---|---|---|---|
| Page scrolls inside Die canvas | **CRITICAL** | `<OrbitControls enableZoom={true}>` traps wheel events | **FIXED** — `enableZoom={false}` |
| Architecture section black box | HIGH | Three.js `Text` font fetch has no Suspense boundary | **FIXED** — `<Suspense fallback={null}>` wrapping |
| CountUp shows 0 permanently | HIGH | Manual RAF loop stales in React Strict Mode | **FIXED** — replaced with framer `animate()` |
| SVG traces invisible | MEDIUM | `d` attribute uses `%` strings inside unit-less viewBox | **FIXED** — `viewBox="0 0 100 100"`, unitless coords |
| Footer read "AUTO-GIT" | LOW | Leftover from Auto-GIT project origin | **FIXED** — now reads "BITBYBIT" |
| Die animation too slow | LOW | `rotation.y += delta * 0.1`, float intensity 0.5 | **FIXED** — 0.3 spin, Float intensity 2 |
| Die animation pauses on hover | LOW | `if (isHovered) return` exited the frame | **FIXED** — now slows to 0.05 on hover |
| 3005 port 500 error | LOW | Stale `.next` cache from mid-session edits | **FIXED** — delete `.next`, `npm run build` |

### Remaining Work / What Needs Building
The following features are either incomplete or have not been started:

1. **Mobile Responsiveness** — The site is designed desktop-first. The PipelineDAG and 3DSystemGraph need responsive fallbacks for screens < 768px.
2. **Die Animation Visibility on Safari** — WebGL with `alpha: false` can sometimes render as pitch black on Safari due to compositor layering issues. Consider `alpha: true` + CSS background fallback.
3. **RunsChart Data** — `RunsChart.tsx` needs real simulation data. Currently it likely uses placeholder data. Replace with actual extracted Icarus Verilog `vvp` timing logs.
4. **Interactive Demo Section** — A `LiveDemoSection.tsx` exists in the component tree but is NOT rendered in `page.tsx`. It was removed during the BitbyBit migration. Consider building a terminal emulator that replays actual simulation commands.
5. **SEO Completeness** — `layout.tsx` has basic title/description. Add `og:image`, `twitter:card`, and structured data JSON-LD.
6. **Performance on Mobile** — Lower the `CopperParticles` count from 200 to 50 on mobile using `useMediaQuery`. Disable Bloom on mobile.
7. **`dev` script port flag** — `npm run dev` doesn't accept a port via `-p` flag the same way. Use `next dev --port 3007` instead. Similarly, `npm start` requires `next start --port 3006` not `npm start -p 3006`.

---

## 10. Commands Reference

```bash
# Development (hot-reload)
npx next dev --port 3000

# Production Build
npm run build          # or: npx next build

# Production Server (specify port correctly!)
npx next start --port 3005

# Lint
npm run lint

# Unit Tests
npx vitest run

# Deploy to Vercel (from project root)
vercel --name bitbybit

# Clear build cache if 500 errors appear
Remove-Item -Recurse -Force .next
npm run build
npx next start --port 3005
```

---

## 11. Deployment

- **Platform:** Vercel
- **Project name:** `bitbybit`
- **Production URL:** `https://bitbybit-sandy.vercel.app`
- **Build command:** `npm run build`
- **Output directory:** `.next` (auto-detected)
- The `.vercel/` directory exists in the project root with project metadata.

---

## 12. Hardware Context (for content accuracy)

The website describes a **real** Verilog-2005 hardware project. All metrics are from actual Icarus Verilog simulation logs. When writing content, use these verified facts:

| Fact | Value |
|---|---|
| Target architecture | Verilog-2005 RTL |
| FPGA target clock | 100 MHz |
| Imprint inference latency | **112 cycles** |
| Dynamic inference latency | **341 cycles** (12-layer) |
| Layer latency | **96 cycles** |
| Embedding extract | **10 ns** |
| Pipeline stages | **6 stages** (Embed → RoPE → GQA → Softmax → GELU → KV Quant) |
| Hardware modules | **51** proven RTL modules |
| Compute math | Ternary (-1, 0, 1 logic) — **Zero multipliers** |
| Memory model | Silicon Imprinting — Gemma 3 weights burned into Verilog ROM |
| Model target | Google Gemma 3 270M parameters |
| Weight compression | Q8.8 fixed-point |
| Previous bottleneck | AXI4-Lite DDR stalls, 19-cycle FP16 MACs, 18-cycle FSM stalls |
| Current speedup | 4.8x on matrix math (SIMD ternary vs FP16 MAC) |

---

*Last updated: March 2026. Analyzed by AI from full source read of `d:\Projects\BitbyBit\custom_gpu_project\website_next`.*
