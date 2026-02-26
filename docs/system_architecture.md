# BitbyBit Custom GPU — Complete System Architecture

## 1. Project Overview

BitbyBit is a custom GPU architecture designed specifically for accelerating Large Language Model (LLM) inference. The core innovation is **zero-skip multiplication** — detecting zero-valued operands before multiplication and bypassing the computation entirely, saving both time and power.

The system has been validated using **OPT-125M** (a 125 million parameter language model by Meta), running pure-NumPy inference with Q8.8 fixed-point quantization to simulate the custom hardware's behavior.

> [!IMPORTANT]
> **Key Achievement:** The hardware-only optimizations (deep pipelining + wide memory + INT4 parallel) deliver **27.8x throughput** with **zero quality loss**. Adding 30% weight pruning pushes this to **~35.8x** while maintaining coherent text output.

---

## 2. Hardware Architecture

The GPU is built from modular RTL blocks organized in three layers:

```mermaid
graph TB
    subgraph "GPU Top Level"
        GPU["gpu_top / gpu_top_pipelined"]
    end

    subgraph "Compute Layer"
        ZDM["Zero-Detect Multiplier<br/>8-bit × 8-bit → 16-bit<br/>Bypasses if either input = 0"]
        VPA["Variable Precision ALU<br/>4-bit (4×), 8-bit (2×), 16-bit (1×)<br/>Configurable parallel modes"]
        MAC["MAC Unit<br/>Multiply-Accumulate<br/>With zero-skip + accumulator"]
        SA["Systolic Array (N×N)<br/>Weight-stationary dataflow<br/>N² parallel MAC operations"]
        INT4["INT4 Pack Unit<br/>Pack 4×4-bit → 16-bit<br/>Per-lane zero detection"]
    end

    subgraph "Memory Layer"
        SMC["Sparse Memory Controller<br/>Compressed weight storage"]
        SMCW["Wide Memory Controller<br/>4-wide read + prefetch buffer<br/>Double buffering"]
        FDQ["Fused Dequantizer<br/>INT4/INT8 → full precision<br/>Scale + offset"]
    end

    GPU --> ZDM
    GPU --> VPA
    GPU --> SMC
    GPU --> FDQ
    ZDM --> MAC
    MAC --> SA
    SMCW --> GPU
    INT4 --> VPA
```

### Component Roles

| Component | Purpose | Key Feature |
|-----------|---------|-------------|
| **Zero-Detect Multiplier** | Core multiplication unit | Skips multiply if either input is zero → saves 1 cycle + power |
| **Variable Precision ALU** | Flexible precision compute | Supports 4-bit (4× parallel), 8-bit (2×), 16-bit (1×) modes |
| **MAC Unit** | Multiply-Accumulate | Accumulates products for dot products; integrates zero-skip |
| **Systolic Array** | Parallel matrix multiply | N×N array of MACs; weight-stationary; processes full rows |
| **Sparse Memory Controller** | Weight storage | Stores only non-zero weights in compressed format |
| **Wide Memory Controller** | High-bandwidth reads | Reads 4 values per cycle + prefetch buffer for next batch |
| **Fused Dequantizer** | Precision conversion | Converts INT4/INT8 stored weights to compute precision |
| **INT4 Pack Unit** | 4× parallel packing | Packs/unpacks 4 INT4 values into one 16-bit word |

---

## 3. Processing Pipeline

### 3.1 Original FSM Pipeline (gpu_top)

The original design uses a Finite State Machine with 5 sequential states. Each operation takes **5 clock cycles**.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> IDLE
    IDLE --> FETCH: start
    FETCH --> DEQUANT: weight loaded
    DEQUANT --> ZERO_CHECK: INT4→INT8 done
    ZERO_CHECK --> ALU: zero/non-zero decided
    ALU --> DONE: result ready
    DONE --> IDLE: output valid

    note right of FETCH: Read weight from\nsparse memory
    note right of DEQUANT: Scale + offset\nto working precision
    note right of ZERO_CHECK: Check if weight=0\nor activation=0
    note right of ALU: Multiply (or skip)\nusing VPA
```

**Throughput:** 1 result per 5 cycles

### 3.2 Deeply Pipelined Pipeline (gpu_top_pipelined)

The optimized design overlaps all 5 stages. While stage 5 outputs a result, stages 1-4 process the *next* 4 operations simultaneously.

```mermaid
gantt
    title Deep Pipeline — 5 Stages Overlapping
    dateFormat X
    axisFormat %s

    section Op 1
    FETCH     :a1, 0, 1
    DEQUANT   :a2, 1, 2
    ZERO_CHK  :a3, 2, 3
    ALU       :a4, 3, 4
    WRITEBACK :a5, 4, 5

    section Op 2
    FETCH     :b1, 1, 2
    DEQUANT   :b2, 2, 3
    ZERO_CHK  :b3, 3, 4
    ALU       :b4, 4, 5
    WRITEBACK :b5, 5, 6

    section Op 3
    FETCH     :c1, 2, 3
    DEQUANT   :c2, 3, 4
    ZERO_CHK  :c3, 4, 5
    ALU       :c4, 5, 6
    WRITEBACK :c5, 6, 7

    section Op 4
    FETCH     :d1, 3, 4
    DEQUANT   :d2, 4, 5
    ZERO_CHK  :d3, 5, 6
    ALU       :d4, 6, 7
    WRITEBACK :d5, 7, 8
```

**Throughput:** 1 result per 1 cycle (after 4-cycle fill latency) → **5× improvement**

### 3.3 Memory Bandwidth System

```mermaid
flowchart LR
    subgraph "External Memory"
        DRAM["Weight Storage<br/>(Compressed Sparse)"]
    end

    subgraph "Wide Memory Controller"
        FETCH["4-Wide Fetch<br/>Reads 4 values/cycle"]
        PB["Prefetch Buffer<br/>Next 4 values pre-loaded"]
        DB["Double Buffer<br/>A: Computing<br/>B: Loading"]
    end

    subgraph "Compute Pipeline"
        DQ["Dequantizer"]
        ZD["Zero Detect"]
        ALU["ALU"]
    end

    DRAM -->|"4 values/cycle"| FETCH
    FETCH --> DB
    PB -->|"pre-load next"| DB
    DB -->|"stream to pipeline"| DQ
    DQ --> ZD
    ZD --> ALU
```

**Impact:** 4× memory bandwidth → eliminates the memory bottleneck that limits compute throughput.

---

## 4. LLM Inference Flow

### 4.1 How OPT-125M Runs on BitbyBit

The model has 12 transformer layers, each containing self-attention and a feed-forward network (FFN). Every layer performs **6 matrix multiplications**.

```mermaid
flowchart TD
    INPUT["Input Token"] --> EMB["Token Embedding + Position Embedding"]
    
    EMB --> LAYER["Transformer Layer (×12)"]
    
    subgraph LAYER["One Transformer Layer"]
        LN1["LayerNorm"] --> QKVO["4 MatMuls:<br/>Q, K, V projections + Output"]
        QKVO --> ATTN["Attention Scores<br/>Softmax → Weighted Sum"]
        ATTN --> RES1["Residual Add"]
        RES1 --> LN2["LayerNorm"]
        LN2 --> FC1["MatMul: fc1<br/>(768 → 3072)"]
        FC1 --> RELU["ReLU Activation<br/>→ 92% zeros!"]
        RELU --> FC2["MatMul: fc2<br/>(3072 → 768)"]
        FC2 --> RES2["Residual Add"]
    end
    
    LAYER --> FLN["Final LayerNorm"]
    FLN --> LOGITS["MatMul: Logits<br/>(768 → 50,272 vocab)"]
    LOGITS --> SAMPLE["Sample Next Token"]

    style RELU fill:#ff6b6b,color:#fff
    style FC2 fill:#51cf66,color:#fff
```

### 4.2 Where Zero-Skip Happens

```mermaid
pie title "Zero-Skip Sources Across All MatMuls"
    "ReLU Zeros (FFN fc2 only)" : 92
    "Q8.8 Weight Zeros" : 5
    "Dense Ops (Q/K/V/out/fc1)" : 3
```

**Key Insight:** ReLU activation in the FFN creates **92% zeros** in the fc2 input. But this only affects **1 of 6 matmuls** per layer. Weight pruning attacks **all 6 matmuls**.

---

## 5. Optimization Stack

### 5.1 All Optimizations — Layered View

```mermaid
graph TB
    subgraph "Layer 4: Sparsity (Software)"
        WP["Weight Pruning<br/>Zero small weights"]
        NM["2:4 Structured Sparsity<br/>2 zeros per 4 weights"]
    end
    
    subgraph "Layer 3: Quantization"
        Q88["Q8.8 Fixed-Point<br/>16-bit: 8 int + 8 frac"]
        INT4Q["INT4 Quantization<br/>4-bit weights"]
    end
    
    subgraph "Layer 2: Compute Hardware"
        PIPE["Deep Pipeline<br/>5-stage, 1 op/cycle"]
        INT4P["INT4 Parallel<br/>4 mults per cycle"]
        ZS["Zero-Skip<br/>Bypass zero operands"]
    end
    
    subgraph "Layer 1: Memory Hardware"
        WIDE["4-Wide Memory Bus<br/>4 reads per cycle"]
        PRE["Prefetch + Double Buffer"]
        SPARSE["Compressed Sparse Storage"]
    end
    
    WP --> ZS
    NM --> ZS
    Q88 --> ZS
    INT4Q --> INT4P
    PIPE --> ZS
    WIDE --> PRE
    SPARSE --> WIDE
```

### 5.2 Measured Results — Quality vs Speed

| Config | Weight Sparse | Zero-Skip | Speedup | Text Quality |
|--------|:------------:|:---------:|:-------:|:------------:|
| Baseline (Q8.8 only) | 4.8% | 25.3% | 1.4× | ✅ Perfect |
| + Pipeline only | 4.8% | 25.3% | 6.9× | ✅ Perfect |
| **Pipeline + MemBW + INT4** | **4.8%** | **35.0%** | **27.8×** | **✅ Perfect** |
| Above + Pruning 10% | 10.2% | 29.4% | ~23× | ✅ Good |
| Above + Pruning 20% | 20.0% | 37.1% | ~30× | ✅ Good |
| **Above + Pruning 30%** | **30.0%** | **45.0%** | **~36×** | **✅ Good** |
| Above + Pruning 40% | 40.0% | 52.9% | ~42× | ⚠️ Repetitive loops |
| Above + Pruning 50% | 50.0% | 60.7% | ~48× | ❌ Nonsensical |
| Above + 2:4 Structured | 50.1% | 65.0% | ~42× | ❌ Garbage |
| Above + Pruning 70% | 70.0% | 76.3% | ~61× | ❌ Broken |

> [!CAUTION]
> **Quality cliff at 40% pruning.** The model starts producing repetitive loops ("I live in a star, I live in a star"). At 50%+, output becomes complete nonsense. Without fine-tuning/retraining, **30% is the maximum safe pruning level**.

### 5.3 Recommended Configuration

```mermaid
flowchart LR
    subgraph "Safe Config (Zero Quality Loss)"
        A["Pipeline<br/>5×"] --> B["Wide Memory<br/>~1.5×"] --> C["INT4 Parallel<br/>~4×"]
    end
    
    subgraph "Result"
        D["27.8× Throughput<br/>✅ Perfect Quality"]
    end
    
    C --> D

    style D fill:#51cf66,color:#fff
```

```mermaid
flowchart LR
    subgraph "Aggressive Config (Mild Quality Trade)"
        A2["Pipeline"] --> B2["Wide Memory"] --> C2["INT4 Parallel"] --> E2["30% Pruning"]
    end
    
    subgraph "Result2"
        D2["~36× Throughput<br/>✅ Good Quality"]
    end
    
    E2 --> D2

    style D2 fill:#ffd43b,color:#000
```

---

## 6. Data Format: Q8.8 Fixed-Point

The GPU operates on **Q8.8 fixed-point numbers**: 8 bits of integer + 8 bits of fraction = 16 bits total.

```mermaid
block-beta
    columns 16
    S["S"] I7["I₇"] I6["I₆"] I5["I₅"] I4["I₄"] I3["I₃"] I2["I₂"] I1["I₁"] F7["F₇"] F6["F₆"] F5["F₅"] F4["F₄"] F3["F₃"] F2["F₂"] F1["F₁"] F0["F₀"]
```

| Property | Value |
|----------|-------|
| Total bits | 16 (8 integer + 8 fraction) |
| Resolution | 1/256 = 0.00390625 |
| Range | −128.0 to +127.99609375 |
| Key property | Values < 1/512 round to **exact zero** → zero-skip! |

---

## 7. File Structure

```
custom_gpu_project/
├── rtl/                          # Verilog RTL hardware designs
│   ├── primitives/
│   │   ├── zero_detect_mult.v    # Core: zero-detecting multiplier
│   │   ├── variable_precision_alu.v  # Multi-mode ALU (4/8/16-bit)
│   │   ├── gpu_top.v             # Original FSM top-level
│   │   └── gpu_top_pipelined.v   # Deep pipelined top-level (5×)
│   ├── compute/
│   │   ├── mac_unit.v            # Multiply-accumulate unit
│   │   ├── systolic_array.v      # N×N systolic array
│   │   └── int4_pack_unit.v      # INT4 packing for 4× parallel
│   └── memory/
│       ├── sparse_memory_ctrl.v  # Original 1-wide memory
│       ├── sparse_memory_ctrl_wide.v  # 4-wide + prefetch buffer
│       └── fused_dequantizer.v   # INT4/INT8 dequantization
│
├── scripts/                      # Python simulation & benchmarks
│   ├── chat_opt.py               # OPT-125M inference engine
│   ├── benchmark_throughput.py   # 12-config throughput benchmark
│   ├── test_quality.py           # Quality vs pruning test
│   └── test_zeroskip.py          # Zero-skip rate analysis
│
├── tb/                           # Verilog testbenches
│   ├── primitives/
│   │   └── gpu_top_pipelined_tb.v
│   └── compute/
│
├── weights/                      # Model weights (git-ignored)
│   └── opt125m/
│       ├── opt125m_weights.npz   # Parsed NumPy weights
│       ├── vocab.json            # BPE tokenizer vocabulary
│       └── merges.txt            # BPE merge rules
│
├── docs/
│   ├── architecture.md
│   └── gpu_visualization.html
│
├── CLAUDE.md                     # Project conventions doc
└── .gitignore
```

---

## 8. FPGA Simulation & Verification

To validate these performance estimates on real hardware, the following free tools can replicate FPGA behavior:

### 8.1 Recommended Toolchain

```mermaid
flowchart LR
    subgraph "1. Simulation"
        V["Verilator<br/>(Fastest RTL sim)<br/>Verilog → C++ → run"]
        IV["Icarus Verilog<br/>(Simple, portable)<br/>Good for learning"]
    end
    
    subgraph "2. Waveform Viewing"
        GTK["GTKWave<br/>View signal traces<br/>Debug pipeline timing"]
    end
    
    subgraph "3. Synthesis"
        YS["Yosys<br/>RTL → Gate-level netlist<br/>Reports area & resources"]
    end
    
    subgraph "4. Place & Route"
        NPR["nextpnr / VPR<br/>Maps to FPGA fabric<br/>Reports timing"]
    end
    
    subgraph "5. Timing Analysis"
        STA["OpenSTA<br/>Static timing analysis<br/>Validates clock speed"]
    end
    
    V --> GTK
    IV --> GTK
    GTK --> YS
    YS --> NPR
    NPR --> STA
```

### 8.2 Tool Comparison

| Tool | Purpose | Why We Need It | Install |
|------|---------|----------------|---------|
| **Verilator** | RTL simulation | Fastest Verilog simulator; translates to C++ for speed. Can verify our pipeline does 1 op/cycle. | `apt install verilator` or [verilator.org](https://verilator.org) |
| **Icarus Verilog** | RTL simulation | Simpler alternative; good for quick testbench runs. Supports our Verilog subset. | `apt install iverilog` or [iverilog.com](http://iverilog.icarus.com) |
| **GTKWave** | Waveform viewer | Visualize pipeline stages, verify zero-skip timing, debug data hazards. | `apt install gtkwave` |
| **Yosys** | Logic synthesis | Converts our RTL to gate-level netlist; reports LUT/FF utilization. Shows if design fits on target FPGA. | `apt install yosys` or [yosyshq.net](https://yosyshq.net) |
| **F4PGA** | Full FPGA toolchain | End-to-end: synthesis + place-and-route + bitstream. Supports Xilinx 7-Series & Lattice FPGAs. | [f4pga.org](https://f4pga.org) |
| **OpenSTA** | Timing analysis | Validates our 100 MHz clock target; finds critical paths that limit frequency. | [github.com/The-OpenROAD-Project/OpenSTA](https://github.com/The-OpenROAD-Project/OpenSTA) |
| **Cocotb** | Python testbenches | Write hardware testbenches in Python instead of Verilog; integrates with Verilator. | `pip install cocotb` |

### 8.3 What Each Tool Validates

| Metric from Benchmark | Tool to Validate | How |
|----------------------|-------------------|-----|
| Pipeline = 1 op/cycle | **Verilator** + GTKWave | Run `gpu_top_pipelined_tb.v`, view waveform, count cycles per result |
| Zero-skip works | **Verilator** | Feed known zeros, verify `skipped` output asserts |
| 100 MHz clock target | **Yosys** + **OpenSTA** | Synthesize, extract timing, check critical path ≤ 10ns |
| Design fits on FPGA | **Yosys** | Check LUT/FF/BRAM usage vs target FPGA capacity |
| 4-wide memory throughput | **Verilator** | Run `sparse_memory_ctrl_wide` testbench, verify 4 values/cycle |
| INT4 4× parallel | **Verilator** | Feed packed INT4, verify 4 products per cycle |
| Overall end-to-end | **F4PGA** | Generate bitstream, deploy to physical FPGA board |

> [!TIP]
> **Quickest path to validation:** Install **Verilator** (or Icarus Verilog on Windows) + **GTKWave**. This lets you simulate all RTL modules and view waveforms to confirm pipeline timing. For full FPGA deployment, use the **F4PGA toolchain** with a Lattice iCE40 or Xilinx Artix-7 board.

---

## 9. Performance Summary

```mermaid
xychart-beta
    title "Throughput Speedup vs Pruning Level"
    x-axis ["0%", "10%", "20%", "30%", "40%", "50%", "70%"]
    y-axis "Speedup (×)" 0 --> 65
    bar [27.8, 23.4, 29.5, 35.8, 42.0, 48.2, 60.6]
    line [27.8, 23.4, 29.5, 35.8, 42.0, 48.2, 60.6]
```

### The Tradeoff

| | Safe Zone | Sweet Spot | Danger Zone |
|---|---|---|---|
| **Pruning** | 0-20% | **30%** | 40%+ |
| **Output Quality** | ✅ Perfect | ✅ Good | ❌ Degraded/Broken |
| **Speedup** | 27.8× | **~36×** | 42-61× (useless) |
| **Zero-Skip** | 25-37% | **45%** | 53-76% |

**Bottom line:** The hardware optimizations alone (pipeline + wide memory + INT4 parallel) deliver **27.8× speedup** with no quality loss at all. Adding 30% magnitude pruning safely pushes this to **~36×**.
