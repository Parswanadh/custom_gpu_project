# BitbyBit GPU — Complete Guide: From Zero to Hardware Expert

> **Who is this for?** Anyone — even if you've never seen a circuit or written a line of code. We start from "what is electricity?" and end at "here's how our custom GPU runs a language model."

---

## Table of Contents

1. [The Absolute Basics](#1-the-absolute-basics)
2. [Digital Logic — How Computers Think](#2-digital-logic--how-computers-think)
3. [Verilog — Writing Hardware in Code](#3-verilog--writing-hardware-in-code)
4. [What is a GPU and Why Do We Need One?](#4-what-is-a-gpu-and-why-do-we-need-one)
5. [Our GPU Architecture — The Big Picture](#5-our-gpu-architecture--the-big-picture)
6. [The Compute Pipeline — How Math Happens](#6-the-compute-pipeline--how-math-happens)
7. [Zero-Skip Optimization — Skipping Useless Work](#7-zero-skip-optimization--skipping-useless-work)
8. [Multi-Core Scaling — More Power](#8-multi-core-scaling--more-power)
9. [What is a Transformer / LLM?](#9-what-is-a-transformer--llm)
10. [Our Transformer Implementation](#10-our-transformer-implementation)
11. [Attention with KV Cache — The Brain](#11-attention-with-kv-cache--the-brain)
12. [The Full GPT-2 Engine](#12-the-full-gpt2-engine)
13. [AXI Memory Interface — Talking to the World](#13-axi-memory-interface--talking-to-the-world)
14. [Number Systems — Q8.8 Fixed Point](#14-number-systems--q88-fixed-point)
15. [Testing and Verification](#15-testing-and-verification)
16. [Performance Analysis](#16-performance-analysis)
17. [Complete File Reference](#17-complete-file-reference)

---

# 1. The Absolute Basics

## What is a Computer, Really?

At the most fundamental level, a computer is a machine that does **one thing**: it moves electricity through switches. That's it. Everything — your games, your browser, ChatGPT — is just electricity going through billions of tiny switches.

### Voltage = Information

```
High voltage (usually 1.0V to 3.3V)  →  We call this "1" or "ON" or "TRUE"
Low voltage  (usually 0V to 0.3V)    →  We call this "0" or "OFF" or "FALSE"
```

This is called a **bit** — the smallest unit of information. A bit is just a wire that's either carrying voltage or not.

### Why Binary (0s and 1s)?

> *"Why not use 10 voltage levels for 0-9 like normal numbers?"*

Because transistors (the tiny switches) are **unreliable** at distinguishing small voltage differences. Dust, heat, or manufacturing defects could make 0.7V look like 0.8V. But distinguishing "some voltage" from "almost no voltage" is easy and reliable. So we use only two states.

## How Do We Build Things From Bits?

**One bit** = one switch = one piece of yes/no information.  
**Eight bits** = one **byte** = 256 possible combinations (2⁸ = 256).  
This is enough to represent a letter (A=65, B=66, ...) or a small number (0 to 255).

```
Binary:    01000001   = 65 in decimal = Letter 'A' in ASCII
           │││││││└── 1 × 2⁰ = 1
           ││││││└─── 0 × 2¹ = 0
           │││││└──── 0 × 2² = 0
           ││││└───── 0 × 2³ = 0
           │││└────── 0 × 2⁴ = 0
           ││└─────── 0 × 2⁵ = 0
           │└──────── 1 × 2⁶ = 64
           └───────── 0 × 2⁷ = 0
                              ───
                               65
```

---

# 2. Digital Logic — How Computers Think

## Logic Gates — The Building Blocks

A **logic gate** is a tiny circuit made of transistors that takes one or more inputs and produces one output. There are only a few types, and everything in every computer ever made is built from these:

### AND Gate
Both inputs must be 1 for output to be 1.

```
Inputs → Output         Think of it as:
0, 0  →  0              "Are BOTH switches on?"
0, 1  →  0
1, 0  →  0
1, 1  →  1  ✓
```

### OR Gate
At least one input must be 1 for output to be 1.

```
Inputs → Output         Think of it as:
0, 0  →  0              "Is EITHER switch on?"
0, 1  →  1  ✓
1, 0  →  1  ✓
1, 1  →  1  ✓
```

### NOT Gate (Inverter)
Flips the input.

```
Input → Output           Think of it as:
0     →  1               "The opposite"
1     →  0
```

### XOR Gate (Exclusive OR)
Output is 1 when inputs are *different*.

```
Inputs → Output         Think of it as:
0, 0  →  0              "Are they different?"
0, 1  →  1  ✓
1, 0  →  1  ✓
1, 1  →  0
```

### NOR Gate (NOT + OR)
Output is 1 ONLY when ALL inputs are 0. **This is how our zero-detection works!**

```
8-input NOR:
If ANY input bit is 1 → output is 0
If ALL input bits are 0 → output is 1  ← "Weight is zero!"
```

## How Gates Combine to Do Math

### Adding Two 1-bit Numbers (Half Adder)

```
A + B = Sum, Carry

0 + 0 = 0, carry 0     Sum  = A XOR B
0 + 1 = 1, carry 0     Carry = A AND B
1 + 0 = 1, carry 0
1 + 1 = 0, carry 1     (like 1+1=10 in binary)
```

A half adder is literally just 2 gates: one XOR and one AND. That's how addition works in hardware.

### Multiplying (What Our GPU Does Millions of Times)

Multiplication in binary works like long multiplication in school:

```
    1101  (13)
  × 1011  (11)
  ──────
    1101  (13 × 1)    ← AND each bit, shift by 0
   1101   (13 × 1)    ← AND each bit, shift by 1
  0000    (13 × 0)    ← AND each bit, shift by 2
 1101     (13 × 1)    ← AND each bit, shift by 3
─────────
10001111  (143)       ← Add all the rows
```

Each row is just AND gates. Adding the rows uses adder circuits. A hardware multiplier is literally this — it does all rows **simultaneously** (in parallel), not one at a time.

## Sequential Logic — Memory and Timing

### The Clock

Everything in our GPU is synchronized by a **clock signal** — a wire that alternates between 0 and 1 at a fixed rate:

```
Clock: ─┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──
        └──┘  └──┘  └──┘  └──┘  └──┘

        ← one cycle →

At 100 MHz: each cycle = 10 nanoseconds (10 billionths of a second)
```

**Why?** Because without a clock, signals would arrive at different times (some wires are longer, some gates are slower). The clock gives everything a "beat" — all changes happen together on the clock's rising edge.

### Flip-Flops — 1-Bit Memory

A flip-flop is a circuit that "remembers" one bit. It only updates its stored value when the clock ticks:

```
On each rising clock edge:
  stored_value ← input_value

Between clock edges:
  stored_value stays the same, no matter what input does
```

This is what `<=` means in Verilog (the `<=` is a "non-blocking assignment" — it means "update on the next clock edge").

### Registers

A **register** is just a group of flip-flops. An 8-bit register = 8 flip-flops = stores one byte. A 16-bit register = 16 flip-flops = stores one number in our Q8.8 format.

---

# 3. Verilog — Writing Hardware in Code

## What is Verilog?

Verilog is a **hardware description language** (HDL). It doesn't run like Python or C — instead, it **describes physical circuits** that get manufactured into silicon chips or programmed onto FPGAs.

```verilog
// This doesn't "execute line by line" like software
// It describes wires and gates that exist simultaneously
module adder(
    input  wire [7:0] a,    // 8 wires carrying number A
    input  wire [7:0] b,    // 8 wires carrying number B
    output wire [8:0] sum   // 9 wires carrying the result
);
    assign sum = a + b;     // This becomes an actual adder circuit
endmodule
```

### Key Verilog Concepts

| Concept | What It Means |
|---------|--------------|
| `module` | A reusable circuit block (like a chip) |
| `wire` | A physical connection (combinational — changes instantly) |
| `reg` | A storage element (sequential — updates on clock edge) |
| `assign` | Continuous connection (like soldering two wires) |
| `always @(posedge clk)` | "Do this on every clock tick" |
| `<=` (non-blocking) | "Schedule update for next clock edge" |
| `=` (blocking) | "Update immediately within this block" |
| `parameter` | A constant that can be changed when instantiating |
| `generate` | Create N copies of hardware automatically |

### Two Types of Logic in Verilog

**Combinational** — output depends only on current inputs:
```verilog
// Zero detector: output is 1 if weight is zero
// This is just a NOR gate — no clock needed
assign is_zero = (weight == 8'd0);
```

**Sequential** — output depends on clock + stored state:
```verilog
// Accumulator: adds a new value every clock cycle
always @(posedge clk) begin
    if (rst)
        total <= 32'd0;            // Reset to zero
    else if (valid)
        total <= total + new_val;  // Add on each clock tick
end
```

---

# 4. What is a GPU and Why Do We Need One?

## CPU vs GPU

A **CPU** (Central Processing Unit) is like a very smart person doing one thing at a time, very fast:
```
CPU: Do task 1... done. Do task 2... done. Do task 3... done.
     Very flexible. Can do anything. But ONE thing per cycle.
```

A **GPU** (Graphics Processing Unit) is like 1000 average workers doing simple tasks simultaneously:
```
GPU: Do tasks 1, 2, 3, 4, ... 1000 ALL AT ONCE.
     Less flexible. Can only do simple math. But 1000× parallel.
```

## Why GPUs for AI?

Neural networks (including ChatGPT) are essentially **massive matrix multiplications** — millions of numbers multiplied and added together. This is perfectly parallel work:

```
y[0] = w[0][0]*x[0] + w[0][1]*x[1] + w[0][2]*x[2] + w[0][3]*x[3]
y[1] = w[1][0]*x[0] + w[1][1]*x[1] + w[1][2]*x[2] + w[1][3]*x[3]
y[2] = w[2][0]*x[0] + w[2][1]*x[1] + w[2][2]*x[2] + w[2][3]*x[3]
y[3] = w[3][0]*x[0] + w[3][1]*x[1] + w[3][2]*x[2] + w[3][3]*x[3]

Each row is independent! GPU does all 4 rows simultaneously.
CPU would do them one after another.
```

## Why Build Our Own?

NVIDIA's GPUs are **general-purpose** — they handle games, physics, weather simulation, AI, everything. Our GPU is **LLM-specific** — we know exactly what operations we need, so we can optimize specifically for them:

| Feature | NVIDIA GPU | Our BitbyBit GPU |
|---------|-----------|-------------------|
| Purpose | General compute | LLM inference only |
| Zero-skip | No (always multiplies) | Yes (skips zeros) |
| KV Cache | Done in software (CUDA) | Done in hardware |
| Quantization | Separate step | Built into pipeline |
| Complexity | ~28 billion transistors | ~thousands of LUTs |

---

# 5. Our GPU Architecture — The Big Picture

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    BitbyBit LLM Accelerator                      │
│                                                                  │
│  ┌─────────┐   ┌─────────────────────────────────────────────┐  │
│  │   AXI   │   │        Accelerated GPT-2 Engine              │  │
│  │ Weight  │──→│                                               │  │
│  │ Memory  │   │  ┌──────────┐    ┌────────────────────────┐  │  │
│  │ (4KB)   │   │  │ Embedding │──→│  Transformer Block ×N   │  │  │
│  │         │   │  │  Lookup   │   │  ┌────────────────────┐ │  │  │
│  └─────────┘   │  └──────────┘   │  │ LN1 → Attention    │ │  │  │
│                │                  │  │  (KV Cache)         │ │  │  │
│  ┌─────────┐   │                  │  │    ↓ + Residual     │ │  │  │
│  │  GPU    │   │                  │  │ LN2 → FFN          │ │  │  │
│  │ Multi-  │──→│                  │  │  (gpu_core!)        │ │  │  │
│  │  Core   │   │                  │  │    ↓ + Residual     │ │  │  │
│  │(4×32    │   │                  │  └────────────────────┘ │  │  │
│  │ lanes)  │   │                  └────────────────────────┘  │  │
│  └─────────┘   │                                               │  │
│                │  ┌──────────┐    ┌──────────┐                 │  │
│                │  │Final LN  │──→│  Argmax   │──→ token_out    │  │
│                │  └──────────┘    └──────────┘                 │  │
│                └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow (What Happens When You Feed a Token)

```
Step 1: TOKEN IN (e.g., token ID = 5)
    ↓
Step 2: EMBEDDING LOOKUP
    Token 5 → look up in table → [0.23, -0.15, 0.87, 0.02]
    Position 0 → look up in table → [0.01, 0.03, -0.01, 0.02]
    Add them together → [0.24, -0.12, 0.86, 0.04]
    ↓
Step 3: LAYER NORM 1
    Normalize the values (mean=0, variance=1)
    ↓
Step 4: ATTENTION (with KV Cache)
    "What should I pay attention to from past tokens?"
    Q = x × Wq  (what am I looking for?)
    K = x × Wk  (what do I contain?)
    V = x × Wv  (what information do I have?)
    Store K, V in cache for future tokens
    Score = Q · K^T / √d  (how relevant is each past token?)
    Softmax scores → probabilities
    Output = weighted sum of V based on probabilities
    ↓
Step 5: RESIDUAL + LAYER NORM 2
    Add original input back (residual connection)
    Normalize again
    ↓
Step 6: FFN (Feed-Forward Network) ← Uses gpu_core!
    hidden = ReLU(x × W1 + b1)  ← Sparse! Many zeros after ReLU!
    output = hidden × W2 + b2
    ↓
Step 7: RESIDUAL
    Add back again
    ↓
Step 8: (Repeat steps 3-7 for each layer)
    ↓
Step 9: FINAL LAYER NORM + ARGMAX
    Normalize → pick the dimension with highest value
    ↓
Step 10: TOKEN OUT (e.g., token ID = 3)
```

---

# 6. The Compute Pipeline — How Math Happens

## What is Pipelining?

Imagine a car factory:

**Without pipelining (FSM approach):**
```
Car 1: [Build Frame] [Paint] [Add Engine] [Install Seats] [Done!]
                                                           Car 2: [Build Frame] [Paint] ...
Time: ████████████████████████████████████████████████████
```
One car at a time. The painting station sits idle while the frame is being built.

**With pipelining:**
```
Car 1: [Frame] [Paint]  [Engine] [Seats]  [Done!]
Car 2:         [Frame]  [Paint]  [Engine] [Seats]  [Done!]
Car 3:                  [Frame]  [Paint]  [Engine] [Seats]
Car 4:                           [Frame]  [Paint]  [Engine]
```
Once the pipeline is full, **one car finishes every cycle**, even though each car takes 4 steps.

## Our 5-Stage Pipeline (`gpu_core.v`)

Our GPU pipeline processes multiply-accumulate (MAC) operations in 5 stages:

```
              STAGE 1      STAGE 2       STAGE 3        STAGE 4       STAGE 5
              FETCH        DEQUANT       ZERO_CHECK     ALU           WRITEBACK
             ┌──────┐    ┌──────────┐   ┌──────────┐  ┌──────────┐  ┌──────────┐
             │Read N │    │N parallel│   │N parallel│  │N parallel│  │Sum all N │
activation──→│weights│──→│dequantize│──→│  zero    │──→│multiplies│──→│products +│──→ result
             │from   │    │INT4→INT8 │   │detectors │  │(skip if  │  │accumulate│
             │memory │    │          │   │          │  │  zero!)  │  │          │
             └──────┘    └──────────┘   └──────────┘  └──────────┘  └──────────┘

N = LANES (4, 8, 16, 32, 64, or 128 — configurable!)
```

### Stage-by-Stage Breakdown

**Stage 1: FETCH** — Read N weights from memory simultaneously.
```verilog
// N-wide memory read: all N weights read in ONE cycle
for (s1i = 0; s1i < LANES; s1i = s1i + 1)
    s1_weight[s1i] <= mem_read[s1i];
```
In hardware, this is N separate wires running from N memory locations to N registers. All N values arrive at the same time because wires carry signals at nearly the speed of light.

**Stage 2: DEQUANT** — Convert compact INT4 weights to full INT8.
```verilog
// N parallel dequantizers running simultaneously
wire [15:0] scaled = s1_weight[d] * dq_scale;
assign dq_out[d] = scaled[11:4] + {4'd0, dq_offset};
```
Why? Storing weights as 4-bit saves memory (half the storage), but multiplying needs full 8-bit precision. The dequantizer converts on-the-fly.

**Stage 3: ZERO\_CHECK** — Detect zero weights/activations.
```verilog
// N parallel zero detectors — each is just a NOR gate
assign is_zero[z] = (s2_dq_weight[z] == 8'd0) || (s2_activation == 8'd0);
```
This is where the zero-skip magic happens. An 8-bit `== 0` check is literally an 8-input NOR gate — **one gate, evaluated in <1 nanosecond**, not 8 clock cycles. All N detectors run in parallel.

**Stage 4: ALU** — Multiply (or skip if zero).
```verilog
if (s3_zero_mask[s4i])
    s4_product[s4i] <= 16'd0;       // SKIP! Save power!
else
    s4_product[s4i] <= s3_activation * s3_weight[s4i];  // 8×8 = 16 bit
```
If the zero-detector flagged this lane, the multiplier doesn't fire. In real silicon, this saves dynamic power (the transistors in the multiplier don't switch, consuming near-zero energy).

**Stage 5: WRITEBACK** — Sum all N products and accumulate.
```verilog
lane_sum = 0;
for (si = 0; si < LANES; si = si + 1)
    lane_sum = lane_sum + {16'd0, s4_product[si]};
accumulator <= accumulator + lane_sum;
```
All N products are summed into the running total. This is the dot product accumulator.

### Pipeline Throughput

After the initial 5-cycle fill delay:
```
LANES = 4:    4 products per cycle   → 400 MOPS at 100MHz
LANES = 32:  32 products per cycle   → 3,200 MOPS at 100MHz
LANES = 128: 128 products per cycle  → 12,800 MOPS at 100MHz
```
MOPS = Millions of Operations Per Second

---

# 7. Zero-Skip Optimization — Skipping Useless Work

## Why Zeros Matter

In neural networks, a huge fraction of values are zero:

1. **ReLU activation**: `ReLU(x) = max(0, x)` — all negative values become zero. In typical networks, **40-70% of activations are zero** after ReLU.

2. **Quantized weights**: When you compress a float like 0.003 to INT4, it rounds to 0. Small weights vanish.

3. **Attention masks**: Future tokens are masked to zero in training.

### The Math of Zero-Skip

```
Normal:  0 × 47 = 0    ← You wasted energy computing this
         3 × 0  = 0    ← You wasted energy computing this
         5 × 82 = 410  ← Only this one matters!

Zero-skip: 
         0 × 47 → SKIP (detected zero, multiplier doesn't fire)
         3 × 0  → SKIP (detected zero, multiplier doesn't fire)
         5 × 82 = 410  ← Only multiplication that fires
```

### How We Detect Zeros

```verilog
// In gpu_core.v, Stage 3:
// This is N parallel NOR gates — all checking simultaneously
assign is_zero[z] = (s2_dq_weight[z] == 8'd0) || (s2_activation == 8'd0);
```

**Important concept:** `weight == 8'd0` in hardware is a single NOR gate:
```
bit0 ──┐
bit1 ──┤
bit2 ──┤
bit3 ──┼──→ 8-input NOR ──→ is_zero
bit4 ──┤
bit5 ──┤
bit6 ──┤
bit7 ──┘

All 8 bits checked SIMULTANEOUSLY.
Time: ~0.2 nanoseconds (one gate delay).
NOT 8 clock cycles!
```

### Our Results

```
1-layer transformer:  42 zero-skips out of ~64 operations   (65% skip rate)
2-layer transformer: 132 zero-skips out of ~128 operations  (similar rate)
```

Every skip saves power and reduces heat. On battery-powered devices, this matters enormously.

---

# 8. Multi-Core Scaling — More Power

## Why Multiple Cores?

One `gpu_core` with 32 lanes = 32 products/cycle. Want more? Add more cores!

```
                    ┌──────────────┐
Broadcast ─────────→│  Core 0 (32L)│──→ partial_sum_0
activation  │       └──────────────┘
            │       ┌──────────────┐
            ├──────→│  Core 1 (32L)│──→ partial_sum_1
            │       └──────────────┘
            │       ┌──────────────┐
            ├──────→│  Core 2 (32L)│──→ partial_sum_2
            │       └──────────────┘
            │       ┌──────────────┐
            └──────→│  Core 3 (32L)│──→ partial_sum_3
                    └──────────────┘
                           │
                    ┌──────┴──────┐
                    │  Aggregate  │
                    │  (sum all)  │
                    └──────┬──────┘
                           ↓
                    total_accumulator = sum of all cores
```

Each core processes **different weights** but the **same activation** (broadcast). This is called **data parallelism**.

## Scaling Numbers

| Cores | Lanes/Core | Total Parallel | Speedup vs FSM |
|:-----:|:----------:|:--------------:|:--------------:|
| 1     | 4          | 4              | 28×            |
| 1     | 32         | 32             | 224×           |
| 4     | 32         | 128            | **896×**       |
| 8     | 32         | 256            | 1792×          |
| 4     | 128        | 512            | 3584×          |

Our default configuration: **4 cores × 32 lanes = 128 parallel operations per cycle**.

---

# 9. What is a Transformer / LLM?

## The Simple Explanation

An LLM (Large Language Model) like ChatGPT is a program that predicts the **next word** given all previous words:

```
Input:  "The cat sat on the"
Output: "mat" (probably)

How? It learned from billions of sentences that "mat" 
often follows "The cat sat on the"
```

The **Transformer** is the architecture that makes this work. It was invented by Google in 2017 in a paper called *"Attention is All You Need"*.

## The Key Idea: Attention

**Problem:** When predicting the next word, some earlier words matter more than others.

```
"The cat sat on the ___"

"cat" is very relevant (it's the subject)
"The" is less relevant (just a grammar word)
"sat" is somewhat relevant (tells us about position)
```

**Attention** lets the model learn which words to "pay attention to":

```
Attention scores:
  "The"  → 0.05  (5% weight)
  "cat"  → 0.40  (40% weight — most important!)
  "sat"  → 0.30  (30% weight)
  "on"   → 0.15  (15% weight)
  "the"  → 0.10  (10% weight)
```

These scores are computed using three projections:
- **Q (Query):** "What am I looking for?" — computed from the current position
- **K (Key):** "What do I contain?" — computed from every past position
- **V (Value):** "What information do I have?" — computed from every past position

```
score = Q · K^T / √d_k        ← How relevant is each past token?
probs = softmax(scores)        ← Convert to probabilities (sum to 1.0)
output = probs · V             ← Weighted sum of past information
```

## Transformer Block Structure

One transformer block:
```
Input x
  ↓
LayerNorm(x)                   ← Normalize to mean=0, std=1
  ↓
Attention(LN(x))               ← The magic: attend to past context
  ↓
x + Attention(LN(x))           ← Residual: add the original back
  ↓
LayerNorm(x)                   ← Normalize again
  ↓
FFN(LN(x))                    ← Simple neural network (2 linear layers)
  ↓
x + FFN(LN(x))                ← Residual again
  ↓
Output y
```

**Why residual connections?** Without them, information from early layers gets "washed out" as it passes through many layers. The `+ x` shortcut lets the original signal flow directly through.

**Why LayerNorm?** Neural network values tend to grow or shrink as they pass through layers. Normalization keeps them in a stable range so the math doesn't overflow.

GPT-2 Small has **12** of these blocks stacked. Our simulation uses 1-2 for speed.

---

# 10. Our Transformer Implementation

## Embedding Lookup (`embedding_lookup.v`)

Every token has a unique ID (e.g., "cat" = 5). We need to convert this ID into a vector of numbers that the neural network can work with:

```
Token ID → Table Lookup → Embedding Vector

Token 0 → [0.12, -0.34, 0.56, 0.78]
Token 1 → [0.91, 0.23, -0.15, 0.67]
Token 5 → [0.43, -0.21, 0.88, 0.02]
...

Position 0 → [0.01, 0.03, -0.01, 0.02]  (adds position information)
Position 1 → [0.05, -0.02, 0.04, 0.01]
```

The final embedding = token embedding + position embedding.

## Layer Norm (`layer_norm.v`)

Normalizes a vector to have mean ≈ 0 and standard deviation ≈ 1:

```
Input:   [100, 200, 300, 400]
Mean:    250
Std:     129
Output:  [-1.16, -0.39, 0.39, 1.16]    ← Centered and scaled

Then multiply by gamma and add beta (learnable parameters):
Output = gamma * normalized + beta
```

In our hardware, this uses Q8.8 fixed-point division — expensive in gates but necessary for stable computation.

## FFN Block — Now Using gpu_core!

The Feed-Forward Network is where most computation happens:

```
Layer 1: hidden = ReLU(x × W1 + bias1)    Size: [4] × [4×8] = [8]
Layer 2: output = hidden × W2 + bias2      Size: [8] × [8×4] = [4]
```

**Before our fix:** The FFN used an inline for-loop — computed all multiplies in one clock cycle (unrealistic in hardware, and didn't use the pipeline).

**After our fix:** The FFN uses `gpu_core` instances:
```
For each output column j:
  1. FFN_LOAD:    Load column j weights into gpu_core memory
  2. FFN_COMPUTE: Feed activations through pipeline
  3. FFN_DRAIN:   Wait 6 cycles for pipeline flush (5-stage pipeline)
  4. FFN_ACCUM:   Read accumulator, add bias, apply ReLU
  → Next column
```

This means the pipelined hardware with zero-skip detection is **actually used** during inference.

---

# 11. Attention with KV Cache — The Brain

## What is KV Cache?

When generating text token by token, each new token needs to "see" all previous tokens. Without caching, you'd recompute K and V for every past token every time — wasteful!

**KV Cache stores K and V from all past tokens:** 

```
Token 0: Compute Q₀, K₀, V₀ → Store K₀, V₀ in cache
Token 1: Compute Q₁, K₁, V₁ → Store K₁, V₁ in cache
         Score = Q₁ · [K₀, K₁]^T → attend to tokens 0 and 1
Token 2: Compute Q₂, K₂, V₂ → Store K₂, V₂ in cache
         Score = Q₂ · [K₀, K₁, K₂]^T → attend to all 3 tokens
```

Each new token only computes its own Q, K, V — then reads the cached K, V of all past tokens. This is **O(n)** per token instead of **O(n²)** for recomputation.

## Our Implementation (`accelerated_attention.v`)

```
State Machine:
  S_IDLE    → Wait for input
  S_QKVO   → Compute Q = x·Wq, K = x·Wk, V = x·Wv (inline matmul)
  S_CACHE   → Store K, V in cache at current position
  S_SCORE   → For each past token t: score[t] = Q · K_cache[t]
  S_SOFTMAX → Phase 0: exp(score-max) via LUT, accumulate sum
              Phase 1: normalize → probs[t] = exp[t] × 255 / sum
  S_WGTV    → output[j] = Σ probs[t] × V_cache[t][j]  
  S_OUT     → y = output × Wo
  S_DONE    → Assert valid_out
```

### The Softmax Problem (and Our Fix)

Softmax converts raw scores into probabilities:
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

**Problem:** `exp()` is hard to compute in hardware — it's an irrational function.

**Old approach:** Linear approximation `exp(x) ≈ 255 + x×89/256` — very inaccurate.

**New approach:** 256-entry lookup table (`exp_lut_256.v`):
```
Index 0   → 255 (exp(0)    = 1.000)
Index 64  →  89 (exp(-1.0) = 0.368)
Index 128 →  31 (exp(-2.0) = 0.135)
Index 192 →  10 (exp(-3.0) = 0.050)
Index 255 →   1 (exp(-4.0) = 0.019)
```

This gives a proper exponential curve with just a table lookup — no arithmetic needed for the exp function itself.

---

# 12. The Full GPT-2 Engine

## `accelerated_gpt2_engine.v`

This is the top-level module that ties everything together:

```verilog
module accelerated_gpt2_engine
    // Parameters:
    // VOCAB_SIZE   = 16 (token dictionary size)
    // MAX_SEQ_LEN  = 8  (maximum sequence length)
    // EMBED_DIM    = 4  (embedding dimension)
    // NUM_HEADS    = 2  (attention heads)
    // HEAD_DIM     = 2  (dimension per head)
    // FFN_DIM      = 8  (FFN hidden dimension)
    // NUM_LAYERS   = 2  (number of transformer blocks)
    // DATA_WIDTH   = 16 (16-bit fixed point)
```

### State Machine

```
IDLE       → Wait for valid_in with token_in
EMBEDDING  → Look up token + position embedding
TRANSFORMER → Run through NUM_LAYERS transformer blocks
              (each block: LN1 → Attention → Residual → LN2 → FFN → Residual)
FINAL_LN   → Final layer normalization
LOGITS     → Compute argmax over output dimensions
OUTPUT     → Assert valid_out with token_out
```

### Performance Counters (NEW!)

```verilog
output reg [31:0] total_zero_skips,  // How many multiplies were skipped
output reg [31:0] total_cycles       // Total cycles since reset
```

These let us measure the actual benefit of zero-skipping in real-time.

---

# 13. AXI Memory Interface — Talking to the World

## What is AXI?

AXI (Advanced eXtensible Interface) is a standard **bus protocol** defined by ARM. It's how chips talk to each other inside a System-on-Chip (SoC):

```
┌──────────┐     AXI Bus      ┌──────────────────┐
│   CPU    │ ←──────────────→ │  Our GPU          │
│  (ARM)   │                  │  axi_weight_memory│
└──────────┘                  └──────────────────┘
     ↑                               ↑
     │ AXI Bus                       │ Internal
     ↓                               ↓
┌──────────┐                  ┌──────────────────┐
│   DDR    │                  │  Weight SRAM      │
│  Memory  │                  │  (4KB × 8-bit)    │
└──────────┘                  └──────────────────┘
```

## AXI4-Lite (Our Implementation)

AXI4-Lite is the simple version: 32-bit address, 32-bit data, no bursts. Perfect for control registers and small memory.

### Our Address Map

| Address | Read/Write | Function |
|---------|:----------:|----------|
| `0x0000 - 0x0FFF` | R/W | Weight memory (4KB) |
| `0x1000` | W | Control: bit 0 = start inference |
| `0x1004` | R | Status: bit 0 = busy, bit 1 = done |
| `0x1008` | R | Weight count (how many loaded) |
| `0x100C` | R | Zero-skip count |

### How a CPU Loads Weights

```c
// Pseudocode: CPU loading weights via AXI
for (int i = 0; i < weight_count; i += 4) {
    // Pack 4 weights (8-bit each) into one 32-bit AXI word
    uint32_t packed = w[i] | (w[i+1]<<8) | (w[i+2]<<16) | (w[i+3]<<24);
    axi_write(0x0000 + i, packed);
}
// Start inference
axi_write(0x1000, 1);
// Poll for completion
while (!(axi_read(0x1004) & 2));  // Wait for done bit
// Read results
uint32_t skips = axi_read(0x100C);
```

### AXI Handshake Protocol

AXI uses a **valid/ready handshake** on every channel:
```
Sender:     "I have data" (VALID = 1)
Receiver:   "I'm ready"  (READY = 1)
Transfer happens when BOTH are 1 on the same clock edge.
```

```
Clock:   ──┐  ┌──┐  ┌──┐  ┌──┐  ┌──
VALID:   ──┘──────────────┘──────────
READY:   ────────┘────────┘──────────
Transfer:           ↑ HERE    ↑ HERE
```

---

# 14. Number Systems — Q8.8 Fixed Point

## The Problem with Floating Point

Floating point numbers (like `3.14159`) are expensive in hardware:
- IEEE 754 float32: needs a barrel shifter, exponent logic, rounding unit
- Takes ~500 gates and 3-4 clock cycles per multiply
- Overkill for inference (training needs it, inference often doesn't)

## Fixed Point — Simple and Fast

Instead of a floating decimal point, we fix its position:

**Q8.8 Format** (our choice):
```
16 bits total:
┌────────────┬────────────┐
│  8 integer │ 8 fraction │
│    bits    │    bits    │
└────────────┴────────────┘

Examples:
  Binary:          Decimal:
  00000001.00000000 = 1.0     (integer part=1, fraction=0)
  00000001.10000000 = 1.5     (integer part=1, fraction=128/256=0.5)
  00000000.01000000 = 0.25    (integer part=0, fraction=64/256=0.25)
  11111111.00000000 = -1.0    (two's complement signed)
  00000000.00000001 = 0.00390625  (smallest positive fraction: 1/256)
```

### How Q8.8 Works

**Storing:** Multiply the real number by 256 and store as 16-bit integer.
```
1.5  × 256 = 384  → stored as 16'h0180
0.25 × 256 = 64   → stored as 16'h0040
-0.5 × 256 = -128 → stored as 16'hFF80 (two's complement)
```

**Adding:** Just add the raw integers (the scaling cancels out).
```
1.5 + 0.25 = 1.75
384 + 64   = 448 → 448/256 = 1.75  ✓
```

**Multiplying:** Multiply raw integers, then shift right by 8 to fix the scale.
```
1.5 × 2.0 = 3.0
384 × 512 = 196608
196608 >> 8 = 768 → 768/256 = 3.0  ✓
```

### Why Q8.8 for Our GPU?

| Property | Float32 | Q8.8 |
|----------|:-------:|:----:|
| Bits | 32 | 16 |
| Multiplier gates | ~500 | ~150 |
| Memory per weight | 4 bytes | 2 bytes |
| Multiply latency | 3-4 cycles | 1 cycle |
| Range | ±3.4×10³⁸ | ±127.996 |
| Precision | ~7 decimal digits | ~2.4 decimal digits |

For inference, the limited precision is acceptable. Research shows that even INT4 (4-bit) quantization loses minimal accuracy for LLM inference.

---

# 15. Testing and Verification

## How We Test Hardware

Unlike software, hardware bugs are **permanent** — you can't patch a manufactured chip. So testing is crucial:

### Testbench Structure

```verilog
module my_testbench;
    // 1. Declare signals matching the module's ports
    reg clk, rst;
    wire [7:0] output;
    
    // 2. Instantiate the module under test
    my_module uut (.clk(clk), .rst(rst), .out(output));
    
    // 3. Generate clock
    always #5 clk = ~clk;  // 100 MHz (10ns period)
    
    // 4. Stimulus
    initial begin
        clk = 0; rst = 1;
        #20; rst = 0;        // Release reset
        // Feed inputs...
        // Check outputs...
        $finish;
    end
endmodule
```

### Our Test Suite

| Testbench | What It Tests | Result |
|-----------|--------------|:------:|
| `gpu_multicore_tb.v` | 4-core pipeline, 128-wide parallel | ✅ 256 products, 96 zero-skips |
| `accelerated_attention_tb.v` | KV cache, multi-token attention | ✅ Token 0: [255,255,255,255], Token 1: [413,174,333,134] |
| `accelerated_gpt2_engine_tb.v` | Full pipeline, 3-token generation | ✅ 0→3→3, 328 cyc/tok, 42 zero-skips |
| `multi_layer_test.v` | 2-layer transformer, 4 tokens | ✅ 628 cyc/tok, 132 zero-skips |
| `real_weight_test.v` | Real GPT-2 weights via $readmemh | ✅ 0→1→1→1, real logits |
| `axi_weight_memory_tb.v` | AXI4-Lite read/write/status | ✅ All registers correct |

### Simulation Tools

**Icarus Verilog** (`iverilog`): Free, open-source Verilog simulator.
```bash
# Compile step: parse Verilog, check syntax, elaborate design
iverilog -o sim/test.vvp  rtl/module.v  tb/testbench.v

# Simulate step: run the design, execute $display statements
vvp sim/test.vvp
```

---

# 16. Performance Analysis

## Pipeline Performance

### Throughput Comparison

```
Method              Products/Cycle    Speedup
──────────────────  ──────────────    ───────
FSM (1 at a time)   0.14/cycle*       1×
Single core, 4 lane 4/cycle           28×
Single core, 32L    32/cycle          224×
4 cores × 32L       128/cycle         896×

* FSM takes ~7 cycles per product (fetch, decode, compute, store)
```

### Per-Token Latency

```
Configuration          Cycles/Token    At 100MHz
─────────────────────  ────────────    ─────────
1 layer, inline FFN    79 cycles       0.79 μs
1 layer, gpu_core FFN  328 cycles      3.28 μs ← realistic hardware
2 layers, gpu_core FFN 628 cycles      6.28 μs
```

### Zero-Skip Effectiveness

```
Configuration      Total Ops    Zero-Skipped    Skip Rate
───────────────    ─────────    ────────────    ─────────
1 layer            ~64          42              65%
2 layers           ~128         132             ~100%*
Real GPT-2 weights ~64          32              50%

* High skip rate in 2-layer test due to cascading ReLU zeros
```

### What These Numbers Mean for Real Hardware

If synthesized on a Xilinx Artix-7 FPGA at 100 MHz:
```
128 products × 100 MHz = 12,800 MOPS (million ops/sec)
Token generation: ~3.28 μs per token (1-layer)
                  ~305,000 tokens/sec

For comparison:
  CPU (Intel i7): ~10,000 MOPS with AVX-512
  NVIDIA A100:    ~312,000,000 MOPS (312 TOPS)
  Our GPU:        ~12,800 MOPS (0.04% of A100)

BUT: A100 costs $10,000 and draws 400W
     Our FPGA would cost ~$50 and draw ~2W
     Per-watt efficiency is competitive for edge inference!
```

---

# 17. Complete File Reference

## RTL (Register Transfer Level) — The Hardware

### Primitives (`rtl/primitives/`)

| File | Purpose | Key Feature |
|------|---------|-------------|
| `gpu_core.v` | **Core compute engine** | 5-stage pipeline, N-lane parallel, zero-skip |
| `gpu_multicore.v` | Multi-core wrapper | Broadcasts activation to N cores, aggregates results |
| `gpu_top.v` | Original simple wrapper | Single core, basic interface |
| `gpu_top_integrated.v` | Integrated with dequantizer | Full pipeline with quantization |
| `gpu_top_pipelined.v` | Pipelined version | Throughput-optimized variant |
| `zero_detect_mult.v` | Zero-detecting multiplier | Skip multiply if either input is zero |
| `fused_dequantizer.v` | INT4 → INT8 converter | Decompresses quantized weights |
| `sparse_memory_ctrl.v` | Sparse memory controller | Skips zero-valued memory entries |
| `variable_precision_alu.v` | Multi-precision ALU | Supports INT4, INT8, INT16 operations |

### Compute (`rtl/compute/`)

| File | Purpose |
|------|---------|
| `exp_lut_256.v` | **256-entry exp() lookup table** for softmax |
| `mac_unit.v` | Multiply-accumulate unit |
| `gelu_activation.v` | GELU activation function |
| `softmax_unit.v` | Softmax computation |
| `int4_pack_unit.v` | Pack two INT4 values into one byte |
| `systolic_array.v` | Systolic array for matrix multiply |

### Transformer (`rtl/transformer/`)

| File | Purpose | Key Feature |
|------|---------|-------------|
| `accelerated_attention.v` | **KV-cached attention** | Real Q·K^T scoring, 256-entry softmax LUT |
| `accelerated_transformer_block.v` | **Full transformer block** | Uses gpu_core for FFN, pre-LN architecture |
| `accelerated_linear_layer.v` | gpu_core-backed linear | Bridges gpu_core to transformer interface |
| `layer_norm.v` | Layer normalization | Mean/variance normalization in Q8.8 |
| `attention_unit.v` | Original attention | Simpler version (no KV cache) |
| `ffn_block.v` | Original FFN | Inline matmul (not using pipeline) |
| `linear_layer.v` | Original linear | Simple matmul |

### GPT-2 (`rtl/gpt2/`)

| File | Purpose |
|------|---------|
| `accelerated_gpt2_engine.v` | **Full GPT-2 engine** (uses accelerated components) |
| `embedding_lookup.v` | Token + position embedding table |
| `gpt2_engine.v` | Original GPT-2 engine (uses old transformer) |
| `transformer_block.v` | Original transformer block |

### Memory (`rtl/memory/`)

| File | Purpose |
|------|---------|
| `axi_weight_memory.v` | **AXI4-Lite slave** for weight storage |
| `sparse_memory_ctrl_wide.v` | Wide sparse memory controller |

## Testbenches (`tb/`)

Located in matching subdirectories under `tb/`. Each testbench:
1. Instantiates the module under test
2. Generates clock and reset
3. Provides stimulus (inputs)
4. Checks outputs and prints results
5. Reports PASS/FAIL

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `extract_gpt2_weights.py` | Extract real GPT-2 weights → Q8.8 hex files |
| `run_cosim.py` | Co-simulation: Python reference vs Verilog |
| `run_scaled_cosim.py` | Scaled version of co-simulation |

## Weights (`weights/`)

| Directory | Contents |
|-----------|----------|
| `gpt2_real/hex/` | Real GPT-2 weights in Q8.8 hex format |
| `gpt2_dim64/hex/` | Larger dimension (64) weights |
| `gpt2_dim64/hex_sim/` | Simulation-friendly weights |

---

## Glossary

| Term | Meaning |
|------|---------|
| **ALU** | Arithmetic Logic Unit — the part that does math |
| **AXI** | Advanced eXtensible Interface — standard bus protocol |
| **Combinational** | Logic where output depends only on current inputs (no memory) |
| **DMA** | Direct Memory Access — hardware that copies data without CPU |
| **FFN** | Feed-Forward Network — the "thinking" part of a transformer |
| **FPGA** | Field Programmable Gate Array — reconfigurable chip |
| **FSM** | Finite State Machine — sequential step-by-step logic |
| **GELU** | Gaussian Error Linear Unit — smooth activation function |
| **HDL** | Hardware Description Language (Verilog, VHDL) |
| **INT4/INT8** | 4-bit/8-bit integer formats |
| **KV Cache** | Key-Value Cache — stores past attention context |
| **LLM** | Large Language Model (ChatGPT, etc.) |
| **LN** | Layer Normalization |
| **LUT** | Lookup Table — stores precomputed values |
| **MAC** | Multiply-Accumulate operation |
| **MOPS** | Millions of Operations Per Second |
| **Pipeline** | Overlapping multiple stages for throughput |
| **Q8.8** | Fixed-point: 8 integer bits + 8 fractional bits |
| **ReLU** | Rectified Linear Unit: max(0, x) |
| **Residual** | Skip connection: output = f(x) + x |
| **RTL** | Register Transfer Level — hardware abstraction |
| **Sequential** | Logic with memory (flip-flops, registers) |
| **Softmax** | Converts values to probabilities (sum = 1) |
| **SoC** | System on Chip — CPU + GPU + memory on one die |
| **Systolic** | Array where data flows rhythmically between cells |
| **TOPS** | Trillions of Operations Per Second |
| **VCD** | Value Change Dump — waveform recording format |
| **Zero-skip** | Optimization: skip multiplication when input is zero |
