# 🧠 BitbyBit: Core Concepts, Visuals & Learning Guide

This document is your master cheat-sheet for the Science Fest. It breaks down **exactly** what we built, **why** we built it this way, and **how** it differs from standard CPUs/GPUs.

---

## 📺 Recommended YouTube Learning Path
Before presenting, watch these to build an unbreakable conceptual foundation:
1. **[Let's build GPT: from scratch, in code, spelled out (Andrej Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY)** - *Mandatory viewing.* Explains the software side of what our hardware is executing.
2. **[How AI Accelerators Work (Asianometry)](https://www.youtube.com/watch?v=y3h3qMlyX1A)** - Explains why standard CPUs are terrible at AI and why Systolic Arrays (like our MAC units) are required.
3. **[FlashAttention Explained (Umar Jamil)](https://www.youtube.com/watch?v=FthNMMA-TKE)** - Gives context on why optimizing memory bandwidth (like our Online Softmax) is the holy grail.

---

## 🏗️ Did we train our own model?
**No, and here is exactly why (tell the judges this):**
Training a "toy model" on a laptop teaches you nothing about *hardware*. It’s a math exercise. We instead took an industry-standard, pre-trained model mapping real English (**OPT-125M** by Meta/Facebook) and built **custom silicon architecture from scratch** to run it at blinding speeds. 

We specifically picked OPT-125M because it natively uses **ReLU activations**. We discovered that ReLU naturally zeros out **92%** of the data in the FFN layers. We built a hardware `zero_detect_mult` unit specifically to exploit this, skipping the math automatically. A custom toy model wouldn't have this fascinating real-world sparsity property!

---

## 🧩 Phase Breakdown & Visuals

### 1. The Compute Core (Systolic MACs vs ALUs)
**The Problem:** CPUs do math sequentially (A * B + C). This is too slow for 125 million parameters.
**Our Solution:** We built a custom Array of MACs (Multiply-Accumulate units).

```text
[Standard CPU]               [Our Custom GPU - Systolic Array]
Data -> [ALU] -> Out         Data_1 -> [MAC] -> [MAC] -> [MAC] ->
                                         |        |        |
                             Data_2 -> [MAC] -> [MAC] -> [MAC] -> Out
```
*Why it matters:* Our GPU processes entire vectors in one clock cycle. The data flows like water through a pipe (pipelined architecture).

### 2. Zero-Skip (The Free 1.35x Speedup)
**The Problem:** The model multiplies millions of numbers by `0`. Standard GPUs do the math anyway.
**Our Solution:** Our `zero_detect_mult` checks if the input is `< 0` or exactly `0`. If yes, it completely bypasses the multiplier and turns off the logic gates to save power. Tests show it skips **25.9%** of all math in the entire model.

### 3. Online Softmax (Streaming math)
**The Problem:** Normal Softmax requires reading all attention scores to find the maximum, then reading them *again* to divide by the sum. This requires 2 full passes over the memory.
**Our Solution (Phase 8):** We implemented single-pass "Streaming" Softmax (based on the Milakov 2018 paper). 

```text
Normal Softmax: [Read Memory] -> Find Max -> [Read Memory] -> Divide -> [Write] (100+ cycles)
Online Softmax: [Data Stream] ==> [Compute Max/Sum on the fly] ==> [Logits Out] (9 cycles!)
```

### 4. Hardware Token Scheduler 
**The Problem:** Usually, the GPU does math, sends the raw answer to the CPU, and the CPU says "Ah, the word is 'Apple'. Okay GPU, now run the next cycle." This CPU-to-GPU ping-pong wastes thousands of clock cycles.
**Our Solution:** We moved the infinite "keep generating words" loop directly into the hardware (`token_scheduler.v`). The CPU just says "Start", and our GPU autonomously generates 5, 10, or 100 tokens.

### 5. 2:4 Structured Sparsity Decoder (NVIDIA Ampere Style)
**The Problem:** Weights take up too much memory.
**Our Solution:** We adopted NVIDIA's newest trick. Out of every 4 weights in memory, we force the 2 smallest to be exactly zero. 

```text
[Dense Weights]:  [0.8,  0.1, -0.2, 1.5]  (4 Multiplies required)
                  [  ↓   CUT    CUT  ↓ ]
[2:4 Sparsity]:   [0.8,    0,    0, 1.5]  (Only 2 Multiplies required! + Bitmap to track index)
```
*Result:* We skip 50% of the attention multiplications directly at the hardware decoder level.

### 6. Power Management Unit (PMU)
**The Problem:** FPGAs melt if you run 100% of cores 100% of the time.
**Our Solution:** We built a finite state machine managing 3 states: `FULL`, `ECO`, and `SLEEP`. If the Token Scheduler is waiting for memory, the PMU instantly clock-gates the compute cores, cutting dynamic power draw by ~30%.
