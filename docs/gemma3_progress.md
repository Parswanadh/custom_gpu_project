# Gemma 3 270M — End-to-End FPGA Implementation Progress

> **Owner:** BitbyBit Custom Silicon
> **Started:** 2026-05-13
> **Target:** Full Gemma 3 270M inference on FPGA with 512 MB SRAM
> **Methodology:** Taalas HC1-inspired hard-wired architecture, SRAM-based (updatable)

---

## Master Execution Plan

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Environment + NVIDIA baseline benchmark | 🟡 IN PROGRESS |
| **2** | Golden model + INT4 weight export | ⬜ NOT STARTED |
| **3** | RTL primitives upgrade (RMSNorm, MQA, Gated MLP, RoPE) | ⬜ NOT STARTED |
| **4** | Gemma3 transformer block assembly | ⬜ NOT STARTED |
| **5** | Gemma3 engine (18-layer + embedding + LM head) | ⬜ NOT STARTED |
| **6** | Memory map + FPGA top wrapper | ⬜ NOT STARTED |
| **7** | End-to-end simulation + validation | ⬜ NOT STARTED |
| **8** | FPGA dump folder + synthesis guide | ⬜ NOT STARTED |
| **9** | NVIDIA RTX 4070 comparison | ⬜ NOT STARTED |
| **10** | Final documentation + deliverables | ⬜ NOT STARTED |

---

## Gemma 3 270M Architecture Reference

| Parameter | Value |
|-----------|-------|
| Total Parameters | ~270M (100M transformer + 170M embedding) |
| Layers | 18 |
| Hidden Size (d_model) | 640 |
| Intermediate Size (FFN) | 2048 |
| Attention Heads (Q) | 4 |
| KV Heads | 1 (Multi-Query Attention) |
| Head Dimension | 256 |
| Vocabulary Size | 262,144 (256K) |
| Max Context Length | 32,768 tokens |
| Positional Encoding | RoPE |
| Normalization | RMSNorm + QK-Norm |
| MLP Structure | Gated MLP (gate/up/down) with GELU |
| Architecture | Text-only, Dense (no MoE) |

## FPGA Constraint: 512 MB SRAM Budget

| Region | Size (INT4 weights) | Address Range |
|--------|---------------------|---------------|
| Embedding matrix (262K × 640) | ~80 MB | 0x00000000 — 0x04FFFFFF |
| 18 transformer layers | ~49 MB | 0x05000000 — 0x07FFFFFF |
| KV cache (32K context) | ~74 MB | 0x08000000 — 0x0BFFFFFF |
| Activation scratch | ~16 MB | 0x0C000000 — 0x0CFFFFFF |
| Softmax/attention scratch | ~16 MB | 0x0D000000 — 0x0DFFFFFF |
| **TOTAL USED** | **~235 MB** | |
| **HEADROOM** | **~277 MB** | |

✅ Model fits comfortably in 512 MB at INT4 quantization.

## Taalas HC1 Reference (Competitive Positioning)

| Metric | Taalas HC1 | BitbyBit FPGA (projected) |
|--------|-----------|---------------------------|
| Process | TSMC 6nm ASIC | FPGA (programmable) |
| Weight storage | Mask-ROM (fixed) | SRAM (updatable) |
| Target model | Llama 3.1 8B | Gemma 3 270M |
| Throughput | 17K tok/sec | ~16-64 tok/sec (est. @ 200MHz) |
| Power | 200-250W | 10-20W |
| Model updatable? | ❌ No | ✅ Yes |
| Cost per chip | Unknown (ASIC tape-out) | $200-500 (FPGA board) |

## NVIDIA RTX 4070 Baseline (Local GPU)

| Spec | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| VRAM | 8188 MiB (8 GB) |
| CUDA Cores | 4608 |
| Tensor Cores | 144 |
| TDP | 110W |
| Driver | 581.95 |
| CUDA | 13.0 |
| PyTorch | 2.5.1+cu121 |

---

## Phase 1: Environment + NVIDIA Baseline

**Started:** 2026-05-13 14:06 IST

### Step 1.1: Environment Check ✅
- nvidia-smi: RTX 4070 Laptop, 8GB VRAM, CUDA 13.0
- Conda base env: PyTorch 2.5.1+cu121, CUDA available
- Icarus Verilog: D:\Tools\iverilog\bin\iverilog.exe (12.0)
- yowasp-yosys available in conda env `yosys-tools`

### Step 1.2: Gemma3 270M NVIDIA Benchmark
- Status: 🟡 IN PROGRESS
- Plan: Download via transformers, run INT4/INT8/FP16, measure tok/sec and latency

---

## Phase 2: Golden Model + Weight Export

### Step 2.1: Download Gemma3 270M
- Source: google/gemma-3-270m-pt (or instruction-tuned variant)

### Step 2.2: Quantize to INT4
- Method: GPTQ or bitsandbytes NF4
- Target: Generate .hex files per layer for Verilog $readmemh

### Step 2.3: Generate Golden Vectors
- For each of 18 layers: save input/output activation pairs in Q8.8
- Save embedding lookup results
- Save final logits
- Used for bit-exact RTL validation

---

## Phase 3-10: [Will be documented as work progresses]

---

## Appendix A: File Manifest

### New RTL Files (to be created)
- `rtl/gemma3/gemma3_block.v` — Full transformer layer
- `rtl/gemma3/gemma3_engine.v` — 18-layer engine
- `rtl/gemma3/mqa_attention.v` — Multi-Query Attention (4Q/1KV)
- `rtl/gemma3/gemma3_embedding.v` — 256K vocab embedding lookup
- `rtl/gemma3/gemma3_memory_map.v` — 512MB SRAM address decoder
- `rtl/gemma3/gemma3_fpga_top.v` — FPGA top wrapper

### New Testbenches
- `tb/gemma3/gemma3_block_tb.v`
- `tb/gemma3/gemma3_engine_tb.v`
- `tb/gemma3/mqa_attention_tb.v`

### New Scripts
- `scripts/gemma3_benchmark_nvidia.py` — RTX 4070 benchmark
- `scripts/gemma3_export_weights.py` — INT4 weight export to hex
- `scripts/gemma3_golden_vectors.py` — Per-layer golden vector generation

### FPGA Dump Folder
- `fpga_dump/` — Complete synthesis-ready package
