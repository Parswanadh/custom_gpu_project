# Architectural Comparison: Gemma-3 270M vs. NanoGPT (GPT-2)

This document outlines the key architectural differences between standard GPT-2 (typically implemented as NanoGPT) and the more modern Gemma-3 270M model. This analysis is aimed at guiding the implementation of Gemma-3 270M support on our custom Verilog GPU.

## High-Level Summary

While both are decoder-only transformers, Gemma-3 introduces several modern efficiency-focused innovations, notably around its attention mechanisms, positional encodings, and feed-forward networks.

| Feature | NanoGPT (Standard GPT-2) | Gemma-3 (270M) |
| :--- | :--- | :--- |
| **Parameters** | 124M (typically) | ~436M Total (~270M excluding embeddings) |
| **Layers** | 12 | 18 |
| **Hidden Size ($d_{model}$)** | 768 | 640 |
| **Vocab Size** | 50,257 | 262,144 |
| **Positional Encoding** | Absolute (Learned) | Rotary (RoPE) |
| **Attention Mechanism** | Multi-Head Attention (MHA) | Multi-Query Attention (MQA) + Interleaved Local/Global |
| **Normalization** | LayerNorm | RMSNorm + QK-Norm |
| **MLP Structure** | Standard 2-Layer with GELU | Gated MLP (gate, up, down) with GELU |

---

## Detailed Architectural Differences

### 1. Positional Encoding
*   **NanoGPT:** Uses **Learned Absolute Positional Embeddings**, added directly to token embeddings at the first layer. This ties the model to the context length it was trained on and doesn't explicitly encode relative distances well.
*   **Gemma-3:** Employs **Rotary Positional Embedding (RoPE)**. Rather than being added to the initial embeddings, RoPE is applied dynamically at each layer by rotating the Query and Key vectors in the attention mechanism.
    *   *Implementation Note for Verilog GPU:* Gemma-3 uses a dynamic frequency base depending on the layer type (10k for local sliding window layers, 1M for global layers).

### 2. Attention Mechanism
*   **NanoGPT:** Standard **Multi-Head Attention (MHA)**. Each head has its own Query, Key, and Value projections. Computes global attention across the entire context window.
*   **Gemma-3:** Introduces several optimizations to reduce memory bandwidth (a key bottleneck on custom hardware):
    *   **Multi-Query Attention (MQA):** Gemma-3 270M uses 4 Query heads but only 1 shared Key/Value head. This drastically reduces the memory footprint of the KV cache during inference.
    *   **Interleaved Local/Global Attention:** Uses a 5:1 ratio of local sliding-window attention (window size of 512 tokens) to full global attention. E.g., 15 sliding window layers and 3 global layers.
    *   **QK-Norm:** Applies RMSNorm to both Query and Key tensors *before* the dot product to stabilize training, an extra step not present in NanoGPT.

### 3. MLP / Feed-Forward Network
*   **NanoGPT:** Uses a traditional 2-layer MLP. An input projection expands the dimension by 4x, applies GELU, and a second projection reduces it back to $d_{model}$.
*   **Gemma-3:** Uses a **Gated MLP** (similar to LLaMA and earlier Gemma models).
    *   Instead of a simple projection, it uses three linear layers: `gate_proj`, `up_proj`, and `down_proj`.
    *   The formula is typically: `down_proj(Activation(gate_proj(x)) * up_proj(x))`.
    *   **Activation:** specifically `gelu_pytorch_tanh` (an approximation of GELU).
    *   **Dimensions:** For the 270M model, the intermediate size is 2048.

### 4. Normalization
*   **NanoGPT:** Uses standard **LayerNorm**, which computes both variance and mean, and learns a scale and bias.
*   **Gemma-3:** Uses **RMSNorm** (Root Mean Square Normalization), which omits the mean centering step, saving computation without sacrificing performance. It also uses QK-Norm inside the attention blocks.

## Impact on Custom GPU Implementation

To support Gemma-3 270M on the custom Verilog GPU, several modules will need updates or complete rewrites:
1.  **RoPE Calculation Unit:** We need a hardware block capable of applying rotary transformations to Q and K tensors dynamically, potentially handling different base frequencies.
2.  **Attention Controller:** Must be upgraded from standard MHA to support MQA (broadcasting 1 KV head to 4 Q heads) and handle the sliding window logic for local layers.
3.  **RMSNorm Unit:** Needs to be implemented alongside (or replacing) the existing LayerNorm block. It's computationally cheaper but must be placed properly (including QK-Norm).
4.  **Gated MLP Datapath:** The MLP pipeline must support the parallel `gate_proj` and `up_proj` matrix multiplications, element-wise multiplication, and then the final `down_proj`.