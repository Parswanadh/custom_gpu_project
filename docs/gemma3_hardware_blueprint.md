# Gemma-3 Hardware Blueprint

This document outlines the Verilog architectural blueprint for implementing the key components of the Gemma-3 270M model on the BitbyBit custom Q8.8 fixed-point pipelined hardware.

## 1. RMSNorm (`rmsnorm.v`)

RMSNorm replaces LayerNorm in Gemma-3. It removes the mean-centering step, reducing the number of synchronization points and computational cost.

### Input/Output Ports
```verilog
module rmsnorm #(
    parameter SEQ_LEN = 32,
    parameter HIDDEN_DIM = 640
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire signed [15:0] x_in [0:HIDDEN_DIM-1],      // Q8.8 format input vector
    input wire signed [15:0] weight_in [0:HIDDEN_DIM-1], // Q8.8 format learned weights
    output reg signed [15:0] x_out [0:HIDDEN_DIM-1],     // Q8.8 format output vector
    output reg done
);
```

### Internal Dataflow (Pipelined)
1. **Square and Accumulate:** Square each input element ($x_i^2$) and accumulate them to compute the sum of squares. This can be pipelined using an adder reduction tree.
2. **Mean of Squares:** Divide the sum by `HIDDEN_DIM` (implemented as a Q8.8 multiplication with a precomputed inverse).
3. **Inverse Square Root:** Pass the mean of squares into a look-up table (e.g., `inv_sqrt_lut_256.v`) to compute $1 / \sqrt{\text{mean} + \epsilon}$.
4. **Scale and Weight:** Multiply each original input element $x_i$ by the inverse square root, then multiply by the learned `weight_in` for that channel.

## 2. Rotary Positional Embedding (`rope_unit.v`)

RoPE dynamically rotates Query and Key vectors. Gemma-3 uses different base frequencies depending on the attention type (10k for local, 1M for global layers).

### Input/Output Ports
```verilog
module rope_unit #(
    parameter HEAD_DIM = 64
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [15:0] pos_id,                        // Current token position
    input wire is_global_layer,                      // 0 = Local (Base 10k), 1 = Global (Base 1M)
    input wire signed [15:0] q_in [0:HEAD_DIM-1],    // Q8.8 format Query head
    input wire signed [15:0] k_in [0:HEAD_DIM-1],    // Q8.8 format Key head
    output reg signed [15:0] q_out [0:HEAD_DIM-1],
    output reg signed [15:0] k_out [0:HEAD_DIM-1],
    output reg done
);
```

### Internal Dataflow (Pipelined)
1. **Angle Generation:** Based on `pos_id` and the frequency base (determined by `is_global_layer`), calculate the rotation angle $\theta_i$ for each pair of dimensions in the head.
2. **Trig LUTs:** Feed the angles into Sine and Cosine Look-Up Tables (LUTs) to obtain Q8.8 fixed-point trigonometric values.
3. **Complex Rotation:** For each adjacent pair ($x_{2i}, x_{2i+1}$) in Q and K:
   - $x_{out\_2i} = (x_{2i} \times \cos(\theta_i)) - (x_{2i+1} \times \sin(\theta_i))$
   - $x_{out\_2i+1} = (x_{2i} \times \sin(\theta_i)) + (x_{2i+1} \times \cos(\theta_i))$
   - Utilizing standard Q8.8 multipliers and adders.

## 3. Gated MLP (`gated_mlp.v`)

Gemma-3 replaces the standard 2-layer FFN with a 3-layer Gated MLP using a specialized GELU activation.

### Input/Output Ports
```verilog
module gated_mlp #(
    parameter HIDDEN_DIM = 640,
    parameter INTERMEDIATE_DIM = 2048
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire signed [15:0] act_in [0:HIDDEN_DIM-1],    // Q8.8 format activation input
    // Note: Weights would stream from SRAM in reality, arrays shown here for simplicity
    input wire signed [15:0] weight_gate [0:HIDDEN_DIM-1][0:INTERMEDIATE_DIM-1],
    input wire signed [15:0] weight_up [0:HIDDEN_DIM-1][0:INTERMEDIATE_DIM-1],
    input wire signed [15:0] weight_down [0:INTERMEDIATE_DIM-1][0:HIDDEN_DIM-1],
    output reg signed [15:0] act_out [0:HIDDEN_DIM-1],   // Q8.8 format output
    output reg done
);
```

### Internal Dataflow (Pipelined)
1. **Parallel Projections:** Feed `act_in` into two parallel Matrix-Vector Multiplication (MVM) pipelines:
   - **Pipeline A:** Computes `gate_out = act_in * weight_gate`.
   - **Pipeline B:** Computes `up_out = act_in * weight_up`.
2. **Activation:** Route `gate_out` through a dedicated LUT (`gelu_pytorch_tanh_lut.v`, adapted from standard GELU) to compute the specific Gemma-3 activation function.
3. **Element-wise Gating:** Perform element-wise Q8.8 multiplication: `gated_val = activation(gate_out) * up_out`.
4. **Down Projection:** Feed `gated_val` into a third MVM pipeline to compute the final output: `act_out = gated_val * weight_down`.