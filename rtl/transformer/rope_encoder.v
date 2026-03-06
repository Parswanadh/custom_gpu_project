`timescale 1ns / 1ps

// ============================================================================
// Module: rope_encoder
// Description: Rotary Positional Encoding (RoPE) Hardware Engine.
//
//   PAPER: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
//          (Su et al., 2021)
//
//   RATIONALE: RoPE is used by ALL modern LLMs:
//     - Llama 1/2/3 (Meta)
//     - Mistral/Mixtral (Mistral AI)
//     - Qwen (Alibaba)
//     - GPT-NeoX (EleutherAI)
//   Without RoPE, the model has NO position awareness — it can't tell
//   if "The cat sat on the mat" vs "mat the on sat cat The".
//
//   WHY HARDWARE: RoPE applies rotations to Q and K vectors EVERY token.
//   A dedicated hardware unit with precomputed sin/cos LUTs makes this
//   single-cycle instead of multi-cycle software computation.
//
//   MATH:
//     Q_rot[2i]   = Q[2i]   × cos(θ_i × pos) - Q[2i+1] × sin(θ_i × pos)
//     Q_rot[2i+1] = Q[2i]   × sin(θ_i × pos) + Q[2i+1] × cos(θ_i × pos)
//   where θ_i = 10000^(-2i/d_model) — precomputed per dimension pair
//
//   IMPLEMENTATION:
//     - 64-entry sin + cos LUT (covers positions 0-63 for 6-bit address)
//     - Processes dimension pairs sequentially (2 dims per cycle)
//     - Q8.8 fixed-point arithmetic throughout
//
// Parameters: DIM, DATA_WIDTH, MAX_POS
// ============================================================================
module rope_encoder #(
    parameter DIM        = 8,       // Embedding dimension (must be even)
    parameter DATA_WIDTH = 16,      // Q8.8 fixed-point
    parameter MAX_POS    = 64       // Maximum sequence position
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     valid_in,
    input  wire [$clog2(MAX_POS)-1:0] position,   // Token position in sequence
    input  wire [DIM*DATA_WIDTH-1:0]  q_in,       // Query vector
    input  wire [DIM*DATA_WIDTH-1:0]  k_in,       // Key vector
    
    output reg  [DIM*DATA_WIDTH-1:0]  q_rot,      // Rotated query
    output reg  [DIM*DATA_WIDTH-1:0]  k_rot,      // Rotated key
    output reg                        valid_out
);

    // Sin/Cos LUT — 64 entries, Q8.8 fixed-point
    // These represent sin(pos × θ_i) and cos(pos × θ_i)
    // For simplicity, θ = pos × (2π/64) = pos × π/32
    // In Q8.8: cos(0)=256, sin(0)=0, cos(π/4)=181, sin(π/4)=181
    reg signed [DATA_WIDTH-1:0] cos_lut [0:63];
    reg signed [DATA_WIDTH-1:0] sin_lut [0:63];
    
    // Initialize LUTs with precomputed values (Q8.8 format)
    // cos(n × 2π/64) × 256 and sin(n × 2π/64) × 256
    integer init_i;
    initial begin
        // Quadrant 1: 0 to π/2
        cos_lut[0]  = 16'sd256;  sin_lut[0]  = 16'sd0;
        cos_lut[1]  = 16'sd255;  sin_lut[1]  = 16'sd25;
        cos_lut[2]  = 16'sd251;  sin_lut[2]  = 16'sd50;
        cos_lut[3]  = 16'sd245;  sin_lut[3]  = 16'sd74;
        cos_lut[4]  = 16'sd236;  sin_lut[4]  = 16'sd98;
        cos_lut[5]  = 16'sd225;  sin_lut[5]  = 16'sd120;
        cos_lut[6]  = 16'sd212;  sin_lut[6]  = 16'sd142;
        cos_lut[7]  = 16'sd197;  sin_lut[7]  = 16'sd162;
        cos_lut[8]  = 16'sd181;  sin_lut[8]  = 16'sd181;
        cos_lut[9]  = 16'sd162;  sin_lut[9]  = 16'sd197;
        cos_lut[10] = 16'sd142;  sin_lut[10] = 16'sd212;
        cos_lut[11] = 16'sd120;  sin_lut[11] = 16'sd225;
        cos_lut[12] = 16'sd98;   sin_lut[12] = 16'sd236;
        cos_lut[13] = 16'sd74;   sin_lut[13] = 16'sd245;
        cos_lut[14] = 16'sd50;   sin_lut[14] = 16'sd251;
        cos_lut[15] = 16'sd25;   sin_lut[15] = 16'sd255;
        // Quadrant 2: π/2 to π
        cos_lut[16] = 16'sd0;    sin_lut[16] = 16'sd256;
        cos_lut[17] = -16'sd25;  sin_lut[17] = 16'sd255;
        cos_lut[18] = -16'sd50;  sin_lut[18] = 16'sd251;
        cos_lut[19] = -16'sd74;  sin_lut[19] = 16'sd245;
        cos_lut[20] = -16'sd98;  sin_lut[20] = 16'sd236;
        cos_lut[21] = -16'sd120; sin_lut[21] = 16'sd225;
        cos_lut[22] = -16'sd142; sin_lut[22] = 16'sd212;
        cos_lut[23] = -16'sd162; sin_lut[23] = 16'sd197;
        cos_lut[24] = -16'sd181; sin_lut[24] = 16'sd181;
        cos_lut[25] = -16'sd197; sin_lut[25] = 16'sd162;
        cos_lut[26] = -16'sd212; sin_lut[26] = 16'sd142;
        cos_lut[27] = -16'sd225; sin_lut[27] = 16'sd120;
        cos_lut[28] = -16'sd236; sin_lut[28] = 16'sd98;
        cos_lut[29] = -16'sd245; sin_lut[29] = 16'sd74;
        cos_lut[30] = -16'sd251; sin_lut[30] = 16'sd50;
        cos_lut[31] = -16'sd255; sin_lut[31] = 16'sd25;
        // Quadrant 3: π to 3π/2
        cos_lut[32] = -16'sd256; sin_lut[32] = 16'sd0;
        cos_lut[33] = -16'sd255; sin_lut[33] = -16'sd25;
        cos_lut[34] = -16'sd251; sin_lut[34] = -16'sd50;
        cos_lut[35] = -16'sd245; sin_lut[35] = -16'sd74;
        cos_lut[36] = -16'sd236; sin_lut[36] = -16'sd98;
        cos_lut[37] = -16'sd225; sin_lut[37] = -16'sd120;
        cos_lut[38] = -16'sd212; sin_lut[38] = -16'sd142;
        cos_lut[39] = -16'sd197; sin_lut[39] = -16'sd162;
        cos_lut[40] = -16'sd181; sin_lut[40] = -16'sd181;
        cos_lut[41] = -16'sd162; sin_lut[41] = -16'sd197;
        cos_lut[42] = -16'sd142; sin_lut[42] = -16'sd212;
        cos_lut[43] = -16'sd120; sin_lut[43] = -16'sd225;
        cos_lut[44] = -16'sd98;  sin_lut[44] = -16'sd236;
        cos_lut[45] = -16'sd74;  sin_lut[45] = -16'sd245;
        cos_lut[46] = -16'sd50;  sin_lut[46] = -16'sd251;
        cos_lut[47] = -16'sd25;  sin_lut[47] = -16'sd255;
        // Quadrant 4: 3π/2 to 2π
        cos_lut[48] = 16'sd0;    sin_lut[48] = -16'sd256;
        cos_lut[49] = 16'sd25;   sin_lut[49] = -16'sd255;
        cos_lut[50] = 16'sd50;   sin_lut[50] = -16'sd251;
        cos_lut[51] = 16'sd74;   sin_lut[51] = -16'sd245;
        cos_lut[52] = 16'sd98;   sin_lut[52] = -16'sd236;
        cos_lut[53] = 16'sd120;  sin_lut[53] = -16'sd225;
        cos_lut[54] = 16'sd142;  sin_lut[54] = -16'sd212;
        cos_lut[55] = 16'sd162;  sin_lut[55] = -16'sd197;
        cos_lut[56] = 16'sd181;  sin_lut[56] = -16'sd181;
        cos_lut[57] = 16'sd197;  sin_lut[57] = -16'sd162;
        cos_lut[58] = 16'sd212;  sin_lut[58] = -16'sd142;
        cos_lut[59] = 16'sd225;  sin_lut[59] = -16'sd120;
        cos_lut[60] = 16'sd236;  sin_lut[60] = -16'sd98;
        cos_lut[61] = 16'sd245;  sin_lut[61] = -16'sd74;
        cos_lut[62] = 16'sd251;  sin_lut[62] = -16'sd50;
        cos_lut[63] = 16'sd255;  sin_lut[63] = -16'sd25;
    end

    // FSM
    reg [2:0] state;
    localparam IDLE     = 3'd0;
    localparam ROTATE   = 3'd1;
    localparam DONE_ST  = 3'd2;
    
    reg [$clog2(DIM/2):0] pair_idx;  // Current dimension pair
    
    // Working registers
    reg signed [DATA_WIDTH-1:0] q_even, q_odd, k_even, k_odd;
    reg signed [DATA_WIDTH-1:0] cos_val, sin_val;
    reg signed [2*DATA_WIDTH-1:0] prod1, prod2;
    
    // LUT address: combine position and dimension index
    // θ_i × pos = (pos * (i+1)) mod 64 — simplified frequency mapping
    wire [5:0] lut_addr = (position * (pair_idx + 1)) & 6'h3F;

    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            valid_out <= 1'b0;
            q_rot     <= 0;
            k_rot     <= 0;
            pair_idx  <= 0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        pair_idx <= 0;
                        state <= ROTATE;
                    end
                end
                
                ROTATE: begin
                    // Extract dimension pair from Q and K
                    q_even = $signed(q_in[(pair_idx*2)*DATA_WIDTH +: DATA_WIDTH]);
                    q_odd  = $signed(q_in[(pair_idx*2+1)*DATA_WIDTH +: DATA_WIDTH]);
                    k_even = $signed(k_in[(pair_idx*2)*DATA_WIDTH +: DATA_WIDTH]);
                    k_odd  = $signed(k_in[(pair_idx*2+1)*DATA_WIDTH +: DATA_WIDTH]);
                    
                    // Get sin/cos for this position + dimension
                    cos_val = cos_lut[lut_addr];
                    sin_val = sin_lut[lut_addr];
                    
                    // Apply rotation to Q:
                    // Q_rot[2i]   = Q[2i] * cos - Q[2i+1] * sin
                    // Q_rot[2i+1] = Q[2i] * sin + Q[2i+1] * cos
                    prod1 = q_even * cos_val - q_odd * sin_val;
                    prod2 = q_even * sin_val + q_odd * cos_val;
                    q_rot[(pair_idx*2)*DATA_WIDTH +: DATA_WIDTH]   <= prod1 >>> 8;
                    q_rot[(pair_idx*2+1)*DATA_WIDTH +: DATA_WIDTH] <= prod2 >>> 8;
                    
                    // Apply same rotation to K:
                    prod1 = k_even * cos_val - k_odd * sin_val;
                    prod2 = k_even * sin_val + k_odd * cos_val;
                    k_rot[(pair_idx*2)*DATA_WIDTH +: DATA_WIDTH]   <= prod1 >>> 8;
                    k_rot[(pair_idx*2+1)*DATA_WIDTH +: DATA_WIDTH] <= prod2 >>> 8;
                    
                    if (pair_idx == DIM/2 - 1)
                        state <= DONE_ST;
                    else
                        pair_idx <= pair_idx + 1;
                end
                
                DONE_ST: begin
                    valid_out <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
