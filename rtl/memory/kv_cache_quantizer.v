`timescale 1ns / 1ps

// ============================================================================
// Module: kv_cache_quantizer
// Description: KV Cache INT4 Quantization / Dequantization Engine.
//
//   PAPER: "QuantSpec: Self-Speculative Decoding with Hierarchical Quantized 
//          KV Cache" (Apple, ICML 2025)
//
//   RATIONALE: The KV cache is the #1 memory bottleneck in LLM inference.
//   For long sequences (2048+ tokens), it can consume GBs of memory.
//   Quantizing KV values from 16-bit to 4-bit gives 4× memory savings
//   with minimal quality loss (<1% perplexity increase).
//
//   WHY THIS MATTERS FOR BITBYBIT:
//   - Our PagedAttention MMU manages KV cache pages
//   - Compressing each page 4× means 4× more context fits in memory
//   - Combined with GQA: 4× (GQA) × 4× (INT4) = 16× total KV reduction!
//   - Enables running larger models on the same FPGA resources
//
//   OPERATION:
//   Quantize: Find per-group min/max → scale = (max-min)/15 → q = (v-min)/scale
//   Dequant:  v_approx = q × scale + min
//
// Parameters: VEC_LEN, DATA_WIDTH
// ============================================================================
module kv_cache_quantizer #(
    parameter VEC_LEN    = 4,       // Values per group
    parameter DATA_WIDTH = 16      // Input precision (Q8.8)
)(
    input  wire                           clk,
    input  wire                           rst,
    
    // Quantize: 16-bit → 4-bit + metadata
    input  wire                           quant_valid,
    input  wire [VEC_LEN*DATA_WIDTH-1:0]  kv_in,
    output reg  [VEC_LEN*4-1:0]           kv_quantized,    // 4 bits per value
    output reg  signed [DATA_WIDTH-1:0]   quant_min,        // Per-group minimum
    output reg  [DATA_WIDTH-1:0]          quant_scale,      // Per-group scale
    output reg                            quant_done,
    
    // Dequantize: 4-bit + metadata → 16-bit
    input  wire                           dequant_valid,
    input  wire [VEC_LEN*4-1:0]           kv_q_in,
    input  wire signed [DATA_WIDTH-1:0]   dequant_min,
    input  wire [DATA_WIDTH-1:0]          dequant_scale,
    output reg  [VEC_LEN*DATA_WIDTH-1:0]  kv_dequantized,
    output reg                            dequant_done,
    
    // Stats
    output reg  [31:0]                    bytes_saved
);

    integer i;
    reg signed [DATA_WIDTH-1:0] val;
    reg signed [DATA_WIDTH-1:0] vmin, vmax;
    reg [DATA_WIDTH-1:0] range;
    reg [DATA_WIDTH-1:0] scale_factor;
    reg signed [2*DATA_WIDTH-1:0] shifted;
    reg [3:0] quantized_val;

    always @(posedge clk) begin
        if (rst) begin
            kv_quantized   <= 0;
            kv_dequantized <= 0;
            quant_min      <= 0;
            quant_scale    <= 0;
            quant_done     <= 1'b0;
            dequant_done   <= 1'b0;
            bytes_saved    <= 32'd0;
        end else begin
            quant_done  <= 1'b0;
            dequant_done <= 1'b0;
            
            // ---- QUANTIZATION: 16-bit → 4-bit ----
            if (quant_valid) begin
                // Step 1: Find min and max
                vmin = $signed(kv_in[0 +: DATA_WIDTH]);
                vmax = vmin;
                for (i = 1; i < VEC_LEN; i = i + 1) begin
                    val = $signed(kv_in[i*DATA_WIDTH +: DATA_WIDTH]);
                    if (val < vmin) vmin = val;
                    if (val > vmax) vmax = val;
                end
                
                // Step 2: Compute range and scale
                range = vmax - vmin;
                // scale = range / 15 ≈ range >> 4 (for synthesizability)
                scale_factor = (range > 0) ? ((range + 7) >> 4) : 16'd1;
                if (scale_factor == 0) scale_factor = 16'd1;
                
                quant_min   <= vmin;
                quant_scale <= scale_factor;
                
                // Step 3: Quantize each value to 4-bit [0..15]
                for (i = 0; i < VEC_LEN; i = i + 1) begin
                    val = $signed(kv_in[i*DATA_WIDTH +: DATA_WIDTH]);
                    shifted = val - vmin;
                    // q = (val - min) / scale ≈ shifted >> log2(scale)
                    // Use shift-based approximation
                    if (scale_factor >= 16'd128)
                        quantized_val = shifted[DATA_WIDTH-1:DATA_WIDTH-4];
                    else if (scale_factor >= 16'd64)
                        quantized_val = shifted[DATA_WIDTH-2:DATA_WIDTH-5];
                    else if (scale_factor >= 16'd32)
                        quantized_val = shifted[DATA_WIDTH-3:DATA_WIDTH-6];
                    else if (scale_factor >= 16'd16)
                        quantized_val = shifted[DATA_WIDTH-4:DATA_WIDTH-7];
                    else if (scale_factor >= 16'd8)
                        quantized_val = shifted[DATA_WIDTH-5:DATA_WIDTH-8];
                    else if (scale_factor >= 16'd4)
                        quantized_val = shifted[3:0];
                    else if (scale_factor >= 16'd2)
                        quantized_val = shifted[3:0];
                    else
                        quantized_val = shifted[3:0];
                    
                    // Clamp to [0, 15]
                    if (quantized_val > 4'd15) quantized_val = 4'd15;
                    kv_quantized[i*4 +: 4] <= quantized_val;
                end
                
                quant_done  <= 1'b1;
                bytes_saved <= bytes_saved + VEC_LEN; // Saved VEC_LEN bytes (16→4 bit)
            end
            
            // ---- DEQUANTIZATION: 4-bit → 16-bit ----
            if (dequant_valid) begin
                for (i = 0; i < VEC_LEN; i = i + 1) begin
                    // v_approx = q × scale + min
                    kv_dequantized[i*DATA_WIDTH +: DATA_WIDTH] <= 
                        dequant_min + ($signed({12'd0, kv_q_in[i*4 +: 4]}) * $signed({1'b0, dequant_scale}));
                end
                dequant_done <= 1'b1;
            end
        end
    end

endmodule
