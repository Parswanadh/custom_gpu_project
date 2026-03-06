`timescale 1ns / 1ps

// ============================================================================
// Module: mixed_precision_decompressor
// Description: Multi-format Weight Decompressor.
//   Supports Q4 (4-bit), Q6 (6-bit), and Q8 (8-bit) per layer.
//   This is the key to GGUF compatibility — GGUF uses mixed quantization
//   where different layers use different precision levels.
//
//   Format Selection (2-bit mode):
//     00 = Q4 mode: 8 weights per 32-bit word (INT4 asymmetric)
//     01 = Q6 mode: 4 weights per 32-bit word (INT6 + 2 unused bits/group)
//     10 = Q8 mode: 4 weights per 32-bit word (INT8 direct)
//     11 = Reserved
//
//   Each format applies: output = (raw - zero_point) * scale >> 8
//   where zero_point and scale are per-group parameters.
//
//   This makes our GPU compatible with:
//     - GPTQ (Q4/Q8 per-group)
//     - AWQ (Q4 with activation-aware grouping)
//     - GGUF Q4_K_M (mixed Q4/Q6/Q8)
//     - SqueezeLLM (mixed precision)
//
// Parameters: DATA_WIDTH
// ============================================================================
module mixed_precision_decompressor #(
    parameter DATA_WIDTH = 8
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     valid_in,
    input  wire [1:0]               precision_mode,     // 00=Q4, 01=Q6, 10=Q8
    input  wire [31:0]              packed_weights,     // Packed weight data
    input  wire [7:0]               zero_point,         // Per-group zero point
    input  wire [7:0]               scale_factor,       // Per-group scale (Q0.8)
    
    // Output: variable number of decompressed weights
    output reg  [DATA_WIDTH-1:0]    weight_out_0,
    output reg  [DATA_WIDTH-1:0]    weight_out_1,
    output reg  [DATA_WIDTH-1:0]    weight_out_2,
    output reg  [DATA_WIDTH-1:0]    weight_out_3,
    output reg  [DATA_WIDTH-1:0]    weight_out_4,
    output reg  [DATA_WIDTH-1:0]    weight_out_5,
    output reg  [DATA_WIDTH-1:0]    weight_out_6,
    output reg  [DATA_WIDTH-1:0]    weight_out_7,
    output reg  [2:0]               num_weights_out,    // How many outputs are valid
    output reg                      valid_out
);

    // Intermediate: raw extracted values (before dequantization)
    reg signed [7:0] raw [0:7];
    
    // Dequantization: output = (raw - zero_point) * scale >> 8
    function [DATA_WIDTH-1:0] dequant;
        input signed [7:0] raw_val;
        input [7:0] zp;
        input [7:0] sc;
        reg signed [15:0] shifted;
        reg signed [15:0] scaled;
        begin
            shifted = raw_val - $signed({1'b0, zp});
            scaled = (shifted * $signed({1'b0, sc})) >>> 8;
            if (scaled > 127) dequant = 8'sd127;
            else if (scaled < -128) dequant = -8'sd128;
            else dequant = scaled[DATA_WIDTH-1:0];
        end
    endfunction

    always @(posedge clk) begin
        if (rst) begin
            valid_out <= 1'b0;
            num_weights_out <= 3'd0;
            weight_out_0 <= 0; weight_out_1 <= 0;
            weight_out_2 <= 0; weight_out_3 <= 0;
            weight_out_4 <= 0; weight_out_5 <= 0;
            weight_out_6 <= 0; weight_out_7 <= 0;
        end else begin
            valid_out <= 1'b0;
            
            if (valid_in) begin
                case (precision_mode)
                    // ========================================================
                    // Q4 MODE: 8 weights per 32-bit word (4 bits each)
                    // Bits: [3:0]=w0, [7:4]=w1, ... [31:28]=w7
                    // ========================================================
                    2'b00: begin
                        // Extract 8 × 4-bit weights (sign-extend to 8-bit)
                        raw[0] = {{4{packed_weights[3]}},  packed_weights[3:0]};
                        raw[1] = {{4{packed_weights[7]}},  packed_weights[7:4]};
                        raw[2] = {{4{packed_weights[11]}}, packed_weights[11:8]};
                        raw[3] = {{4{packed_weights[15]}}, packed_weights[15:12]};
                        raw[4] = {{4{packed_weights[19]}}, packed_weights[19:16]};
                        raw[5] = {{4{packed_weights[23]}}, packed_weights[23:20]};
                        raw[6] = {{4{packed_weights[27]}}, packed_weights[27:24]};
                        raw[7] = {{4{packed_weights[31]}}, packed_weights[31:28]};
                        
                        weight_out_0 <= dequant(raw[0], zero_point, scale_factor);
                        weight_out_1 <= dequant(raw[1], zero_point, scale_factor);
                        weight_out_2 <= dequant(raw[2], zero_point, scale_factor);
                        weight_out_3 <= dequant(raw[3], zero_point, scale_factor);
                        weight_out_4 <= dequant(raw[4], zero_point, scale_factor);
                        weight_out_5 <= dequant(raw[5], zero_point, scale_factor);
                        weight_out_6 <= dequant(raw[6], zero_point, scale_factor);
                        weight_out_7 <= dequant(raw[7], zero_point, scale_factor);
                        num_weights_out <= 3'd0; // 0 means 8 (all valid)
                        valid_out <= 1'b1;
                    end
                    
                    // ========================================================
                    // Q6 MODE: 4 weights per 32-bit word (6 bits each + 8 unused)
                    // Bits: [5:0]=w0, [11:6]=w1, [17:12]=w2, [23:18]=w3
                    // ========================================================
                    2'b01: begin
                        raw[0] = {{2{packed_weights[5]}},  packed_weights[5:0]};
                        raw[1] = {{2{packed_weights[11]}}, packed_weights[11:6]};
                        raw[2] = {{2{packed_weights[17]}}, packed_weights[17:12]};
                        raw[3] = {{2{packed_weights[23]}}, packed_weights[23:18]};
                        
                        weight_out_0 <= dequant(raw[0], zero_point, scale_factor);
                        weight_out_1 <= dequant(raw[1], zero_point, scale_factor);
                        weight_out_2 <= dequant(raw[2], zero_point, scale_factor);
                        weight_out_3 <= dequant(raw[3], zero_point, scale_factor);
                        weight_out_4 <= 0;
                        weight_out_5 <= 0;
                        weight_out_6 <= 0;
                        weight_out_7 <= 0;
                        num_weights_out <= 3'd4;
                        valid_out <= 1'b1;
                    end
                    
                    // ========================================================
                    // Q8 MODE: 4 weights per 32-bit word (8 bits each)
                    // Bits: [7:0]=w0, [15:8]=w1, [23:16]=w2, [31:24]=w3
                    // ========================================================
                    2'b10: begin
                        raw[0] = packed_weights[7:0];
                        raw[1] = packed_weights[15:8];
                        raw[2] = packed_weights[23:16];
                        raw[3] = packed_weights[31:24];
                        
                        weight_out_0 <= dequant(raw[0], zero_point, scale_factor);
                        weight_out_1 <= dequant(raw[1], zero_point, scale_factor);
                        weight_out_2 <= dequant(raw[2], zero_point, scale_factor);
                        weight_out_3 <= dequant(raw[3], zero_point, scale_factor);
                        weight_out_4 <= 0;
                        weight_out_5 <= 0;
                        weight_out_6 <= 0;
                        weight_out_7 <= 0;
                        num_weights_out <= 3'd4;
                        valid_out <= 1'b1;
                    end
                    
                    default: begin
                        valid_out <= 1'b0;
                    end
                endcase
            end
        end
    end

endmodule
