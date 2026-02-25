// ============================================================================
// Module: int4_pack_unit
// Description: INT4 packing/unpacking unit for 4x parallel processing.
//   Packs 4 INT4 values (4-bit each) into a single 16-bit word.
//   Unpacks a 16-bit word into 4 separate INT4 values.
//   When combined with variable_precision_alu in mode 00,
//   enables 4 multiplications per cycle instead of 1.
//
// Example:
//   Pack:   {w3, w2, w1, w0} → 16'h{w3}{w2}{w1}{w0}
//   Unpack: 16'h{w3}{w2}{w1}{w0} → {w3, w2, w1, w0}
// ============================================================================
module int4_pack_unit (
    input  wire        clk,
    input  wire        rst,
    // Pack mode: 4 separate INT4 → 1 packed 16-bit
    input  wire        pack_valid,
    input  wire [3:0]  in_val0,
    input  wire [3:0]  in_val1,
    input  wire [3:0]  in_val2,
    input  wire [3:0]  in_val3,
    output reg  [15:0] packed_out,
    output reg         pack_valid_out,
    // Unpack mode: 1 packed 16-bit → 4 separate INT4
    input  wire        unpack_valid,
    input  wire [15:0] packed_in,
    output reg  [3:0]  out_val0,
    output reg  [3:0]  out_val1,
    output reg  [3:0]  out_val2,
    output reg  [3:0]  out_val3,
    output reg         unpack_valid_out,
    // Zero detection for all 4 lanes
    output wire [3:0]  zero_mask       // 1 = this lane is zero
);

    // Pack: combine 4 INT4 values into 16-bit word
    always @(posedge clk) begin
        if (rst) begin
            packed_out     <= 16'd0;
            pack_valid_out <= 1'b0;
        end else if (pack_valid) begin
            packed_out     <= {in_val3, in_val2, in_val1, in_val0};
            pack_valid_out <= 1'b1;
        end else begin
            pack_valid_out <= 1'b0;
        end
    end

    // Unpack: split 16-bit word into 4 INT4 values
    always @(posedge clk) begin
        if (rst) begin
            out_val0         <= 4'd0;
            out_val1         <= 4'd0;
            out_val2         <= 4'd0;
            out_val3         <= 4'd0;
            unpack_valid_out <= 1'b0;
        end else if (unpack_valid) begin
            out_val0         <= packed_in[3:0];
            out_val1         <= packed_in[7:4];
            out_val2         <= packed_in[11:8];
            out_val3         <= packed_in[15:12];
            unpack_valid_out <= 1'b1;
        end else begin
            unpack_valid_out <= 1'b0;
        end
    end

    // Per-lane zero detection (combinational)
    assign zero_mask[0] = (packed_in[3:0]   == 4'd0);
    assign zero_mask[1] = (packed_in[7:4]   == 4'd0);
    assign zero_mask[2] = (packed_in[11:8]  == 4'd0);
    assign zero_mask[3] = (packed_in[15:12] == 4'd0);

endmodule
