`timescale 1ns/1ps

module rope_unit_v2 #(
    parameter DATA_WIDTH = 16,
    parameter LUT_ADDR_WIDTH = 10,
    parameter LUT_DATA_WIDTH = 32,
    parameter LUT_DEPTH = 1024,
    parameter FRAC_BITS = 14
)(
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        enable,
    
    // Query and Key Input Pairs
    input  wire signed [DATA_WIDTH-1:0] q_in_0,
    input  wire signed [DATA_WIDTH-1:0] q_in_1,
    input  wire signed [DATA_WIDTH-1:0] k_in_0,
    input  wire signed [DATA_WIDTH-1:0] k_in_1,
    
    // Address for LUT
    input  wire [LUT_ADDR_WIDTH-1:0]   sin_addr,
    input  wire [LUT_ADDR_WIDTH-1:0]   cos_addr,

    // LUT Write interface (Initialization)
    input  wire                        lut_we,
    input  wire [LUT_ADDR_WIDTH-1:0]   lut_waddr,
    input  wire [LUT_DATA_WIDTH-1:0]   lut_din,
    
    // Output Pairs
    output reg signed [DATA_WIDTH-1:0] q_out_0,
    output reg signed [DATA_WIDTH-1:0] q_out_1,
    output reg signed [DATA_WIDTH-1:0] k_out_0,
    output reg signed [DATA_WIDTH-1:0] k_out_1,
    output reg                         valid_out
);

    // Dual-Port LUT instantiation
    wire [LUT_DATA_WIDTH-1:0] dout_sin;
    wire [LUT_DATA_WIDTH-1:0] dout_cos;
    
    dual_port_lut #(
        .ADDR_WIDTH(LUT_ADDR_WIDTH),
        .DATA_WIDTH(LUT_DATA_WIDTH),
        .DEPTH(LUT_DEPTH)
    ) trig_lut (
        .clk(clk),
        .we(lut_we),
        .waddr(lut_waddr),
        .din(lut_din),
        .raddr1(sin_addr),
        .dout1(dout_sin),
        .raddr2(cos_addr),
        .dout2(dout_cos)
    );

    // Extract lower bits for Sine/Cosine assuming they fit in DATA_WIDTH
    wire signed [DATA_WIDTH-1:0] sin_val = dout_sin[DATA_WIDTH-1:0];
    wire signed [DATA_WIDTH-1:0] cos_val = dout_cos[DATA_WIDTH-1:0];

    // Pipeline stage to sync with LUT read latency (1 cycle)
    reg signed [DATA_WIDTH-1:0] q_in_0_r, q_in_1_r;
    reg signed [DATA_WIDTH-1:0] k_in_0_r, k_in_1_r;
    reg valid_r;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_in_0_r <= 0;
            q_in_1_r <= 0;
            k_in_0_r <= 0;
            k_in_1_r <= 0;
            valid_r  <= 0;
        end else begin
            valid_r <= enable;
            q_in_0_r <= q_in_0;
            q_in_1_r <= q_in_1;
            k_in_0_r <= k_in_0;
            k_in_1_r <= k_in_1;
        end
    end

    // Compute rotations (1 cycle latency)
    wire signed [(2*DATA_WIDTH)-1:0] q0_cos = q_in_0_r * cos_val;
    wire signed [(2*DATA_WIDTH)-1:0] q1_sin = q_in_1_r * sin_val;
    wire signed [(2*DATA_WIDTH)-1:0] q1_cos = q_in_1_r * cos_val;
    wire signed [(2*DATA_WIDTH)-1:0] q0_sin = q_in_0_r * sin_val;

    wire signed [(2*DATA_WIDTH)-1:0] k0_cos = k_in_0_r * cos_val;
    wire signed [(2*DATA_WIDTH)-1:0] k1_sin = k_in_1_r * sin_val;
    wire signed [(2*DATA_WIDTH)-1:0] k1_cos = k_in_1_r * cos_val;
    wire signed [(2*DATA_WIDTH)-1:0] k0_sin = k_in_0_r * sin_val;

    // Rounding constant for correct bit shifting
    wire signed [(2*DATA_WIDTH)-1:0] round_const = (1 << (FRAC_BITS - 1));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q_out_0   <= 0;
            q_out_1   <= 0;
            k_out_0   <= 0;
            k_out_1   <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= valid_r;
            q_out_0 <= (q0_cos - q1_sin + round_const) >>> FRAC_BITS;
            q_out_1 <= (q1_cos + q0_sin + round_const) >>> FRAC_BITS;
            k_out_0 <= (k0_cos - k1_sin + round_const) >>> FRAC_BITS;
            k_out_1 <= (k1_cos + k0_sin + round_const) >>> FRAC_BITS;
        end
    end

endmodule
