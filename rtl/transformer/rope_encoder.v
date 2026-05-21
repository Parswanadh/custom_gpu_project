`timescale 1ns / 1ps

// ============================================================================
// Module: rope_encoder
// Description: Rotary Positional Encoding (RoPE) Hardware Engine.
//   Upgraded to use Dual-Port Synchronous LUTs for SOTA Edge AI efficiency.
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

    // Sin/Cos Synchronous Dual-Port LUTs
    wire signed [DATA_WIDTH-1:0] cos_val, sin_val;
    wire [5:0] lut_addr;
    
    // Address and control logic
    reg [$clog2(DIM/2):0] pair_idx;
    assign lut_addr = (position * (pair_idx + 1)) & 6'h3F;

    // LUT Instantiations
    // Port 2 is unused here but available for future parallel head access.
    dual_port_lut #(.ADDR_WIDTH(6), .DATA_WIDTH(DATA_WIDTH), .DEPTH(64)) u_cos_lut (
        .clk(clk), .we(1'b0), .waddr(6'd0), .din(16'd0),
        .raddr1(lut_addr), .dout1(cos_val),
        .raddr2(6'd0), .dout2()
    );

    dual_port_lut #(.ADDR_WIDTH(6), .DATA_WIDTH(DATA_WIDTH), .DEPTH(64)) u_sin_lut (
        .clk(clk), .we(1'b0), .waddr(6'd0), .din(16'd0),
        .raddr1(lut_addr), .dout1(sin_val),
        .raddr2(6'd0), .dout2()
    );

    // Initial values for LUTs (Copied from original)
    initial begin
        // These are written to the 'mem' inside u_cos_lut and u_sin_lut via hierarchical paths for simulation.
        // In a real synthesis, $readmemh or a ROM primitive would be used.
        u_cos_lut.mem[0]  = 16'sd256;  u_sin_lut.mem[0]  = 16'sd0;
        u_cos_lut.mem[1]  = 16'sd255;  u_sin_lut.mem[1]  = 16'sd25;
        u_cos_lut.mem[2]  = 16'sd251;  u_sin_lut.mem[2]  = 16'sd50;
        u_cos_lut.mem[3]  = 16'sd245;  u_sin_lut.mem[3]  = 16'sd74;
        u_cos_lut.mem[4]  = 16'sd236;  u_sin_lut.mem[4]  = 16'sd98;
        u_cos_lut.mem[5]  = 16'sd225;  u_sin_lut.mem[5]  = 16'sd120;
        u_cos_lut.mem[6]  = 16'sd212;  u_sin_lut.mem[6]  = 16'sd142;
        u_cos_lut.mem[7]  = 16'sd197;  u_sin_lut.mem[7]  = 16'sd162;
        u_cos_lut.mem[8]  = 16'sd181;  u_sin_lut.mem[8]  = 16'sd181;
        u_cos_lut.mem[9]  = 16'sd162;  u_sin_lut.mem[9]  = 16'sd197;
        u_cos_lut.mem[10] = 16'sd142;  u_sin_lut.mem[10] = 16'sd212;
        u_cos_lut.mem[11] = 16'sd120;  u_sin_lut.mem[11] = 16'sd225;
        u_cos_lut.mem[12] = 16'sd98;   u_sin_lut.mem[12] = 16'sd236;
        u_cos_lut.mem[13] = 16'sd74;   u_sin_lut.mem[13] = 16'sd245;
        u_cos_lut.mem[14] = 16'sd50;   u_sin_lut.mem[14] = 16'sd251;
        u_cos_lut.mem[15] = 16'sd25;   u_sin_lut.mem[15] = 16'sd255;
        u_cos_lut.mem[16] = 16'sd0;    u_sin_lut.mem[16] = 16'sd256;
        u_cos_lut.mem[17] = -16'sd25;  u_sin_lut.mem[17] = 16'sd255;
        u_cos_lut.mem[18] = -16'sd50;  u_sin_lut.mem[18] = 16'sd251;
        u_cos_lut.mem[19] = -16'sd74;  u_sin_lut.mem[19] = 16'sd245;
        u_cos_lut.mem[20] = -16'sd98;  u_sin_lut.mem[20] = 16'sd236;
        u_cos_lut.mem[21] = -16'sd120; u_sin_lut.mem[21] = 16'sd225;
        u_cos_lut.mem[22] = -16'sd142; u_sin_lut.mem[22] = 16'sd212;
        u_cos_lut.mem[23] = -16'sd162; u_sin_lut.mem[23] = 16'sd197;
        u_cos_lut.mem[24] = -16'sd181; u_sin_lut.mem[24] = 16'sd181;
        u_cos_lut.mem[25] = -16'sd197; u_sin_lut.mem[25] = 16'sd162;
        u_cos_lut.mem[26] = -16'sd212; u_sin_lut.mem[26] = 16'sd142;
        u_cos_lut.mem[27] = -16'sd225; u_sin_lut.mem[27] = 16'sd120;
        u_cos_lut.mem[28] = -16'sd236; u_sin_lut.mem[28] = 16'sd98;
        u_cos_lut.mem[29] = -16'sd245; u_sin_lut.mem[29] = 16'sd74;
        u_cos_lut.mem[30] = -16'sd251; u_sin_lut.mem[30] = 16'sd50;
        u_cos_lut.mem[31] = -16'sd255; u_sin_lut.mem[31] = 16'sd25;
        u_cos_lut.mem[32] = -16'sd256; u_sin_lut.mem[32] = 16'sd0;
        u_cos_lut.mem[33] = -16'sd255; u_sin_lut.mem[33] = -16'sd25;
        u_cos_lut.mem[34] = -16'sd251; u_sin_lut.mem[34] = -16'sd50;
        u_cos_lut.mem[35] = -16'sd245; u_sin_lut.mem[35] = -16'sd74;
        u_cos_lut.mem[36] = -16'sd236; u_sin_lut.mem[36] = -16'sd98;
        u_cos_lut.mem[37] = -16'sd225; u_sin_lut.mem[37] = -16'sd120;
        u_cos_lut.mem[38] = -16'sd212; u_sin_lut.mem[38] = -16'sd142;
        u_cos_lut.mem[39] = -16'sd197; u_sin_lut.mem[39] = -16'sd162;
        u_cos_lut.mem[40] = -16'sd181; u_sin_lut.mem[40] = -16'sd181;
        u_cos_lut.mem[41] = -16'sd162; u_sin_lut.mem[41] = -16'sd197;
        u_cos_lut.mem[42] = -16'sd142; u_sin_lut.mem[42] = -16'sd212;
        u_cos_lut.mem[43] = -16'sd120; u_sin_lut.mem[43] = -16'sd225;
        u_cos_lut.mem[44] = -16'sd98;  u_sin_lut.mem[44] = -16'sd236;
        u_cos_lut.mem[45] = -16'sd74;  u_sin_lut.mem[45] = -16'sd245;
        u_cos_lut.mem[46] = -16'sd50;  u_sin_lut.mem[46] = -16'sd251;
        u_cos_lut.mem[47] = -16'sd25;  u_sin_lut.mem[47] = -16'sd255;
        u_cos_lut.mem[48] = 16'sd0;    u_sin_lut.mem[48] = -16'sd256;
        u_cos_lut.mem[49] = 16'sd25;   u_sin_lut.mem[49] = -16'sd255;
        u_cos_lut.mem[50] = 16'sd50;   u_sin_lut.mem[50] = -16'sd251;
        u_cos_lut.mem[51] = 16'sd74;   u_sin_lut.mem[51] = -16'sd245;
        u_cos_lut.mem[52] = 16'sd98;   u_sin_lut.mem[52] = -16'sd236;
        u_cos_lut.mem[53] = 16'sd120;  u_sin_lut.mem[53] = -16'sd225;
        u_cos_lut.mem[54] = 16'sd142;  u_sin_lut.mem[54] = -16'sd212;
        u_cos_lut.mem[55] = 16'sd162;  u_sin_lut.mem[55] = -16'sd197;
        u_cos_lut.mem[56] = 16'sd181;  u_sin_lut.mem[56] = -16'sd181;
        u_cos_lut.mem[57] = 16'sd197;  u_sin_lut.mem[57] = -16'sd162;
        u_cos_lut.mem[58] = 16'sd212;  u_sin_lut.mem[58] = -16'sd142;
        u_cos_lut.mem[59] = 16'sd225;  u_sin_lut.mem[59] = -16'sd120;
        u_cos_lut.mem[60] = 16'sd236;  u_sin_lut.mem[60] = -16'sd98;
        u_cos_lut.mem[61] = 16'sd245;  u_sin_lut.mem[61] = -16'sd74;
        u_cos_lut.mem[62] = 16'sd251;  u_sin_lut.mem[62] = -16'sd50;
        u_cos_lut.mem[63] = 16'sd255;  u_sin_lut.mem[63] = -16'sd25;
    end

    // FSM
    reg [2:0] state;
    localparam IDLE     = 3'd0;
    localparam PREFETCH = 3'd1;
    localparam ROTATE   = 3'd2;
    localparam DONE_ST  = 3'd3;
    
    // Working registers
    reg signed [DATA_WIDTH-1:0] q_even, q_odd, k_even, k_odd;
    reg signed [2*DATA_WIDTH-1:0] prod1, prod2;
    
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
                        state <= PREFETCH;
                    end
                end
                
                PREFETCH: begin
                    // Address 'lut_addr' is already presented to dual_port_lut.
                    // Data will be available on the next cycle (ROTATE state).
                    state <= ROTATE;
                end

                ROTATE: begin
                    // Sample dimension pair from Q and K
                    q_even = $signed(q_in[(pair_idx*2)*DATA_WIDTH +: DATA_WIDTH]);
                    q_odd  = $signed(q_in[(pair_idx*2+1)*DATA_WIDTH +: DATA_WIDTH]);
                    k_even = $signed(k_in[(pair_idx*2)*DATA_WIDTH +: DATA_WIDTH]);
                    k_odd  = $signed(k_in[(pair_idx*2+1)*DATA_WIDTH +: DATA_WIDTH]);
                    
                    // Apply rotation to Q using synchronous LUT outputs:
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
                    else begin
                        pair_idx <= pair_idx + 1;
                        state <= PREFETCH; // Need another cycle to prefetch next sine/cosine
                    end
                end
                
                DONE_ST: begin
                    valid_out <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
