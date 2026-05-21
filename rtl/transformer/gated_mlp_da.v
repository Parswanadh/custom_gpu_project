`timescale 1ns / 1ps
// ============================================================================
// Module: gated_mlp_da
// Description: Dual-Lane Asymmetrical Gated MLP prototype.
//   - Lane 1 (Gate): Ternary weights with zero-skip logic.
//   - Lane 2 (Up): INT8 weights.
//   - Lane 3 (Down): INT8 weights.
//   - Integrates gelu_lut_256 on the Gate output before multiplying by Up path.
//   - Outputs 16-bit Q8.8 format matching the Python golden model.
// ============================================================================

module gated_mlp_da (
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output reg  done,

    // Input X ROM
    output reg  [4:0] x_addr,
    input  wire signed [15:0] x_data,

    // Gate Weights ROM
    output reg  [6:0] gw_addr,
    input  wire signed [7:0] gw_data,

    // Up Weights ROM
    output reg  [6:0] uw_addr,
    input  wire signed [7:0] uw_data,

    // Down Weights ROM
    output reg  [6:0] dw_addr,
    input  wire signed [7:0] dw_data,

    // Output RAM
    output wire [4:0] out_addr,
    output wire signed [15:0] out_data,
    output wire out_we
);

    // FSM States
    localparam IDLE         = 4'd0;
    localparam INIT_N       = 4'd1;
    localparam INIT_H       = 4'd2;
    localparam INIT_I       = 4'd3;
    localparam MAC_HI_ADDR  = 4'd4;
    localparam MAC_HI_ACC   = 4'd5;
    localparam GELU_STORE   = 4'd6;
    localparam INIT_O       = 4'd7;
    localparam INIT_H2      = 4'd8;
    localparam MAC_OUT_ADDR = 4'd9;
    localparam MAC_OUT_ACC  = 4'd10;
    localparam WRITE_OUT    = 4'd11;
    localparam NEXT_N       = 4'd12;
    localparam DONE         = 4'd13;

    reg [3:0] state;

    // Loop counters
    reg [2:0] n; // 0 to 3
    reg [4:0] h; // 0 to 15
    reg [3:0] i; // 0 to 7
    reg [3:0] o; // 0 to 7

    // Accumulators
    reg signed [31:0] gate_acc;
    reg signed [31:0] up_acc;
    reg signed [47:0] out_acc;

    // Hidden Buffer
    reg signed [31:0] hidden_buf [0:15];

    // Clamping function to Q8.8 (16 bits)
    function signed [15:0] clamp16;
        input signed [47:0] val;
        begin
            if (val > 48'sd32767) clamp16 = 16'h8000;
            else if (val < -48'sd32768) clamp16 = -16'sd32768;
            else clamp16 = val[15:0];
        end
    endfunction

    // ------------------------------------------------------------------------
    // Datapath Components
    // ------------------------------------------------------------------------

    // Lane 1 (Gate): Ternary Multiplexer
    wire gate_is_zero = (gw_data == 8'sd0);
    wire signed [15:0] gate_mux_out = (gw_data == 8'sd1) ? x_data :
                                      (gw_data == -8'sd1) ? -x_data : 16'sd0;

    // Zero-Skip enable for Lane 1 clock gating
    wire gate_acc_en = !gate_is_zero;

    // Lane 2 (Up): Product
    wire signed [23:0] up_mult_out = x_data * uw_data;

    // Fix: Explicitly clamp the accumulator intermediate values
    wire signed [31:0] gate_acc_clamped = (gate_acc > 32'sd32767) ? 32'sd32767 : (gate_acc < -32'sd32768) ? -32'sd32768 : gate_acc;
    wire signed [31:0] up_acc_clamped   = (up_acc > 32'sd32767) ? 32'sd32767 : (up_acc < -32'sd32768) ? -32'sd32768 : up_acc;

    // GELU Instantiation
    wire signed [15:0] gate_acc_16 = gate_acc_clamped[15:0];
    wire signed [15:0] gelu_out;
    
    gelu_lut_256 gelu_inst (
        .x_in(gate_acc_16),
        .gelu_out(gelu_out)
    );

    // Gated Up multiplication
    wire signed [47:0] gated_up_full = gelu_out * up_acc_clamped;

    // Down multiplication
    wire signed [39:0] down_mult_out = hidden_buf[h] * dw_data;

    // Final Output scaling
    wire signed [47:0] out_acc_rounded = (out_acc + 48'sd128) >>> 8;

    // Output assignments
    assign out_we = (state == WRITE_OUT);
    assign out_addr = (n << 3) + o;
    assign out_data = clamp16(out_acc_rounded);

    // ------------------------------------------------------------------------
    // State Machine
    // ------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 0;
            n <= 0; h <= 0; i <= 0; o <= 0;
            gate_acc <= 0; up_acc <= 0; out_acc <= 0;
            x_addr <= 0; gw_addr <= 0; uw_addr <= 0; dw_addr <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) state <= INIT_N;
                end
                INIT_N: begin
                    n <= 0;
                    state <= INIT_H;
                end
                INIT_H: begin
                    h <= 0;
                    state <= INIT_I;
                end
                INIT_I: begin
                    i <= 0;
                    gate_acc <= 0;
                    up_acc <= 0;
                    state <= MAC_HI_ADDR;
                end
                MAC_HI_ADDR: begin
                    x_addr <= (n << 3) + i;
                    gw_addr <= (i << 4) + h;
                    uw_addr <= (i << 4) + h;
                    state <= MAC_HI_ACC;
                end
                MAC_HI_ACC: begin
                    // Zero-skip logic applied using enable
                    if (gate_acc_en) begin
                        gate_acc <= gate_acc + gate_mux_out;
                    end
                    up_acc <= up_acc + up_mult_out;
                    
                    if (i == 4'd7) begin
                        state <= GELU_STORE;
                    end else begin
                        i <= i + 1;
                        state <= MAC_HI_ADDR;
                    end
                end
                GELU_STORE: begin
                    // Store Q16.16 intermediate result
                    hidden_buf[h] <= gated_up_full;
                    
                    if (h == 5'd15) begin
                        state <= INIT_O;
                    end else begin
                        h <= h + 1;
                        state <= INIT_I;
                    end
                end
                INIT_O: begin
                    o <= 0;
                    state <= INIT_H2;
                end
                INIT_H2: begin
                    h <= 0;
                    out_acc <= 0;
                    state <= MAC_OUT_ADDR;
                end
                MAC_OUT_ADDR: begin
                    dw_addr <= (h << 3) + o;
                    state <= MAC_OUT_ACC;
                end
                MAC_OUT_ACC: begin
                    out_acc <= out_acc + down_mult_out;
                    
                    if (h == 5'd15) begin
                        state <= WRITE_OUT;
                    end else begin
                        h <= h + 1;
                        state <= MAC_OUT_ADDR;
                    end
                end
                WRITE_OUT: begin
                    // Combinational writes are active here.
                    if (o == 4'd7) begin
                        state <= NEXT_N;
                    end else begin
                        o <= o + 1;
                        state <= INIT_H2;
                    end
                end
                NEXT_N: begin
                    if (n == 3'd3) begin
                        done <= 1;
                        state <= IDLE;
                    end else begin
                        n <= n + 1;
                        state <= INIT_H;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
