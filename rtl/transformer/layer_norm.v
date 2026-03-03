// ============================================================================
// Module: layer_norm
// Description: Layer normalization for transformer blocks.
//   Computes: y[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
//   All in Q8.8 fixed-point arithmetic.
//   Sequential: accumulate sum → mean → variance → normalize
//
//   FIXES APPLIED:
//     - Issue #3: inv_sqrt via 256-entry LUT (replaces 5-bucket step function)
//     - Issue #5: Division replaced with shift for power-of-2 DIM
//
// Parameters: DIM (embedding dimension, must be power of 2)
// ============================================================================
module layer_norm #(
    parameter DIM        = 4,
    parameter DATA_WIDTH = 16,
    parameter DIM_LOG2   = $clog2(DIM)   // Auto-computed shift amount
)(
    input  wire                           clk,
    input  wire                           rst,
    input  wire                           valid_in,
    input  wire [DIM*DATA_WIDTH-1:0]      x_in,
    input  wire [DIM*DATA_WIDTH-1:0]      gamma_in,
    input  wire [DIM*DATA_WIDTH-1:0]      beta_in,
    output reg  [DIM*DATA_WIDTH-1:0]      y_out,
    output reg                            valid_out
);

    // Internal storage
    reg signed [DATA_WIDTH-1:0] x_buf  [0:DIM-1];
    reg signed [DATA_WIDTH-1:0] gamma  [0:DIM-1];
    reg signed [DATA_WIDTH-1:0] beta   [0:DIM-1];

    reg signed [DATA_WIDTH+7:0]   sum_acc;
    reg signed [DATA_WIDTH-1:0]   mean_val;
    reg signed [2*DATA_WIDTH-1:0] var_acc;
    reg signed [DATA_WIDTH-1:0]   var_val;
    reg signed [DATA_WIDTH-1:0]   inv_std;
    reg signed [DATA_WIDTH-1:0]   diff;
    reg signed [2*DATA_WIDTH-1:0] norm_val;
    reg signed [2*DATA_WIDTH-1:0] scaled;

    integer i;
    reg [3:0] state;
    localparam IDLE        = 4'd0;
    localparam CALC_MEAN   = 4'd1;
    localparam CALC_VAR    = 4'd2;
    localparam CALC_INVSTD = 4'd3;
    localparam NORMALIZE   = 4'd4;
    localparam OUTPUT      = 4'd5;

    reg [$clog2(DIM):0] idx;

    // Issue #3: Use 256-entry inv_sqrt LUT instead of 5-bucket function
    wire [15:0] inv_sqrt_var_input;
    wire signed [15:0] inv_sqrt_result;

    assign inv_sqrt_var_input = (var_val < 0) ? 16'd0 : var_val;

    inv_sqrt_lut_256 u_inv_sqrt (
        .var_in(inv_sqrt_var_input),
        .inv_sqrt_out(inv_sqrt_result)
    );

    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            valid_out <= 1'b0;
            sum_acc   <= 0;
            var_acc   <= 0;
            idx       <= 0;
            y_out     <= 0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        for (i = 0; i < DIM; i = i + 1) begin
                            x_buf[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                            gamma[i] <= gamma_in[i*DATA_WIDTH +: DATA_WIDTH];
                            beta[i]  <= beta_in[i*DATA_WIDTH +: DATA_WIDTH];
                        end
                        sum_acc <= 0;
                        idx <= 0;
                        state <= CALC_MEAN;
                    end
                end

                CALC_MEAN: begin
                    if (idx < DIM) begin
                        sum_acc <= sum_acc + {{8{x_buf[idx][DATA_WIDTH-1]}}, x_buf[idx]};
                        idx <= idx + 1;
                    end else begin
                        // Issue #5: Replace division with arithmetic right shift
                        // mean = sum / DIM = sum >>> log2(DIM) for power-of-2 DIM
                        mean_val <= sum_acc >>> DIM_LOG2;
                        var_acc <= 0;
                        idx <= 0;
                        state <= CALC_VAR;
                    end
                end

                CALC_VAR: begin
                    if (idx < DIM) begin
                        diff = x_buf[idx] - mean_val;
                        var_acc <= var_acc + (diff * diff);
                        idx <= idx + 1;
                    end else begin
                        // Issue #5: Replace division with shift
                        // var = var_acc / DIM → var_acc >>> log2(DIM)
                        // Q16.16 → Q8.8: take bits [DATA_WIDTH+7:8], then shift
                        var_val <= var_acc[DATA_WIDTH+7:8] >>> DIM_LOG2;
                        state <= CALC_INVSTD;
                    end
                end

                CALC_INVSTD: begin
                    // Issue #3: Use LUT result (combinational, available this cycle)
                    inv_std <= inv_sqrt_result[DATA_WIDTH-1:0];
                    idx <= 0;
                    state <= NORMALIZE;
                end

                NORMALIZE: begin
                    if (idx < DIM) begin
                        diff = x_buf[idx] - mean_val;
                        norm_val = diff * inv_std;

                        scaled = gamma[idx] * norm_val[DATA_WIDTH+7:8];
                        y_out[idx*DATA_WIDTH +: DATA_WIDTH] <= scaled[DATA_WIDTH+7:8] + beta[idx];
                        idx <= idx + 1;
                    end else begin
                        state <= OUTPUT;
                    end
                end

                OUTPUT: begin
                    valid_out <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
