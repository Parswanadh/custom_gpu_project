// ============================================================================
// Module: layer_norm
// Description: Layer normalization for transformer blocks.
//   Computes: y[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
//   All in Q8.8 fixed-point arithmetic.
//   Sequential: accumulate sum → mean → variance → normalize
// Parameters: DIM (embedding dimension)
// ============================================================================
module layer_norm #(
    parameter DIM        = 4,     // Embedding dimension
    parameter DATA_WIDTH = 16     // Q8.8 fixed-point
)(
    input  wire                           clk,
    input  wire                           rst,
    input  wire                           valid_in,
    input  wire [DIM*DATA_WIDTH-1:0]      x_in,       // Input vector
    input  wire [DIM*DATA_WIDTH-1:0]      gamma_in,   // Scale parameters
    input  wire [DIM*DATA_WIDTH-1:0]      beta_in,    // Bias parameters
    output reg  [DIM*DATA_WIDTH-1:0]      y_out,      // Normalized output
    output reg                            valid_out
);

    // Internal storage
    reg signed [DATA_WIDTH-1:0] x_buf  [0:DIM-1];
    reg signed [DATA_WIDTH-1:0] gamma  [0:DIM-1];
    reg signed [DATA_WIDTH-1:0] beta   [0:DIM-1];

    reg signed [DATA_WIDTH+7:0]   sum_acc;   // Accumulator for sum
    reg signed [DATA_WIDTH-1:0]   mean_val;
    reg signed [2*DATA_WIDTH-1:0] var_acc;   // Accumulator for variance
    reg signed [DATA_WIDTH-1:0]   var_val;
    reg signed [DATA_WIDTH-1:0]   inv_std;   // 1/sqrt(var+eps)
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

    // Approximate 1/sqrt(x) using iterative method
    // For Q8.8: if var = V, then 1/sqrt(V) ≈ lookup + 1 Newton iteration
    // Simplified: use reciprocal approximation
    function signed [DATA_WIDTH-1:0] approx_inv_sqrt;
        input signed [DATA_WIDTH-1:0] val;
        reg [DATA_WIDTH-1:0] abs_val;
        begin
            abs_val = (val < 0) ? -val : val;
            if (abs_val == 0)
                approx_inv_sqrt = 16'sd256;  // 1.0 in Q8.8
            else if (abs_val <= 16'sd64)      // var <= 0.25
                approx_inv_sqrt = 16'sd512;  // ~2.0
            else if (abs_val <= 16'sd256)     // var <= 1.0
                approx_inv_sqrt = 16'sd256;  // ~1.0
            else if (abs_val <= 16'sd1024)    // var <= 4.0
                approx_inv_sqrt = 16'sd128;  // ~0.5
            else
                approx_inv_sqrt = 16'sd64;   // ~0.25
        end
    endfunction

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
                        // Unpack inputs
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
                        // mean = sum / DIM (use shift for power-of-2 DIM)
                        mean_val <= sum_acc / DIM;
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
                        // var = var_acc / DIM (Q16.16 → Q8.8)
                        var_val <= var_acc[DATA_WIDTH+7:8] / DIM;
                        state <= CALC_INVSTD;
                    end
                end

                CALC_INVSTD: begin
                    inv_std <= approx_inv_sqrt(var_val);
                    idx <= 0;
                    state <= NORMALIZE;
                end

                NORMALIZE: begin
                    if (idx < DIM) begin
                        // norm = (x - mean) * inv_std
                        diff = x_buf[idx] - mean_val;
                        norm_val = diff * inv_std;  // Q8.8 * Q8.8 = Q16.16

                        // y = gamma * norm + beta (Q8.8)
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
