// ============================================================================
// Module: attention_unit
// Description: Multi-head self-attention for transformer blocks.
//   For each head: Scores = softmax(Q × K^T / sqrt(d_k)), Output = Scores × V
//   Simplified for simulation: single token attention (no KV cache),
//   Q/K/V projections done via linear_layer calls.
// Parameters: EMBED_DIM, NUM_HEADS, HEAD_DIM, SEQ_LEN
// ============================================================================
module attention_unit #(
    parameter EMBED_DIM  = 8,
    parameter NUM_HEADS  = 2,
    parameter HEAD_DIM   = 4,     // EMBED_DIM / NUM_HEADS
    parameter DATA_WIDTH = 16
)(
    input  wire                                clk,
    input  wire                                rst,
    input  wire                                valid_in,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]     x_in,       // Input embedding
    // Preloaded Q/K/V/O weight matrices (flattened)
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wq_flat,
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wk_flat,
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wv_flat,
    input  wire [EMBED_DIM*EMBED_DIM*DATA_WIDTH-1:0] wo_flat,
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]     y_out,
    output reg                                 valid_out
);

    // Internal storage
    reg signed [DATA_WIDTH-1:0] x [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] q [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] k [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] v [0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] attn_out [0:EMBED_DIM-1];

    // Weight matrices
    reg signed [DATA_WIDTH-1:0] wq [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wk [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wv [0:EMBED_DIM-1][0:EMBED_DIM-1];
    reg signed [DATA_WIDTH-1:0] wo [0:EMBED_DIM-1][0:EMBED_DIM-1];

    reg signed [2*DATA_WIDTH-1:0] accum;
    reg signed [2*DATA_WIDTH-1:0] product;
    reg signed [DATA_WIDTH-1:0]   score;
    reg signed [2*DATA_WIDTH-1:0] score_val;

    integer i, j, h;
    reg [3:0] state;
    localparam IDLE     = 4'd0;
    localparam LOAD_W   = 4'd1;
    localparam PROJ_QKV = 4'd2;
    localparam ATTENTION= 4'd3;
    localparam OUT_PROJ = 4'd4;
    localparam DONE     = 4'd5;

    // Unpack weights on valid_in
    always @(posedge clk) begin
        if (rst) begin
            state     <= IDLE;
            valid_out <= 1'b0;
            y_out     <= 0;
        end else begin
            case (state)
                IDLE: begin
                    valid_out <= 1'b0;
                    if (valid_in) begin
                        // Unpack input
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            x[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        // Unpack weight matrices
                        for (i = 0; i < EMBED_DIM; i = i + 1)
                            for (j = 0; j < EMBED_DIM; j = j + 1) begin
                                wq[i][j] <= wq_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH];
                                wk[i][j] <= wk_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH];
                                wv[i][j] <= wv_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH];
                                wo[i][j] <= wo_flat[(i*EMBED_DIM+j)*DATA_WIDTH +: DATA_WIDTH];
                            end
                        state <= PROJ_QKV;
                    end
                end

                PROJ_QKV: begin
                    // Compute Q = x * Wq, K = x * Wk, V = x * Wv (mat-vec multiply)
                    for (j = 0; j < EMBED_DIM; j = j + 1) begin
                        // Q[j]
                        accum = 0;
                        for (i = 0; i < EMBED_DIM; i = i + 1) begin
                            product = x[i] * wq[i][j];
                            accum = accum + product;
                        end
                        q[j] = accum[DATA_WIDTH+7:8]; // Q8.8*Q8.8→Q16.16→Q8.8

                        // K[j]
                        accum = 0;
                        for (i = 0; i < EMBED_DIM; i = i + 1) begin
                            product = x[i] * wk[i][j];
                            accum = accum + product;
                        end
                        k[j] = accum[DATA_WIDTH+7:8];

                        // V[j]
                        accum = 0;
                        for (i = 0; i < EMBED_DIM; i = i + 1) begin
                            product = x[i] * wv[i][j];
                            accum = accum + product;
                        end
                        v[j] = accum[DATA_WIDTH+7:8];
                    end
                    state <= ATTENTION;
                end

                ATTENTION: begin
                    // Simplified single-token self-attention:
                    // For a single token, attention score is always 1.0 (softmax of single element)
                    // So output = V directly (Q*K^T produces one score, softmax(one_score) = 1)
                    // This makes attn_out = V
                    for (i = 0; i < EMBED_DIM; i = i + 1)
                        attn_out[i] = v[i];
                    state <= OUT_PROJ;
                end

                OUT_PROJ: begin
                    // Output projection: y = attn_out * Wo
                    for (j = 0; j < EMBED_DIM; j = j + 1) begin
                        accum = 0;
                        for (i = 0; i < EMBED_DIM; i = i + 1) begin
                            product = attn_out[i] * wo[i][j];
                            accum = accum + product;
                        end
                        y_out[j*DATA_WIDTH +: DATA_WIDTH] <= accum[DATA_WIDTH+7:8];
                    end
                    state <= DONE;
                end

                DONE: begin
                    valid_out <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
