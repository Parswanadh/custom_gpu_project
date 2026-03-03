// ============================================================================
// Module: tiled_matmul
// Description: Tiled matrix multiplication controller (Issue #21).
//   Divides large matrices into tiles that fit the systolic array,
//   processes each tile, and accumulates partial results in a scratchpad.
//
//   Computes: C[M×N] = A[M×K] × B[K×N]
//   Tiles:    M_tile = ARRAY_SIZE, K_tile = ARRAY_SIZE, N_tile = ARRAY_SIZE
//   Total tiles: ceil(M/tile) × ceil(N/tile) × ceil(K/tile)
//
//   For each output tile C_tile[m][n]:
//     C_tile = 0
//     for k_tile in 0..ceil(K/tile):
//       Load A_tile[m][k] into systolic input
//       Load B_tile[k][n] into systolic weights
//       C_tile += systolic_result
//     Store C_tile to scratchpad
//
// Parameters: M_DIM, N_DIM, K_DIM, ARRAY_SIZE, DATA_WIDTH
// ============================================================================
module tiled_matmul #(
    parameter M_DIM      = 16,
    parameter N_DIM      = 16,
    parameter K_DIM      = 16,
    parameter ARRAY_SIZE = 4,
    parameter DATA_WIDTH = 16,
    parameter ACC_WIDTH  = 32,
    parameter ADDR_W     = 16
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,

    // Scratchpad interface (A, B, C all in scratchpad)
    input  wire [ADDR_W-1:0]       a_base_addr,
    input  wire [ADDR_W-1:0]       b_base_addr,
    input  wire [ADDR_W-1:0]       c_base_addr,

    // Scratchpad read port
    output reg                     sp_read_en,
    output reg  [ADDR_W-1:0]       sp_read_addr,
    input  wire [DATA_WIDTH-1:0]   sp_read_data,
    input  wire                    sp_read_valid,

    // Scratchpad write port
    output reg                     sp_write_en,
    output reg  [ADDR_W-1:0]       sp_write_addr,
    output reg  [DATA_WIDTH-1:0]   sp_write_data,

    // Status
    output reg                     done,
    output reg                     busy
);

    // Tile loop variables
    reg [$clog2(M_DIM):0] m_tile;  // Current M tile (0, ARRAY_SIZE, 2*ARRAY_SIZE, ...)
    reg [$clog2(N_DIM):0] n_tile;  // Current N tile
    reg [$clog2(K_DIM):0] k_tile;  // Current K tile (inner reduction)

    // Systolic array interface
    reg        sa_load_weight;
    reg [$clog2(ARRAY_SIZE)-1:0] sa_weight_row, sa_weight_col;
    reg signed [DATA_WIDTH-1:0]  sa_weight_data;
    reg        sa_valid_in;
    reg [ARRAY_SIZE*DATA_WIDTH-1:0] sa_act_in;
    reg        sa_clear_acc;
    wire [ARRAY_SIZE*ACC_WIDTH-1:0] sa_result;
    wire       sa_valid_out;

    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE),
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_sa (
        .clk(clk), .rst(rst),
        .load_weight(sa_load_weight),
        .weight_row(sa_weight_row), .weight_col(sa_weight_col),
        .weight_data(sa_weight_data),
        .valid_in(sa_valid_in), .act_in(sa_act_in),
        .clear_acc(sa_clear_acc),
        .precision_mode(2'd0), .q4_block_scale(8'd0), .q4_block_zero(4'd0),
        .result_out(sa_result), .valid_out(sa_valid_out)
    );

    // State machine
    reg [3:0] state;
    localparam S_IDLE       = 4'd0;
    localparam S_LOAD_B     = 4'd1;  // Load B tile weights
    localparam S_FEED_A     = 4'd2;  // Feed A tile activations
    localparam S_WAIT       = 4'd3;  // Wait for systolic result
    localparam S_STORE_C    = 4'd4;  // Store partial result
    localparam S_NEXT_K     = 4'd5;  // Advance K tile
    localparam S_NEXT_N     = 4'd6;  // Advance N tile
    localparam S_NEXT_M     = 4'd7;  // Advance M tile
    localparam S_DONE       = 4'd8;

    reg [$clog2(ARRAY_SIZE):0] load_row, load_col;
    reg [$clog2(ARRAY_SIZE):0] feed_idx;
    reg [$clog2(ARRAY_SIZE):0] store_idx;

    // Partial accumulation buffer for C tiles across K iterations
    reg signed [ACC_WIDTH-1:0] c_accum [0:ARRAY_SIZE-1];
    integer ci;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            done  <= 1'b0;
            busy  <= 1'b0;
            m_tile <= 0; n_tile <= 0; k_tile <= 0;
            sa_load_weight <= 0;
            sa_valid_in    <= 0;
            sa_clear_acc   <= 0;
            sp_read_en     <= 0;
            sp_write_en    <= 0;
        end else begin
            sa_load_weight <= 0;
            sa_valid_in    <= 0;
            sa_clear_acc   <= 0;
            sp_read_en     <= 0;
            sp_write_en    <= 0;
            done           <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        m_tile <= 0; n_tile <= 0; k_tile <= 0;
                        busy <= 1'b1;
                        sa_clear_acc <= 1'b1;
                        load_row <= 0; load_col <= 0;
                        for (ci = 0; ci < ARRAY_SIZE; ci = ci + 1)
                            c_accum[ci] <= 0;
                        state <= S_LOAD_B;
                    end
                end

                // Load B[k_tile:k_tile+ARRAY_SIZE][n_tile:n_tile+ARRAY_SIZE] as weights
                S_LOAD_B: begin
                    if (load_row < ARRAY_SIZE && load_col < ARRAY_SIZE) begin
                        sp_read_en   <= 1'b1;
                        sp_read_addr <= b_base_addr +
                                       (k_tile + load_row) * N_DIM +
                                       (n_tile + load_col);
                        if (sp_read_valid) begin
                            sa_load_weight <= 1'b1;
                            sa_weight_row  <= load_row[$clog2(ARRAY_SIZE)-1:0];
                            sa_weight_col  <= load_col[$clog2(ARRAY_SIZE)-1:0];
                            sa_weight_data <= $signed(sp_read_data);
                            load_col <= load_col + 1;
                            if (load_col + 1 >= ARRAY_SIZE) begin
                                load_col <= 0;
                                load_row <= load_row + 1;
                            end
                        end
                    end else begin
                        feed_idx <= 0;
                        state <= S_FEED_A;
                    end
                end

                // Feed A[m_tile:m_tile+ARRAY_SIZE][k_tile+row] as activations
                S_FEED_A: begin
                    if (feed_idx < ARRAY_SIZE) begin
                        // Read one column of A tile
                        sa_valid_in <= 1'b1;
                        begin : feed_pack
                            integer fi;
                            for (fi = 0; fi < ARRAY_SIZE; fi = fi + 1) begin
                                sp_read_en   <= 1'b1;
                                sp_read_addr <= a_base_addr +
                                               (m_tile + fi) * K_DIM +
                                               (k_tile + feed_idx);
                            end
                            // Simple: for simulation, feed zeros if not ready
                            for (fi = 0; fi < ARRAY_SIZE; fi = fi + 1)
                                sa_act_in[fi*DATA_WIDTH +: DATA_WIDTH] <= sp_read_data;
                        end
                        feed_idx <= feed_idx + 1;
                    end else begin
                        state <= S_WAIT;
                    end
                end

                S_WAIT: begin
                    if (sa_valid_out) begin
                        // Accumulate partial C results
                        for (ci = 0; ci < ARRAY_SIZE; ci = ci + 1)
                            c_accum[ci] <= c_accum[ci] +
                                $signed(sa_result[ci*ACC_WIDTH +: ACC_WIDTH]);
                        state <= S_NEXT_K;
                    end
                end

                S_NEXT_K: begin
                    k_tile <= k_tile + ARRAY_SIZE;
                    if (k_tile + ARRAY_SIZE < K_DIM) begin
                        sa_clear_acc <= 1'b1;
                        load_row <= 0; load_col <= 0;
                        state <= S_LOAD_B;
                    end else begin
                        store_idx <= 0;
                        state <= S_STORE_C;
                    end
                end

                // Store accumulated C tile to scratchpad
                S_STORE_C: begin
                    if (store_idx < ARRAY_SIZE) begin
                        sp_write_en   <= 1'b1;
                        sp_write_addr <= c_base_addr +
                                        (m_tile + store_idx) * N_DIM + n_tile;
                        sp_write_data <= c_accum[store_idx][DATA_WIDTH-1:0];
                        store_idx <= store_idx + 1;
                    end else begin
                        for (ci = 0; ci < ARRAY_SIZE; ci = ci + 1)
                            c_accum[ci] <= 0;
                        state <= S_NEXT_N;
                    end
                end

                S_NEXT_N: begin
                    n_tile <= n_tile + ARRAY_SIZE;
                    k_tile <= 0;
                    if (n_tile + ARRAY_SIZE < N_DIM) begin
                        sa_clear_acc <= 1'b1;
                        load_row <= 0; load_col <= 0;
                        state <= S_LOAD_B;
                    end else begin
                        state <= S_NEXT_M;
                    end
                end

                S_NEXT_M: begin
                    m_tile <= m_tile + ARRAY_SIZE;
                    n_tile <= 0; k_tile <= 0;
                    if (m_tile + ARRAY_SIZE < M_DIM) begin
                        sa_clear_acc <= 1'b1;
                        load_row <= 0; load_col <= 0;
                        state <= S_LOAD_B;
                    end else begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule

