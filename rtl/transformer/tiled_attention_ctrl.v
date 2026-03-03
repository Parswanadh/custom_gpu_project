// ============================================================================
// Module: tiled_attention_ctrl
// Description: FlashAttention-style tile sequencer for attention computation.
//
//   Standard attention computes the FULL N×N score matrix, requiring O(N²)
//   memory. For N=256 tokens, that's 128KB — far exceeding any on-chip SRAM.
//
//   Tiled attention processes Q×K^T in small tiles (B_r × B_c):
//     For each Q tile (B_r rows):
//       For each K/V tile (B_c columns):
//         1. Compute partial scores: Q_tile × K_tile^T
//         2. Stream scores into online_softmax_unit (one per cycle)
//         3. Online softmax maintains running max, denom, and V accumulator
//       After all K tiles: read final output from online softmax
//
//   Memory: O(B_r × B_c) instead of O(N²). With B_r=B_c=4: 32 bytes.
//   Sequences of 256-1024 tokens become feasible on our 4KB SRAM.
//
//   This module is the CONTROLLER — it orchestrates which tiles to load
//   and when to feed scores to the online softmax unit. It wraps around
//   an existing KV cache and score computation pipeline.
//
//   Reference: Dao et al., "FlashAttention: Fast and Memory-Efficient
//   Exact Attention with IO-Awareness" (arXiv:2205.14135, 2022)
// ============================================================================
module tiled_attention_ctrl #(
    parameter EMBED_DIM    = 4,
    parameter DATA_WIDTH   = 16,
    parameter MAX_SEQ_LEN  = 256,
    parameter TILE_SIZE    = 4          // B_r = B_c = TILE_SIZE
)(
    input  wire                              clk,
    input  wire                              rst,

    // Start attention for a query at position `seq_pos`
    input  wire                              start,
    input  wire [$clog2(MAX_SEQ_LEN)-1:0]   seq_pos,       // Current token position (0..N-1)

    // Q vector for the current token (already projected)
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]   q_vector,

    // KV cache read interface
    output reg  [$clog2(MAX_SEQ_LEN)-1:0]    kv_read_addr,   // Which token's K/V to read
    output reg                               kv_read_en,
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]   k_read_data,    // K vector from cache
    input  wire [EMBED_DIM*DATA_WIDTH-1:0]   v_read_data,    // V vector from cache

    // Online softmax unit interface
    output reg                               osm_start,       // Reset softmax state
    output reg                               osm_score_valid,
    output reg  signed [DATA_WIDTH-1:0]      osm_score,
    output reg  [EMBED_DIM*DATA_WIDTH-1:0]   osm_value,
    output reg                               osm_finalize,

    // Completion
    output reg                               done,
    output reg  [$clog2(MAX_SEQ_LEN)-1:0]    tiles_processed  // How many tiles completed
);

    // FSM States
    localparam S_IDLE      = 0;
    localparam S_INIT      = 1;   // Reset online softmax, set up tile counters
    localparam S_LOAD_KV   = 2;   // Read K/V from cache for current tile position
    localparam S_WAIT_KV   = 3;   // Wait 1 cycle for cache read
    localparam S_SCORE     = 4;   // Compute Q·K dot product, feed to softmax
    localparam S_NEXT_POS  = 5;   // Advance to next position in tile
    localparam S_NEXT_TILE = 6;   // Advance to next tile
    localparam S_FINALIZE  = 7;   // Signal online softmax to produce output
    localparam S_DONE      = 8;

    reg [3:0] state;

    // Tile tracking
    reg [$clog2(MAX_SEQ_LEN)-1:0] cur_pos;       // Current position within sequence
    reg [$clog2(MAX_SEQ_LEN)-1:0] tile_start;    // Start of current tile
    reg [$clog2(MAX_SEQ_LEN)-1:0] tile_end;      // End of current tile
    reg [$clog2(MAX_SEQ_LEN)-1:0] last_pos;      // Last valid position (= seq_pos)
    reg [$clog2(MAX_SEQ_LEN)-1:0] tile_count;

    // Score computation
    reg signed [2*DATA_WIDTH-1:0] dot_product;
    integer d;

    // Registered Q vector
    reg signed [DATA_WIDTH-1:0] q_reg [0:EMBED_DIM-1];

    always @(posedge clk) begin
        if (rst) begin
            state           <= S_IDLE;
            done            <= 1'b0;
            kv_read_en      <= 1'b0;
            osm_start       <= 1'b0;
            osm_score_valid <= 1'b0;
            osm_finalize    <= 1'b0;
            tiles_processed <= 0;
            cur_pos         <= 0;
            tile_start      <= 0;
            tile_end        <= 0;
            tile_count      <= 0;
        end else begin

            // Default deassertions
            kv_read_en      <= 1'b0;
            osm_start       <= 1'b0;
            osm_score_valid <= 1'b0;
            osm_finalize    <= 1'b0;
            done            <= 1'b0;

            case (state)

            S_IDLE: begin
                if (start) begin
                    last_pos   <= seq_pos;
                    // Register Q vector
                    for (d = 0; d < EMBED_DIM; d = d + 1)
                        q_reg[d] <= $signed(q_vector[d*DATA_WIDTH +: DATA_WIDTH]);
                    state <= S_INIT;
                end
            end

            S_INIT: begin
                // Reset online softmax for new attention computation
                osm_start   <= 1'b1;
                tile_start  <= 0;
                tile_count  <= 0;
                // First tile ends at min(TILE_SIZE-1, last_pos)
                tile_end    <= (TILE_SIZE - 1 < last_pos) ?
                               (TILE_SIZE - 1) : last_pos;
                cur_pos     <= 0;
                state       <= S_LOAD_KV;
            end

            S_LOAD_KV: begin
                // Request K/V for current position
                kv_read_addr <= cur_pos;
                kv_read_en   <= 1'b1;
                state        <= S_WAIT_KV;
            end

            S_WAIT_KV: begin
                // K/V data available this cycle (1-cycle read latency)
                state <= S_SCORE;
            end

            S_SCORE: begin
                // Compute dot product: Q · K[cur_pos]
                dot_product = 0;
                for (d = 0; d < EMBED_DIM; d = d + 1) begin
                    dot_product = dot_product +
                        q_reg[d] * $signed(k_read_data[d*DATA_WIDTH +: DATA_WIDTH]);
                end

                // Feed score and V to online softmax
                osm_score_valid <= 1'b1;
                osm_score       <= (dot_product >>> 8) >>> 1;  // Q8.8 + /sqrt(d)
                osm_value       <= v_read_data;   // Pass V vector through

                state <= S_NEXT_POS;
            end

            S_NEXT_POS: begin
                if (cur_pos < tile_end && cur_pos < last_pos) begin
                    // More positions in this tile
                    cur_pos <= cur_pos + 1;
                    state   <= S_LOAD_KV;
                end else begin
                    // Tile complete
                    tile_count <= tile_count + 1;
                    state      <= S_NEXT_TILE;
                end
            end

            S_NEXT_TILE: begin
                if (tile_end >= last_pos) begin
                    // All tiles processed → finalize
                    tiles_processed <= tile_count;
                    state           <= S_FINALIZE;
                end else begin
                    // More tiles: advance to next
                    tile_start <= tile_end + 1;
                    cur_pos    <= tile_end + 1;
                    tile_end   <= (tile_end + TILE_SIZE < last_pos) ?
                                  (tile_end + TILE_SIZE) : last_pos;
                    state      <= S_LOAD_KV;
                end
            end

            S_FINALIZE: begin
                osm_finalize <= 1'b1;
                state        <= S_DONE;
            end

            S_DONE: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end

            endcase
        end
    end

endmodule
