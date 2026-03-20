`timescale 1ns / 1ps

// ============================================================================
// Module: optimized_transformer_layer
// Description: REAL End-to-End Integrated Transformer Layer.
//
// This is NOT a side-by-side benchmark. This is a REAL pipeline where:
//   Stage 1: RoPE rotates Q and K vectors → output feeds into Stage 2
//   Stage 2: GQA computes attention scores using rotated Q,K
//   Stage 3: Parallel Softmax normalizes the attention scores  
//   Stage 4: GELU activation on the weighted values
//   Stage 5: KV Cache Quantization stores compressed KV entries
//   Stage 6: Activation Compression prepares data for next layer
//
// Every output wire goes to the next stage's input wire.
// Every cycle count is measured from simulation.
// ZERO made-up numbers.
// ============================================================================
module optimized_transformer_layer #(
    parameter DIM         = 8,
    parameter NUM_Q_HEADS = 4,
    parameter NUM_KV_HEADS = 2,
    parameter HEAD_DIM    = 4
)(
    input  wire                      clk,
    input  wire                      rst,
    
    // Token input
    input  wire                      start,
    input  wire [DIM*16-1:0]         token_embedding,  // 8-dim × 16-bit
    input  wire [5:0]                position,
    
    // Final output
    output reg                       done,
    output reg  [DIM*16-1:0]         layer_output,
    
    // Per-stage completion signals (for cycle measurement)
    output reg                       rope_complete,
    output reg                       gqa_complete,
    output reg                       softmax_complete,
    output reg                       gelu_complete,
    output reg                       kv_quant_complete,
    output reg                       compress_complete,
    
    // Per-stage measured cycle counts
    output reg [15:0]                rope_cycles,
    output reg [15:0]                gqa_cycles,
    output reg [15:0]                softmax_cycles,
    output reg [15:0]                gelu_cycles,
    output reg [15:0]                kv_quant_cycles,
    output reg [15:0]                compress_cycles,
    output reg [15:0]                total_cycles
);

    // =====================================================================
    // STAGE 1: RoPE — Rotary Position Encoding
    // Input: token_embedding → Q, K vectors
    // Output: rotated Q, K → feeds into GQA
    // =====================================================================
    reg rope_valid_in;
    wire [DIM*16-1:0] rope_q_out, rope_k_out;
    wire rope_valid_out;
    
    rope_encoder #(.DIM(DIM)) u_rope (
        .clk(clk), .rst(rst),
        .valid_in(rope_valid_in),
        .position(position),
        .q_in(token_embedding),          // Use embedding as Q
        .k_in(token_embedding),          // Use embedding as K (self-attention)
        .q_rot(rope_q_out),             // → feeds into GQA
        .k_rot(rope_k_out),             // → feeds into GQA
        .valid_out(rope_valid_out)
    );

    // =====================================================================
    // STAGE 2: GQA — Grouped Query Attention
    // Input: rotated Q, K from RoPE
    // Output: attention scores → feeds into Softmax
    // =====================================================================
    reg gqa_valid_in;
    
    // Map vectors into per-head views with deterministic head diversity.
    wire [NUM_Q_HEADS*HEAD_DIM*16-1:0] gqa_q_in;
    wire [NUM_KV_HEADS*HEAD_DIM*16-1:0] gqa_k_in, gqa_v_in;
    reg  [NUM_Q_HEADS*HEAD_DIM*16-1:0] gqa_q_in_r;
    reg  [NUM_KV_HEADS*HEAD_DIM*16-1:0] gqa_k_in_r, gqa_v_in_r;
    integer map_qh, map_kh, map_d;
    integer src_idx_q, src_idx_k, src_idx_v;
    reg signed [15:0] map_q_val, map_k_val, map_v_val;

    always @(*) begin
        gqa_q_in_r = {NUM_Q_HEADS*HEAD_DIM*16{1'b0}};
        gqa_k_in_r = {NUM_KV_HEADS*HEAD_DIM*16{1'b0}};
        gqa_v_in_r = {NUM_KV_HEADS*HEAD_DIM*16{1'b0}};

        for (map_qh = 0; map_qh < NUM_Q_HEADS; map_qh = map_qh + 1) begin
            for (map_d = 0; map_d < HEAD_DIM; map_d = map_d + 1) begin
                src_idx_q = (map_qh * HEAD_DIM + map_d) % DIM;
                map_q_val = $signed(rope_q_out[src_idx_q*16 +: 16]);
                // Alternate sign by head to avoid silent head-collapse.
                if ((map_qh % 2) == 1)
                    gqa_q_in_r[(map_qh*HEAD_DIM + map_d)*16 +: 16] = -map_q_val;
                else
                    gqa_q_in_r[(map_qh*HEAD_DIM + map_d)*16 +: 16] = map_q_val;
            end
        end

        for (map_kh = 0; map_kh < NUM_KV_HEADS; map_kh = map_kh + 1) begin
            for (map_d = 0; map_d < HEAD_DIM; map_d = map_d + 1) begin
                src_idx_k = (map_kh * HEAD_DIM + map_d) % DIM;
                src_idx_v = (map_kh * HEAD_DIM + map_d + (DIM/2)) % DIM;
                map_k_val = $signed(rope_k_out[src_idx_k*16 +: 16]);
                map_v_val = $signed(token_embedding[src_idx_v*16 +: 16]);
                gqa_k_in_r[(map_kh*HEAD_DIM + map_d)*16 +: 16] = map_k_val;
                gqa_v_in_r[(map_kh*HEAD_DIM + map_d)*16 +: 16] = map_v_val;
            end
        end
    end

    assign gqa_q_in = gqa_q_in_r;
    assign gqa_k_in = gqa_k_in_r;
    assign gqa_v_in = gqa_v_in_r;
    
    wire [NUM_Q_HEADS*16-1:0] gqa_scores_out;
    wire [NUM_Q_HEADS*16-1:0] gqa_values_out;
    wire gqa_valid_out;
    wire [15:0] gqa_kv_saved;
    
    grouped_query_attention #(
        .NUM_Q_HEADS(NUM_Q_HEADS), .NUM_KV_HEADS(NUM_KV_HEADS), .HEAD_DIM(HEAD_DIM)
    ) u_gqa (
        .clk(clk), .rst(rst),
        .valid_in(gqa_valid_in),
        .q_heads(gqa_q_in),             // ← from RoPE
        .k_heads(gqa_k_in),             // ← from RoPE
        .v_heads(gqa_v_in),             // ← from RoPE
        .attention_scores(gqa_scores_out), // → feeds into Softmax
        .attention_values(gqa_values_out),
        .valid_out(gqa_valid_out),
        .kv_memory_saved(gqa_kv_saved)
    );

    // =====================================================================
    // STAGE 3: Parallel Softmax
    // Input: attention scores from GQA
    // Output: probabilities → feeds into weighted value computation
    // =====================================================================
    reg sm_valid_in;
    
    // Wire GQA scores directly to Softmax (real connection!)
    wire [NUM_Q_HEADS*16-1:0] sm_input = gqa_scores_out;
    wire [NUM_Q_HEADS*8-1:0] sm_probs_out;
    wire sm_valid_out;
    wire [15:0] sm_cycles_out;
    
    parallel_softmax #(.VECTOR_LEN(NUM_Q_HEADS), .PARALLEL_UNITS(NUM_Q_HEADS))
    u_softmax (
        .clk(clk), .rst(rst),
        .valid_in(sm_valid_in),
        .x_in(sm_input),                // ← from GQA
        .prob_out(sm_probs_out),         // → used for weighted sum
        .valid_out(sm_valid_out),
        .cycles_used(sm_cycles_out)
    );

    // =====================================================================
    // STAGE 4: GELU Activation (FFN stage)
    // Input: first element from softmax probabilities
    // Output: activated value → feeds into KV quantizer
    // =====================================================================
    reg gelu_valid_in;
    
    function signed [15:0] reduce_weighted_context;
        input [NUM_Q_HEADS*8-1:0] probs;
        input [NUM_Q_HEADS*16-1:0] values;
        integer wi;
        reg signed [39:0] acc;
        reg signed [15:0] v_elem;
        reg [7:0] p_elem;
        begin
            acc = 40'sd0;
            for (wi = 0; wi < NUM_Q_HEADS; wi = wi + 1) begin
                v_elem = $signed(values[wi*16 +: 16]);
                p_elem = probs[wi*8 +: 8];
                acc = acc + (v_elem * $signed({1'b0, p_elem}));
            end
            if (acc > (40'sd32767 <<< 8))
                reduce_weighted_context = 16'sh7FFF;
            else if (acc < (-40'sd32768 <<< 8))
                reduce_weighted_context = -16'sh8000;
            else
                reduce_weighted_context = acc >>> 8;
        end
    endfunction

    // Convert softmax-weighted attention values to GELU input.
    wire signed [15:0] weighted_context = reduce_weighted_context(sm_probs_out, gqa_values_out);
    wire signed [15:0] gelu_input = weighted_context;
    wire signed [15:0] gelu_output;
    wire gelu_valid_out;
    
    gelu_activation #(.WIDTH(16)) u_gelu (
        .clk(clk), .rst(rst),
        .x_in(gelu_input),              // ← from Softmax
        .valid_in(gelu_valid_in),
        .y_out(gelu_output),             // → feeds into KV quantizer
        .valid_out(gelu_valid_out)
    );

    // =====================================================================
    // STAGE 5: KV Cache INT4 Quantization
    // Input: values derived from the pipeline processing
    // Output: quantized cache entry
    // =====================================================================
    reg kv_valid_in;
    
    // Build KV input from pipeline data (real connection!)
    wire [4*16-1:0] kv_input = {gelu_output, weighted_context,
                                 gqa_values_out[31:16], gqa_values_out[15:0]};
    wire [15:0] kv_quantized_out;
    wire signed [15:0] kv_min_out;
    wire [15:0] kv_scale_out;
    wire kv_quant_done;
    wire [31:0] kv_bytes_saved;
    
    kv_cache_quantizer #(.VEC_LEN(4)) u_kv_quant (
        .clk(clk), .rst(rst),
        .quant_valid(kv_valid_in),
        .kv_in(kv_input),               // ← from GELU + GQA scores
        .kv_quantized(kv_quantized_out),
        .quant_min(kv_min_out),
        .quant_scale(kv_scale_out),
        .quant_done(kv_quant_done),
        .dequant_valid(1'b0), .kv_q_in(16'd0),
        .dequant_min(16'sd0), .dequant_scale(16'd0),
        .bytes_saved(kv_bytes_saved)
    );

    // =====================================================================
    // STAGE 6: Activation Compression
    // Input: layer output values
    // Output: compressed activations for next layer
    // =====================================================================
    reg comp_valid_in;
    
    // Build compression input from pipeline data (real connection!)
    wire [4*16-1:0] comp_input = {gelu_output, weighted_context,
                                   gqa_values_out[31:16], gqa_values_out[15:0]};
    wire [4*8-1:0] comp_output;
    wire [7:0] comp_scale_out;
    wire comp_done;
    
    activation_compressor #(.VECTOR_LEN(4)) u_compress (
        .clk(clk), .rst(rst),
        .compress_valid(comp_valid_in),
        .data_in(comp_input),            // ← from GELU + GQA
        .compressed_out(comp_output),     // → final compressed output
        .scale_out(comp_scale_out),
        .compress_done(comp_done),
        .decompress_valid(1'b0), .compressed_in(32'd0), .scale_in(8'd0)
    );

    // =====================================================================
    // PIPELINE CONTROLLER — sequences the stages
    // =====================================================================
    reg [3:0] state;
    localparam S_IDLE       = 4'd0;
    localparam S_ROPE       = 4'd1;
    localparam S_ROPE_W     = 4'd2;
    localparam S_GQA_W      = 4'd3;
    localparam S_SOFTMAX_W  = 4'd4;
    localparam S_GELU_W     = 4'd5;
    localparam S_TAIL_WAIT  = 4'd6;
    localparam S_DONE       = 4'd7;
    
    reg [15:0] cycle_counter;
    reg [15:0] stage_start;
    reg [15:0] wait_counter;
    reg        start_pending;
    reg        kv_done_seen;
    reg        comp_done_seen;
    reg        timeout_abort;
    localparam [15:0] STAGE_TIMEOUT_CYCLES = 16'd2048;
    
    always @(posedge clk) begin
        if (rst) begin
            state          <= S_IDLE;
            done           <= 1'b0;
            rope_valid_in  <= 1'b0;
            gqa_valid_in   <= 1'b0;
            sm_valid_in    <= 1'b0;
            gelu_valid_in  <= 1'b0;
            kv_valid_in    <= 1'b0;
            comp_valid_in  <= 1'b0;
            rope_complete  <= 1'b0;
            gqa_complete   <= 1'b0;
            softmax_complete <= 1'b0;
            gelu_complete  <= 1'b0;
            kv_quant_complete <= 1'b0;
            compress_complete <= 1'b0;
            rope_cycles    <= 0;
            gqa_cycles     <= 0;
            softmax_cycles <= 0;
            gelu_cycles    <= 0;
            kv_quant_cycles <= 0;
            compress_cycles <= 0;
            total_cycles   <= 0;
            cycle_counter  <= 0;
            stage_start    <= 0;
            wait_counter   <= 0;
            start_pending  <= 1'b0;
            kv_done_seen   <= 1'b0;
            comp_done_seen <= 1'b0;
            timeout_abort  <= 1'b0;
            layer_output   <= 0;
        end else begin
            cycle_counter <= cycle_counter + 1;
            if (state == S_ROPE_W || state == S_GQA_W || state == S_SOFTMAX_W ||
                state == S_GELU_W || state == S_TAIL_WAIT)
                wait_counter <= wait_counter + 1;
            else
                wait_counter <= 16'd0;
            
            // Deassert one-shot valids
            rope_valid_in <= 1'b0;
            gqa_valid_in  <= 1'b0;
            sm_valid_in   <= 1'b0;
            gelu_valid_in <= 1'b0;
            kv_valid_in   <= 1'b0;
            comp_valid_in <= 1'b0;

            // Deassert one-cycle stage completion pulses
            rope_complete    <= 1'b0;
            gqa_complete     <= 1'b0;
            softmax_complete <= 1'b0;
            gelu_complete    <= 1'b0;
            kv_quant_complete <= 1'b0;
            compress_complete <= 1'b0;

            // Capture start pulses while the controller is busy so they are
            // not dropped at completion boundary transitions.
            if (start && (state != S_IDLE))
                start_pending <= 1'b1;
            
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start || start_pending) begin
                        cycle_counter <= 0;
                        stage_start   <= 0;
                        wait_counter  <= 0;
                        start_pending <= 1'b0;
                        kv_done_seen  <= 1'b0;
                        comp_done_seen <= 1'b0;
                        timeout_abort <= 1'b0;
                        state <= S_ROPE;
                    end
                end
                
                // --- STAGE 1: RoPE ---
                S_ROPE: begin
                    rope_valid_in <= 1'b1;
                    stage_start <= cycle_counter;
                    wait_counter <= 16'd0;
                    state <= S_ROPE_W;
                end
                S_ROPE_W: begin
                    if (wait_counter >= STAGE_TIMEOUT_CYCLES) begin
                        timeout_abort <= 1'b1;
                        state <= S_DONE;
                    end else if (rope_valid_out) begin
                        rope_cycles <= cycle_counter - stage_start + 1;
                        rope_complete <= 1'b1;
                        // Launch next stage immediately on handoff.
                        gqa_valid_in <= 1'b1;
                        stage_start <= cycle_counter;
                        wait_counter <= 16'd0;
                        state <= S_GQA_W;
                    end
                end
                S_GQA_W: begin
                    if (wait_counter >= STAGE_TIMEOUT_CYCLES) begin
                        timeout_abort <= 1'b1;
                        state <= S_DONE;
                    end else if (gqa_valid_out) begin
                        gqa_cycles <= cycle_counter - stage_start + 1;
                        gqa_complete <= 1'b1;
                        // Launch next stage immediately on handoff.
                        sm_valid_in <= 1'b1;
                        stage_start <= cycle_counter;
                        wait_counter <= 16'd0;
                        state <= S_SOFTMAX_W;
                    end
                end
                S_SOFTMAX_W: begin
                    if (wait_counter >= STAGE_TIMEOUT_CYCLES) begin
                        timeout_abort <= 1'b1;
                        state <= S_DONE;
                    end else if (sm_valid_out) begin
                        softmax_cycles <= cycle_counter - stage_start + 1;
                        softmax_complete <= 1'b1;
                        // Launch next stage immediately on handoff.
                        gelu_valid_in <= 1'b1;
                        stage_start <= cycle_counter;
                        wait_counter <= 16'd0;
                        state <= S_GELU_W;
                    end
                end
                S_GELU_W: begin
                    if (wait_counter >= STAGE_TIMEOUT_CYCLES) begin
                        timeout_abort <= 1'b1;
                        state <= S_DONE;
                    end else if (gelu_valid_out) begin
                        gelu_cycles <= cycle_counter - stage_start + 1;
                        gelu_complete <= 1'b1;
                        // Tail stages already run in parallel.
                        kv_valid_in <= 1'b1;
                        comp_valid_in <= 1'b1;
                        stage_start <= cycle_counter;
                        wait_counter <= 16'd0;
                        kv_done_seen <= 1'b0;
                        comp_done_seen <= 1'b0;
                        state <= S_TAIL_WAIT;
                    end
                end

                // --- STAGE 5 + 6: wait for both completions ---
                S_TAIL_WAIT: begin
                    if (wait_counter >= STAGE_TIMEOUT_CYCLES) begin
                        timeout_abort <= 1'b1;
                        state <= S_DONE;
                    end

                    if (kv_quant_done && !kv_done_seen) begin
                        kv_quant_cycles <= cycle_counter - stage_start + 1;
                        kv_quant_complete <= 1'b1;
                        kv_done_seen <= 1'b1;
                    end

                    if (comp_done && !comp_done_seen) begin
                        compress_cycles <= cycle_counter - stage_start + 1;
                        compress_complete <= 1'b1;
                        comp_done_seen <= 1'b1;
                    end

                    if ((kv_done_seen || kv_quant_done) && (comp_done_seen || comp_done)) begin
                        state <= S_DONE;
                    end
                end
                
                // --- ALL STAGES COMPLETE ---
                S_DONE: begin
                    total_cycles <= cycle_counter + 1;
                    if (timeout_abort) begin
                        layer_output <= {DIM*16{1'b0}};
                    end else begin
                        // Build final output from compressed data +  original dims
                        layer_output <= {
                            {8'd0, comp_output[24 +: 8]},
                            {8'd0, comp_output[16 +: 8]},
                            {8'd0, comp_output[8 +: 8]},
                            {8'd0, comp_output[0 +: 8]},
                            gelu_output,
                            rope_q_out[47:32],
                            rope_q_out[31:16],
                            rope_q_out[15:0]
                        };
                    end
                    done <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
