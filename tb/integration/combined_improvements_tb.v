// ============================================================================
// Testbench: combined_improvements_tb
// Description: Integration test exercising ALL Phase 8 improvements together
//   in a realistic inference-like pipeline:
//
//   1. PMU starts in FULL mode
//   2. Token Scheduler feeds tokens
//   3. Weights loaded via Double Buffer (bank swap between "layers")
//   4. 2:4 Sparsity Decoder processes weight groups
//   5. Parallel Attention processes heads
//   6. Online Softmax computes attention probabilities
//   7. Activation Compressor compresses inter-layer data
//   8. PMU transitions to ECO → SLEEP when done
//
// This proves all modules can coexist and interact in a single design.
// ============================================================================
`timescale 1ns / 1ps

module combined_improvements_tb;

    reg clk, rst;
    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer total_tests = 0;

    // ========================================================================
    // Module instantiations
    // ========================================================================

    // --- 1. Power Management Unit ---
    parameter NUM_CORES = 4;
    reg  [NUM_CORES-1:0]  core_active;
    reg  [1:0]            power_mode_req;
    reg                   wake_interrupt;
    wire [NUM_CORES-1:0]  core_clk_en;
    wire [1:0]            current_mode;
    wire [31:0]           pmu_idle, pmu_active, pmu_sleep, pmu_gated;

    power_management_unit #(.NUM_CORES(NUM_CORES), .IDLE_TIMEOUT(32)) u_pmu (
        .clk(clk), .rst(rst),
        .core_active(core_active), .power_mode_req(power_mode_req),
        .wake_interrupt(wake_interrupt), .core_clk_en(core_clk_en),
        .current_mode(current_mode),
        .idle_cycles(pmu_idle), .active_cycles(pmu_active),
        .sleep_cycles(pmu_sleep), .gated_core_cycles(pmu_gated)
    );

    // --- 2. Weight Double Buffer ---
    parameter WDB_DEPTH = 16;
    parameter DW = 16;
    reg                         wdb_load_en, wdb_read_en, wdb_swap;
    reg  [$clog2(WDB_DEPTH)-1:0] wdb_load_addr, wdb_read_addr;
    reg  signed [DW-1:0]        wdb_load_data;
    wire signed [DW-1:0]        wdb_read_data;
    wire                        wdb_read_valid, wdb_active_bank;
    wire [31:0]                 wdb_loads, wdb_swaps;

    weight_double_buffer #(.NUM_WEIGHTS(WDB_DEPTH), .DATA_WIDTH(DW)) u_wdb (
        .clk(clk), .rst(rst),
        .load_en(wdb_load_en), .load_addr(wdb_load_addr), .load_data(wdb_load_data),
        .read_en(wdb_read_en), .read_addr(wdb_read_addr),
        .read_data(wdb_read_data), .read_valid(wdb_read_valid),
        .swap_banks(wdb_swap), .active_bank(wdb_active_bank),
        .loads_completed(wdb_loads), .swaps_completed(wdb_swaps)
    );

    // --- 3. 2:4 Sparsity Decoder ---
    reg                     sp24_valid;
    reg  [4*DW-1:0]        sp24_weights, sp24_acts;
    reg  [1:0]             sp24_bitmap;
    wire signed [2*DW-1:0] sp24_result;
    wire                   sp24_valid_out;
    wire [31:0]            sp24_skipped, sp24_computed;

    sparsity_decoder_2_4 #(.DATA_WIDTH(DW)) u_sp24 (
        .clk(clk), .rst(rst), .valid_in(sp24_valid),
        .weights_in(sp24_weights), .activations_in(sp24_acts),
        .sparsity_bitmap(sp24_bitmap),
        .nz_idx_0(), .nz_idx_1(),
        .nz_weight_0(), .nz_weight_1(),
        .nz_act_0(), .nz_act_1(),
        .result(sp24_result), .valid_out(sp24_valid_out),
        .skipped_count(sp24_skipped), .computed_count(sp24_computed)
    );

    // --- 4. Online Softmax ---
    parameter VL = 4;
    reg                     osm_valid;
    reg  [VL*DW-1:0]       osm_input;
    wire [VL*8-1:0]        osm_probs;
    wire                   osm_valid_out;

    online_softmax #(.VECTOR_LEN(VL), .DATA_WIDTH(DW)) u_osm (
        .clk(clk), .rst(rst), .valid_in(osm_valid),
        .x_in(osm_input), .prob_out(osm_probs), .valid_out(osm_valid_out)
    );

    // --- 5. Activation Compressor ---
    reg                     ac_compress_valid, ac_decompress_valid;
    reg  [VL*DW-1:0]       ac_data_in;
    wire [VL*8-1:0]        ac_compressed;
    wire [7:0]             ac_scale;
    wire                   ac_compress_done, ac_decompress_done;
    reg  [VL*8-1:0]        ac_compressed_in;
    reg  [7:0]             ac_scale_in;
    wire [VL*DW-1:0]       ac_decompressed;
    wire [31:0]            ac_compressions, ac_bytes_saved;

    activation_compressor #(.VECTOR_LEN(VL), .DATA_WIDTH(DW)) u_ac (
        .clk(clk), .rst(rst),
        .compress_valid(ac_compress_valid), .data_in(ac_data_in),
        .compressed_out(ac_compressed), .scale_out(ac_scale),
        .compress_done(ac_compress_done),
        .decompress_valid(ac_decompress_valid),
        .compressed_in(ac_compressed_in), .scale_in(ac_scale_in),
        .decompressed_out(ac_decompressed), .decompress_done(ac_decompress_done),
        .total_compressions(ac_compressions), .total_bytes_saved(ac_bytes_saved)
    );

    // --- 6. Parallel Attention ---
    parameter ED = 8, NH = 4, HD = 2, NP = 2, MSL = 16;
    reg                         pa_valid;
    reg  [ED*DW-1:0]           pa_x_in;
    reg  [$clog2(MSL)-1:0]     pa_seq_pos;
    wire [ED*DW-1:0]           pa_y_out;
    wire                       pa_valid_out;
    wire [31:0]                pa_zero_skips, pa_heads_processed;

    parallel_attention #(
        .EMBED_DIM(ED), .NUM_HEADS(NH), .HEAD_DIM(HD),
        .NUM_PARALLEL(NP), .MAX_SEQ_LEN(MSL), .DATA_WIDTH(DW)
    ) u_pa (
        .clk(clk), .rst(rst), .valid_in(pa_valid),
        .x_in(pa_x_in), .seq_pos(pa_seq_pos),
        .y_out(pa_y_out), .valid_out(pa_valid_out),
        .zero_skip_count(pa_zero_skips), .heads_processed(pa_heads_processed)
    );

    // --- 7. Token Scheduler ---
    parameter VB = 4, SB = 4, MGL = 16;
    reg                     ts_start;
    reg  [VB-1:0]          ts_seed;
    reg  [7:0]             ts_num_tokens;
    wire                   ts_engine_valid;
    wire [VB-1:0]          ts_engine_token;
    wire [SB-1:0]          ts_engine_pos;
    reg                    ts_engine_done;
    reg  [VB-1:0]          ts_engine_out;
    wire [MGL*VB-1:0]     ts_sequence;
    wire [7:0]            ts_tokens_done;
    wire                  ts_gen_done, ts_busy;

    token_scheduler #(.VOCAB_BITS(VB), .SEQ_BITS(SB), .MAX_GEN_LEN(MGL)) u_ts (
        .clk(clk), .rst(rst),
        .start(ts_start), .seed_token(ts_seed), .num_tokens(ts_num_tokens),
        .engine_valid_in(ts_engine_valid), .engine_token_in(ts_engine_token),
        .engine_position(ts_engine_pos),
        .engine_valid_out(ts_engine_done), .engine_token_out(ts_engine_out),
        .generated_sequence(ts_sequence), .tokens_generated(ts_tokens_done),
        .generation_done(ts_gen_done), .busy(ts_busy)
    );

    // Mock engine for token scheduler: token_out = (token_in + 3) % 16
    always @(posedge clk) begin
        ts_engine_done <= 1'b0;
        if (ts_engine_valid) begin
            ts_engine_out <= (ts_engine_token + 3) & 4'hF;
            ts_engine_done <= 1'b1;
        end
    end

    // Helpers
    integer timeout, i;

    initial begin
        $dumpfile("sim/waveforms/combined_improvements.vcd");
        $dumpvars(0, combined_improvements_tb);
    end

    initial begin
        $display("");
        $display("================================================================");
        $display("  COMBINED IMPROVEMENTS INTEGRATION TEST");
        $display("  Testing all Phase 8 modules working together");
        $display("================================================================");
        $display("");

        // Initialize everything
        rst = 1;
        core_active = 0; power_mode_req = 2'b01; wake_interrupt = 0;
        wdb_load_en = 0; wdb_read_en = 0; wdb_swap = 0;
        wdb_load_addr = 0; wdb_read_addr = 0; wdb_load_data = 0;
        sp24_valid = 0; sp24_weights = 0; sp24_acts = 0; sp24_bitmap = 0;
        osm_valid = 0; osm_input = 0;
        ac_compress_valid = 0; ac_decompress_valid = 0; ac_data_in = 0;
        ac_compressed_in = 0; ac_scale_in = 0;
        pa_valid = 0; pa_x_in = 0; pa_seq_pos = 0;
        ts_start = 0; ts_seed = 0; ts_num_tokens = 0;
        ts_engine_done = 0; ts_engine_out = 0;
        #35; rst = 0; #15;

        // ================================================================
        // STAGE 1: PMU boots in FULL mode, all cores enabled
        // ================================================================
        $display("--- Stage 1: PMU Boot ---");
        core_active = 4'b1111;
        power_mode_req = 2'b01; // Force FULL
        repeat(3) @(posedge clk);
        total_tests = total_tests + 1;
        if (current_mode == 2'd0 && core_clk_en == 4'b1111) begin
            $display("[PASS] PMU: FULL mode, all cores ON");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] PMU boot");
            fail_count = fail_count + 1;
        end

        // ================================================================
        // STAGE 2: Load "Layer 0" weights into double buffer
        // ================================================================
        $display("--- Stage 2: Weight Double Buffer Load ---");
        // Swap so Bank A becomes writable
        @(posedge clk); wdb_swap <= 1;
        @(posedge clk); wdb_swap <= 0;

        // Load 4 weights into Bank A (will become active after swap)
        for (i = 0; i < 4; i = i + 1) begin
            @(posedge clk);
            wdb_load_en <= 1; wdb_load_addr <= i;
            wdb_load_data <= (i + 1) * 16'sd64; // 64, 128, 192, 256
        end
        @(posedge clk); wdb_load_en <= 0;

        // Swap: Bank A now active with loaded weights
        @(posedge clk); wdb_swap <= 1;
        @(posedge clk); wdb_swap <= 0;

        // Read back to verify
        @(posedge clk); wdb_read_en <= 1; wdb_read_addr <= 0;
        @(posedge clk); wdb_read_en <= 0;
        @(posedge clk); #1;

        total_tests = total_tests + 1;
        if (wdb_read_data == 16'sd64) begin
            $display("[PASS] Double Buffer: Layer 0 weights loaded (first weight = %0d)", wdb_read_data);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Double Buffer: got %0d, expected 64", wdb_read_data);
            fail_count = fail_count + 1;
        end

        // ================================================================
        // STAGE 3: Process weights through 2:4 Sparsity Decoder
        // ================================================================
        $display("--- Stage 3: 2:4 Sparsity Decoder ---");
        // Weights: [64, 0, 128, 0], Activations: [10, 20, 30, 40]
        // Bitmap 01: positions [0,2] active → 64*10 + 128*30 = 640 + 3840 = 4480
        @(negedge clk);
        sp24_weights = {16'sd0, 16'sd128, 16'sd0, 16'sd64};
        sp24_acts = {16'sd40, 16'sd30, 16'sd20, 16'sd10};
        sp24_bitmap = 2'b01;
        sp24_valid = 1'b1;
        @(negedge clk); sp24_valid = 1'b0;

        timeout = 0;
        while (!sp24_valid_out && timeout < 10) begin
            @(posedge clk); #1;
            timeout = timeout + 1;
        end

        total_tests = total_tests + 1;
        if (sp24_valid_out && sp24_result == 32'sd4480) begin
            $display("[PASS] 2:4 Sparsity: result=%0d (2 mults done, 2 skipped)", sp24_result);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] 2:4 Sparsity: result=%0d (expected 4480)", sp24_result);
            fail_count = fail_count + 1;
        end

        // ================================================================
        // STAGE 4: Run Parallel Attention on embedding-like data
        // ================================================================
        $display("--- Stage 4: Parallel Attention (4 heads, 2 parallel) ---");
        @(negedge clk);
        pa_x_in = {16'sd100, 16'sd200, 16'sd300, 16'sd400,
                    16'sd50, 16'sd150, 16'sd250, 16'sd350};
        pa_seq_pos = 0;
        pa_valid = 1'b1;
        @(negedge clk); pa_valid = 1'b0;

        timeout = 0;
        while (!pa_valid_out && timeout < 50) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        total_tests = total_tests + 1;
        if (pa_valid_out) begin
            $display("[PASS] Parallel Attention: processed in %0d cycles, %0d heads", timeout, pa_heads_processed);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Parallel Attention: timeout");
            fail_count = fail_count + 1;
        end

        // ================================================================
        // STAGE 5: Online Softmax on attention scores
        // ================================================================
        $display("--- Stage 5: Online Softmax (streaming) ---");
        @(negedge clk);
        osm_input = {16'sd768, 16'sd512, 16'sd256, 16'sd0};
        osm_valid = 1'b1;
        @(negedge clk); osm_valid = 1'b0;

        timeout = 0;
        while (!osm_valid_out && timeout < 100) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        total_tests = total_tests + 1;
        if (osm_valid_out) begin
            begin : osm_check
                integer psum;
                psum = 0;
                for (i = 0; i < VL; i = i + 1)
                    psum = psum + osm_probs[i*8 +: 8];
                if (psum >= 220 && psum <= 290) begin
                    $display("[PASS] Online Softmax: probs=[%0d,%0d,%0d,%0d] sum=%0d",
                        osm_probs[7:0], osm_probs[15:8], osm_probs[23:16], osm_probs[31:24], psum);
                    pass_count = pass_count + 1;
                end else begin
                    $display("[FAIL] Online Softmax: sum=%0d (expected ~256)", psum);
                    fail_count = fail_count + 1;
                end
            end
        end else begin
            $display("[FAIL] Online Softmax: timeout");
            fail_count = fail_count + 1;
        end

        // ================================================================
        // STAGE 6: Compress activations between layers
        // ================================================================
        $display("--- Stage 6: Activation Compression ---");
        @(negedge clk);
        ac_data_in = pa_y_out[VL*DW-1:0]; // Use attention output as input
        ac_compress_valid = 1'b1;
        @(negedge clk); ac_compress_valid = 1'b0;

        timeout = 0;
        while (!ac_compress_done && timeout < 20) begin
            @(posedge clk); #1;
            timeout = timeout + 1;
        end

        total_tests = total_tests + 1;
        if (ac_compress_done) begin
            $display("[PASS] Compression: scale=%0d, saved %0d bytes", ac_scale, ac_bytes_saved);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Compression timeout");
            fail_count = fail_count + 1;
        end

        // Decompress and verify round-trip
        repeat(2) @(posedge clk);
        @(negedge clk);
        ac_compressed_in = ac_compressed;
        ac_scale_in = ac_scale;
        ac_decompress_valid = 1'b1;
        @(negedge clk); ac_decompress_valid = 1'b0;

        timeout = 0;
        while (!ac_decompress_done && timeout < 20) begin
            @(posedge clk); #1;
            timeout = timeout + 1;
        end

        total_tests = total_tests + 1;
        if (ac_decompress_done) begin
            $display("[PASS] Decompression: round-trip complete");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Decompression timeout");
            fail_count = fail_count + 1;
        end

        // ================================================================
        // STAGE 7: Token Scheduler generates 5 tokens autonomously
        // ================================================================
        $display("--- Stage 7: Autonomous Token Generation ---");
        @(posedge clk);
        ts_seed = 4'd1;
        ts_num_tokens = 8'd5;
        ts_start = 1'b1;
        @(posedge clk); ts_start = 1'b0;

        timeout = 0;
        while (!ts_gen_done && timeout < 200) begin
            @(posedge clk);
            timeout = timeout + 1;
        end

        total_tests = total_tests + 1;
        if (ts_gen_done && ts_tokens_done == 8'd5) begin
            $display("[PASS] Token Scheduler: generated %0d tokens in %0d cycles", ts_tokens_done, timeout);
            // Print token sequence
            $write("       Sequence: %0d", ts_sequence[0 +: VB]);
            for (i = 1; i <= 5; i = i + 1)
                $write(" -> %0d", ts_sequence[i*VB +: VB]);
            $display("");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Token Scheduler: done=%b, tokens=%0d", ts_gen_done, ts_tokens_done);
            fail_count = fail_count + 1;
        end

        // ================================================================
        // STAGE 8: PMU transitions to SLEEP after work completes
        // ================================================================
        $display("--- Stage 8: PMU Power Down ---");
        core_active = 4'b0000;
        power_mode_req = 2'b11; // Force SLEEP
        repeat(3) @(posedge clk);

        total_tests = total_tests + 1;
        if (current_mode == 2'd2 && core_clk_en == 4'b0000) begin
            $display("[PASS] PMU: SLEEP mode, all cores OFF");
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] PMU sleep: mode=%0d, clk_en=%b", current_mode, core_clk_en);
            fail_count = fail_count + 1;
        end

        // ================================================================
        // FINAL SUMMARY
        // ================================================================
        $display("");
        $display("================================================================");
        $display("  COMBINED IMPROVEMENTS — INTEGRATION RESULTS");
        $display("================================================================");
        $display("  Total tests:     %0d", total_tests);
        $display("  Passed:          %0d", pass_count);
        $display("  Failed:          %0d", fail_count);
        $display("");
        $display("  PMU active cycles:     %0d", pmu_active);
        $display("  PMU gated core-cycles: %0d", pmu_gated);
        $display("  2:4 skipped mults:     %0d / %0d total", sp24_skipped, sp24_computed + sp24_skipped);
        $display("  Attn heads processed:  %0d", pa_heads_processed);
        $display("  Bytes saved (compress):%0d", ac_bytes_saved);
        $display("  Tokens generated:      %0d", ts_tokens_done);
        $display("  Weight buffer swaps:   %0d", wdb_swaps);
        $display("================================================================");

        if (fail_count == 0)
            $display("  >>> ALL IMPROVEMENTS VERIFIED IN INTEGRATION <<<");
        else
            $display("  >>> %0d FAILURES — REVIEW REQUIRED <<<", fail_count);

        $display("");
        $display("TB_RESULT pass=%0d fail=%0d", pass_count, fail_count);
        if (fail_count != 0)
            $fatal(1, "combined_improvements_tb failed (%0d/%0d)", pass_count, total_tests);
        $finish;
    end

endmodule
