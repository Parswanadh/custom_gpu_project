`timescale 1ns / 1ps

// ============================================================================
// Full-model inference emulation using the hardwired MINI imprint path.
//
// Scope (measured):
//   [Imprinted Embedding] -> 12x [imprinted_mini_transformer_core] -> [MEDUSA]
//
// This bench is used for measured base-vs-imprint token-throughput comparison.
// ============================================================================
module full_model_inference_imprint_tb;

    reg clk, rst;
    always #5 clk = ~clk; // 100 MHz

    parameter DIM = 8;
    parameter NUM_LAYERS = 12;
    parameter TOKEN_SPACE = 16;
    parameter POSITION_SPACE = 8;

    // ---------------------------------------------------------------------
    // Imprinted embedding ROM (profile 2'b01 = MINI hardwired path)
    // ---------------------------------------------------------------------
    reg [15:0] token_id;
    reg [15:0] position;
    reg [1:0] profile_sel;
    wire [DIM*16-1:0] imprint_embedding;
    wire profile_supported;

    imprinted_embedding_rom #(
        .DIM(DIM),
        .DATA_WIDTH(16)
    ) u_imprint_rom (
        .token_id(token_id),
        .position(position),
        .profile_sel(profile_sel),
        .embedding_out(imprint_embedding),
        .profile_supported(profile_supported)
    );

    // ---------------------------------------------------------------------
    // Hardwired mini core (reused 12x)
    // ---------------------------------------------------------------------
    reg core_start;
    reg [DIM*16-1:0] core_in;
    reg [5:0] core_pos;
    wire core_done;
    wire [DIM*16-1:0] core_out;
    wire [15:0] core_cycles;

    imprinted_mini_transformer_core #(
        .DIM(DIM),
        .DATA_W(16),
        .LATENCY(8)
    ) u_imprint_core (
        .clk(clk),
        .rst(rst),
        .start(core_start),
        .token_embedding(core_in),
        .position(core_pos),
        .done(core_done),
        .output_vector(core_out),
        .cycles_used(core_cycles)
    );

    // ---------------------------------------------------------------------
    // MEDUSA predictor
    // ---------------------------------------------------------------------
    reg med_valid, med_wld;
    reg [1:0] med_hs;
    reg [2:0] med_wi;
    reg signed [15:0] med_wd;
    reg [DIM*16-1:0] med_hidden_state;
    wire [3*8-1:0] med_tokens;
    wire med_done;
    wire [31:0] med_total_pred;

    medusa_head_predictor #(
        .NUM_HEADS(3),
        .HIDDEN_DIM(DIM)
    ) u_medusa (
        .clk(clk),
        .rst(rst),
        .valid_in(med_valid),
        .hidden_state(med_hidden_state),
        .weight_load_en(med_wld),
        .head_sel(med_hs),
        .weight_idx(med_wi),
        .weight_data(med_wd),
        .predicted_tokens(med_tokens),
        .valid_out(med_done),
        .verify_en(1'b0),
        .actual_tokens(24'd0),
        .total_predictions(med_total_pred)
    );

    // ---------------------------------------------------------------------
    // Benchmark bookkeeping
    // ---------------------------------------------------------------------
    integer i, h, d, layer_num;
    integer t_start_time, t_end_time;
    integer emb_start, emb_end, emb_cycles;
    integer total_layer_cycles;
    integer med_start, med_end, med_cycles;
    integer full_inference_cycles;
    integer tests_passed, tests_total;
    integer runtime_token_id, runtime_position;
    reg layer_completed, med_completed;
    reg [DIM*16-1:0] layer_outputs [0:NUM_LAYERS];

    initial begin
        clk = 0;
        rst = 1;

        token_id = 0;
        position = 0;
        profile_sel = 2'b01;

        core_start = 0;
        core_in = 0;
        core_pos = 0;

        med_valid = 0;
        med_wld = 0;
        med_hs = 0;
        med_wi = 0;
        med_wd = 0;
        med_hidden_state = 0;

        tests_passed = 0;
        tests_total = 0;
        total_layer_cycles = 0;

        @(negedge clk); @(negedge clk); rst = 0;
        repeat(3) @(negedge clk);

        $display("");
        $display("================================================================");
        $display("  MINI IMPRINT 12-LAYER INFERENCE EMULATION");
        $display("  Architecture: hardwired embedding + 12x fixed-latency mini core");
        $display("  Pipeline: Imprint Embedding -> 12x Mini Core -> MEDUSA");
        $display("  Clock: 100 MHz | ALL numbers from simulation");
        $display("================================================================");

        // ------------------------------------------------------
        // Load MEDUSA head weights
        // ------------------------------------------------------
        for (h = 0; h < 3; h = h + 1) begin
            for (d = 0; d < DIM; d = d + 1) begin
                @(negedge clk);
                med_wld = 1;
                med_hs = h;
                med_wi = d;
                med_wd = (h + 1) * (d + 1) * 3;
            end
        end
        @(negedge clk); med_wld = 0;
        repeat(2) @(negedge clk);

        runtime_token_id = 5;
        runtime_position = 2;
        if (!$value$plusargs("TOKEN_ID=%d", runtime_token_id))
            runtime_token_id = 5;
        if (!$value$plusargs("POSITION=%d", runtime_position))
            runtime_position = 2;
        if (runtime_token_id < 0 || runtime_token_id >= TOKEN_SPACE) begin
            $display("[WARN] TOKEN_ID=%0d out of baseline range [0,%0d]. Using default 5.",
                     runtime_token_id, TOKEN_SPACE - 1);
            runtime_token_id = 5;
        end
        if (runtime_position < 0 || runtime_position >= POSITION_SPACE) begin
            $display("[WARN] POSITION=%0d out of baseline range [0,%0d]. Using default 2.",
                     runtime_position, POSITION_SPACE - 1);
            runtime_position = 2;
        end

        // ------------------------------------------------------
        // Full inference (same token/position defaults as baseline bench)
        // ------------------------------------------------------
        t_start_time = $time;

        emb_start = $time / 10;
        token_id = runtime_token_id[15:0];
        position = runtime_position[15:0];
        profile_sel = 2'b01;
        @(negedge clk);
        emb_end = $time / 10;
        emb_cycles = emb_end - emb_start;
        layer_outputs[0] = imprint_embedding;

        $display("");
        $display("  [Embedding]  Token %0d -> imprint vector (%0d cycle)", runtime_token_id, emb_cycles);
        $display("    emb_out[0:3] = [%0d, %0d, %0d, %0d]",
                 $signed(layer_outputs[0][0*16 +: 16]),
                 $signed(layer_outputs[0][1*16 +: 16]),
                 $signed(layer_outputs[0][2*16 +: 16]),
                 $signed(layer_outputs[0][3*16 +: 16]));

        $display("");
        $display("  %-6s %-8s", "Layer", "Total");
        $display("  %-6s %-8s", "-----", "------");

        for (layer_num = 0; layer_num < NUM_LAYERS; layer_num = layer_num + 1) begin
            core_in = layer_outputs[layer_num];
            core_pos = position[5:0];

            core_start = 1'b1;
            @(negedge clk);
            core_start = 1'b0;

            layer_completed = 1'b0;
            begin : wait_layer
                integer t;
                for (t = 0; t < 200; t = t + 1) begin
                    @(negedge clk);
                    if (core_done) begin
                        layer_completed = 1'b1;
                        t = 200;
                    end
                end
            end

            if (!layer_completed) begin
                $display("[FAIL] Imprint layer %0d timeout", layer_num + 1);
                $fatal(1);
            end

            layer_outputs[layer_num + 1] = core_out;
            total_layer_cycles = total_layer_cycles + core_cycles;

            $display("  L%-5d %4dcy", layer_num + 1, core_cycles);
        end

        med_hidden_state = layer_outputs[NUM_LAYERS];
        med_start = $time / 10;
        med_valid = 1'b1;
        @(negedge clk);
        med_valid = 1'b0;

        med_completed = 1'b0;
        begin : wait_med
            integer t;
            for (t = 0; t < 100; t = t + 1) begin
                @(negedge clk);
                if (med_done) begin
                    med_completed = 1'b1;
                    t = 100;
                end
            end
        end

        if (!med_completed) begin
            $display("[FAIL] MEDUSA timeout");
            $fatal(1);
        end

        med_end = $time / 10;
        med_cycles = med_end - med_start;

        t_end_time = $time;
        full_inference_cycles = (t_end_time - t_start_time) / 10;

        $display("");
        $display("  [MEDUSA]     3 draft token predictions (%0d cycles)", med_cycles);
        $display("    Predicted tokens: [%0d, %0d, %0d]",
                 med_tokens[0*8 +: 8], med_tokens[1*8 +: 8], med_tokens[2*8 +: 8]);

        $display("");
        $display("================================================================");
        $display("  MINI IMPRINT INFERENCE RESULTS (ALL measured)");
        $display("================================================================");
        $display("");
        $display("  STAGE                    CYCLES    TIME @100MHz");
        $display("  -----                    ------    ------------");
        $display("  Imprint Embedding        %4d cy    %4d ns", emb_cycles, emb_cycles * 10);
        $display("  12x Mini Core Layers     %4d cy    %4d ns", total_layer_cycles, total_layer_cycles * 10);
        $display("    Per-layer average      %4d cy    %4d ns",
                 total_layer_cycles / NUM_LAYERS,
                 (total_layer_cycles / NUM_LAYERS) * 10);
        $display("  MEDUSA Prediction        %4d cy    %4d ns", med_cycles, med_cycles * 10);
        $display("  -----------------------------------------");
        $display("  TOTAL INFERENCE          %4d cy    %4d ns",
                 full_inference_cycles,
                 full_inference_cycles * 10);
        $display("");
        $display("  Throughput:");
        $display("    Single token:    %0d cycles = %0d ns = %.3f us",
                 full_inference_cycles,
                 full_inference_cycles * 10,
                 full_inference_cycles * 0.01);
        $display("    Tokens/second:   ~%0d (100 MHz FPGA)",
                 100000000 / full_inference_cycles);
        $display("    MEDUSA draft:    3 tokens predicted -> effective %0d tok/s",
                 3 * 100000000 / full_inference_cycles);
        $display("");

        // ------------------------------------------------------
        // Checks
        // ------------------------------------------------------
        tests_total = tests_total + 1;
        if (profile_supported && layer_outputs[0] != 0) begin
            $display("[PASS] Test 1: MINI imprint embedding active and non-zero");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 1: MINI imprint embedding invalid");
        end

        tests_total = tests_total + 1;
        if (total_layer_cycles == (NUM_LAYERS * 8)) begin
            $display("[PASS] Test 2: 12 layers completed at fixed 8-cycle latency");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 2: Layer cycles mismatch (%0d)", total_layer_cycles);
        end

        tests_total = tests_total + 1;
        if (layer_outputs[NUM_LAYERS] != layer_outputs[0]) begin
            $display("[PASS] Test 3: Data transformed across layers");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 3: Data unchanged after 12 layers");
        end

        tests_total = tests_total + 1;
        if (med_completed) begin
            $display("[PASS] Test 4: MEDUSA produced predictions");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 4: MEDUSA not completed");
        end

        tests_total = tests_total + 1;
        if (full_inference_cycles > 0 && full_inference_cycles < 5000) begin
            $display("[PASS] Test 5: Total inference cycles in expected range");
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 5: Total inference cycles out of range (%0d)", full_inference_cycles);
        end

        $display("");
        $display("================================================================");
        $display("  MINI Imprint Inference Tests: %0d / %0d PASSED", tests_passed, tests_total);
        $display("================================================================");

        if (tests_passed != tests_total)
            $fatal(1, "full_model_inference_imprint_tb failed (%0d/%0d)",
                   tests_passed, tests_total);

        #20 $finish;
    end

    initial begin
        #10000000;
        $display("[FATAL] TIMEOUT");
        $fatal(1);
    end

endmodule

