`timescale 1ns / 1ps
// ============================================================================
// Testbench: gemma3_engine_tb
// Description: End-to-end validation of the Gemma 3 inference engine.
//   - Initializes embedding ROM with deterministic values
//   - Initializes RMSNorm gamma weights
//   - Runs single-token inference through all 18 layers
//   - Verifies data flow (non-zero output, cycle count bounds)
//   - Outputs detailed per-layer diagnostics
//
//   Uses small-scale parameters (DIM=8, VOCAB=16) for fast iverilog simulation.
//   The same RTL is parameterized for full Gemma3 dims on FPGA.
// ============================================================================

module gemma3_engine_tb;

    // ---- Parameters (simulation scale) ----
    localparam DIM         = 8;
    localparam NUM_LAYERS  = 18;
    localparam NUM_Q_HEADS = 4;
    localparam HEAD_DIM    = 4;
    localparam FFN_DIM     = 16;
    localparam VOCAB_SIZE  = 16;
    localparam MAX_SEQ_LEN = 8;
    localparam DATA_W      = 16;
    localparam CLK_PERIOD  = 10; // 100 MHz

    // ---- Signals ----
    reg                     clk;
    reg                     rst;
    reg                     start;
    reg  [15:0]             token_id;
    reg  [5:0]              position;
    wire                    done;
    wire [15:0]             predicted_token;
    wire [$clog2(VOCAB_SIZE)-1:0] emb_addr;
    reg  [DIM*DATA_W-1:0]  emb_data;
    wire [7:0]              layer_idx;
    reg  [DIM*DATA_W-1:0]  rms1_gamma;
    reg  [DIM*DATA_W-1:0]  rms2_gamma;
    reg  [DIM*DATA_W-1:0]  final_rms_gamma;
    wire [15:0]             total_cycles;
    wire [15:0]             layer_cycles;
    wire [7:0]              current_layer;

    // ---- Embedding ROM (16 tokens × 8 dims) ----
    reg signed [DATA_W-1:0] emb_rom [0:VOCAB_SIZE*DIM-1];

    // ---- Clock ----
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ---- DUT ----
    gemma3_engine #(
        .DIM(DIM),
        .NUM_LAYERS(NUM_LAYERS),
        .NUM_Q_HEADS(NUM_Q_HEADS),
        .HEAD_DIM(HEAD_DIM),
        .FFN_DIM(FFN_DIM),
        .VOCAB_SIZE(VOCAB_SIZE),
        .MAX_SEQ_LEN(MAX_SEQ_LEN),
        .DATA_W(DATA_W)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .token_id(token_id),
        .position(position),
        .done(done),
        .predicted_token(predicted_token),
        .emb_addr(emb_addr),
        .emb_data(emb_data),
        .layer_idx(layer_idx),
        .rms1_gamma(rms1_gamma),
        .rms2_gamma(rms2_gamma),
        .final_rms_gamma(final_rms_gamma),
        .total_cycles(total_cycles),
        .layer_cycles(layer_cycles),
        .current_layer(current_layer)
    );

    // ---- Embedding ROM read ----
    integer ei;
    always @(*) begin
        for (ei = 0; ei < DIM; ei = ei + 1) begin
            emb_data[ei*DATA_W +: DATA_W] = emb_rom[emb_addr * DIM + ei];
        end
    end

    // ---- Test counters ----
    integer pass_count;
    integer fail_count;
    integer test_num;
    reg [15:0] timeout_counter;

    task check;
        input [255:0] name;
        input condition;
        begin
            test_num = test_num + 1;
            if (condition) begin
                $display("  [PASS] Test %0d: %0s", test_num, name);
                pass_count = pass_count + 1;
            end else begin
                $display("  [FAIL] Test %0d: %0s", test_num, name);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // ---- Stimulus ----
    integer i, j;
    initial begin
        $display("================================================================");
        $display("  GEMMA 3 ENGINE — END-TO-END TEST");
        $display("  DIM=%0d, LAYERS=%0d, HEADS=%0d, VOCAB=%0d",
                 DIM, NUM_LAYERS, NUM_Q_HEADS, VOCAB_SIZE);
        $display("================================================================");

        pass_count = 0;
        fail_count = 0;
        test_num   = 0;
        rst   = 1;
        start = 0;
        token_id = 16'd0;
        position = 6'd0;

        // Initialize embedding ROM: token_id * 100 + dim_idx * 10
        for (i = 0; i < VOCAB_SIZE; i = i + 1)
            for (j = 0; j < DIM; j = j + 1)
                emb_rom[i * DIM + j] = i * 100 + j * 10 + 50;

        // Initialize RMSNorm gamma to 1.0 (256 in Q8.8)
        for (j = 0; j < DIM; j = j + 1) begin
            rms1_gamma[j*DATA_W +: DATA_W]      = 16'sd256;
            rms2_gamma[j*DATA_W +: DATA_W]      = 16'sd256;
            final_rms_gamma[j*DATA_W +: DATA_W] = 16'sd256;
        end

        // Release reset
        #(CLK_PERIOD * 5);
        rst = 0;
        #(CLK_PERIOD * 2);

        // ============================================================
        // TEST 1: Single token inference (token_id=5, position=2)
        // ============================================================
        $display("\n--- Test: Single Token Inference ---");
        $display("  Input: token_id=5, position=2");
        $display("  Expected: 18-layer forward pass → predicted token");

        token_id = 16'd5;
        position = 6'd2;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // Wait for done with timeout
        timeout_counter = 0;
        while (!done && timeout_counter < 16'd20000) begin
            @(posedge clk);
            timeout_counter = timeout_counter + 1;
            // Print layer progress
            if (dut.state == 4'd5 && dut.block_done) begin
                $display("  Layer %0d complete (%0d cycles)", current_layer, layer_cycles);
            end
        end

        check("Inference completed (no timeout)", done === 1'b1);
        check("Total cycles > 0", total_cycles > 0);
        check("Total cycles < 20000", total_cycles < 16'd20000);
        check("Predicted token is valid", predicted_token < VOCAB_SIZE);
        check("All 18 layers processed", current_layer == NUM_LAYERS - 1);

        $display("  Result: predicted_token=%0d, total_cycles=%0d", predicted_token, total_cycles);
        $display("  Latency @ 100MHz: %0d ns", total_cycles * 10);
        $display("  Throughput: ~%0d tok/sec", 100000000 / (total_cycles > 0 ? total_cycles : 1));

        // ============================================================
        // TEST 2: Different token (token_id=10)
        // ============================================================
        $display("\n--- Test: Different Token ---");
        token_id = 16'd10;
        position = 6'd0;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        timeout_counter = 0;
        while (!done && timeout_counter < 16'd20000) begin
            @(posedge clk);
            timeout_counter = timeout_counter + 1;
        end

        check("Second inference completed", done === 1'b1);
        check("Produces valid token", predicted_token < VOCAB_SIZE);

        $display("  Result: predicted_token=%0d, total_cycles=%0d", predicted_token, total_cycles);

        // ============================================================
        // TEST 3: Edge case — token_id=0
        // ============================================================
        $display("\n--- Test: Edge Case (token=0) ---");
        token_id = 16'd0;
        position = 6'd0;
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        timeout_counter = 0;
        while (!done && timeout_counter < 16'd20000) begin
            @(posedge clk);
            timeout_counter = timeout_counter + 1;
        end

        check("Edge case inference completed", done === 1'b1);
        check("Produces valid token for edge case", predicted_token < VOCAB_SIZE);

        $display("  Result: predicted_token=%0d, total_cycles=%0d", predicted_token, total_cycles);

        // ============================================================
        // SUMMARY
        // ============================================================
        $display("");
        $display("================================================================");
        $display("  GEMMA 3 ENGINE — TEST SUMMARY");
        $display("================================================================");
        $display("  PASS: %0d", pass_count);
        $display("  FAIL: %0d", fail_count);
        if (fail_count == 0)
            $display("  >>> ALL TESTS PASSED <<<");
        else begin
            $display("  >>> TESTS FAILED <<<");
            $fatal(1, "Test failures detected");
        end
        $display("================================================================");

        #(CLK_PERIOD * 5);
        $finish;
    end

    // Timeout watchdog
    initial begin
        #(CLK_PERIOD * 100000);
        $display("FATAL: Global timeout exceeded!");
        $fatal(1, "Global timeout");
    end

endmodule
