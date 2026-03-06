`timescale 1ns / 1ps
module medusa_head_predictor_tb;
    parameter NUM_HEADS=3, HIDDEN_DIM=8, VOCAB_BITS=8, DATA_WIDTH=16;
    reg clk, rst, valid_in, weight_load_en, verify_en;
    reg [$clog2(NUM_HEADS)-1:0] head_sel;
    reg [$clog2(HIDDEN_DIM)-1:0] weight_idx;
    reg signed [DATA_WIDTH-1:0] weight_data;
    reg [HIDDEN_DIM*DATA_WIDTH-1:0] hidden_state;
    wire [NUM_HEADS*VOCAB_BITS-1:0] predicted_tokens;
    wire valid_out;
    reg [NUM_HEADS*VOCAB_BITS-1:0] actual_tokens;
    wire [NUM_HEADS-1:0] accept_mask;
    wire [$clog2(NUM_HEADS):0] accepted_count;
    wire [31:0] total_predictions, total_accepted;

    medusa_head_predictor #(.NUM_HEADS(NUM_HEADS), .HIDDEN_DIM(HIDDEN_DIM),
        .VOCAB_BITS(VOCAB_BITS), .DATA_WIDTH(DATA_WIDTH)
    ) dut (.clk(clk), .rst(rst), .valid_in(valid_in), .hidden_state(hidden_state),
        .weight_load_en(weight_load_en), .head_sel(head_sel), .weight_idx(weight_idx),
        .weight_data(weight_data), .predicted_tokens(predicted_tokens), .valid_out(valid_out),
        .verify_en(verify_en), .actual_tokens(actual_tokens), .accept_mask(accept_mask),
        .accepted_count(accepted_count), .total_predictions(total_predictions),
        .total_accepted(total_accepted));

    always #5 clk = ~clk;
    integer tp=0, tt=0, h, d;
    
    task wait_done; input integer n;
        integer c; begin for(c=0;c<n;c=c+1) begin @(negedge clk); if(valid_out) c=n; end end
    endtask

    initial begin
        clk=0; rst=1; valid_in=0; weight_load_en=0; verify_en=0;
        head_sel=0; weight_idx=0; weight_data=0; hidden_state=0; actual_tokens=0;
        @(negedge clk); @(negedge clk); rst=0; @(negedge clk);

        $display("=================================================");
        $display("   MEDUSA Multi-Head Draft Predictor Tests");
        $display("   Paper: 'MEDUSA' (Cai et al., 2024)");
        $display("   Config: %0d draft heads, %0d hidden dim", NUM_HEADS, HIDDEN_DIM);
        $display("=================================================");

        // Load weights for all heads
        for (h = 0; h < NUM_HEADS; h = h + 1) begin
            for (d = 0; d < HIDDEN_DIM; d = d + 1) begin
                @(negedge clk);
                weight_load_en = 1; head_sel = h; weight_idx = d;
                weight_data = (h + 1) * (d + 1);  // Different pattern per head
            end
        end
        @(negedge clk); weight_load_en = 0;

        // TEST 1: Generate predictions from all heads
        tt = tt + 1;
        for (d = 0; d < HIDDEN_DIM; d = d + 1)
            hidden_state[d*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
        valid_in = 1; @(negedge clk); valid_in = 0;
        wait_done(30);
        if (valid_out) begin
            $display("[PASS] Test 1: 3 heads predicted tokens: T[t+1]=%0d, T[t+2]=%0d, T[t+3]=%0d",
                predicted_tokens[7:0], predicted_tokens[15:8], predicted_tokens[23:16]);
            tp = tp + 1;
        end else $display("[FAIL] Test 1");

        // TEST 2: Different heads produce different tokens (different weights)
        tt = tt + 1;
        if (predicted_tokens[7:0] != predicted_tokens[15:8]) begin
            $display("[PASS] Test 2: Different heads → different predictions (head diversity)");
            tp = tp + 1;
        end else $display("[FAIL] Test 2: Heads predict same token");

        // TEST 3: Verification — match predictions as "actual"
        tt = tt + 1;
        actual_tokens = predicted_tokens;  // Perfect match
        verify_en = 1; @(negedge clk); verify_en = 0; @(negedge clk);
        if (accept_mask == {NUM_HEADS{1'b1}}) begin
            $display("[PASS] Test 3: Verification — all %0d heads accepted (100%% accuracy)", NUM_HEADS);
            tp = tp + 1;
        end else $display("[FAIL] Test 3: accept_mask=%b", accept_mask);

        // TEST 4: Verification — partial match
        tt = tt + 1;
        actual_tokens = predicted_tokens;
        actual_tokens[7:0] = predicted_tokens[7:0] + 1;  // Force head 0 to mismatch
        verify_en = 1; @(negedge clk); verify_en = 0; @(negedge clk);
        if (accept_mask[0] == 1'b0 && accept_mask[1] == 1'b1 && accept_mask[2] == 1'b1) begin
            $display("[PASS] Test 4: Partial verification — head0 rejected, heads 1,2 accepted");
            tp = tp + 1;
        end else $display("[FAIL] Test 4: accept_mask=%b", accept_mask);

        $display("=================================================");
        $display("   MEDUSA Tests: %0d / %0d PASSED", tp, tt);
        $display("=================================================");
        #10 $finish;
    end
    initial begin #50000; $display("TIMEOUT"); $finish; end
endmodule
