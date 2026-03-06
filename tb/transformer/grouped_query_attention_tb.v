`timescale 1ns / 1ps
module grouped_query_attention_tb;
    parameter EMBED_DIM=8, NUM_Q_HEADS=4, NUM_KV_HEADS=2, HEAD_DIM=4, MAX_SEQ_LEN=16, DATA_WIDTH=16;
    reg clk, rst, valid_in;
    reg [NUM_Q_HEADS*HEAD_DIM*DATA_WIDTH-1:0] q_heads;
    reg [NUM_KV_HEADS*HEAD_DIM*DATA_WIDTH-1:0] k_heads, v_heads;
    wire [NUM_Q_HEADS*DATA_WIDTH-1:0] attention_scores;
    wire valid_out;
    wire [15:0] kv_memory_saved;

    grouped_query_attention #(.EMBED_DIM(EMBED_DIM), .NUM_Q_HEADS(NUM_Q_HEADS),
        .NUM_KV_HEADS(NUM_KV_HEADS), .HEAD_DIM(HEAD_DIM), .MAX_SEQ_LEN(MAX_SEQ_LEN)
    ) dut (.clk(clk), .rst(rst), .valid_in(valid_in),
        .q_heads(q_heads), .k_heads(k_heads), .v_heads(v_heads),
        .attention_scores(attention_scores), .valid_out(valid_out),
        .kv_memory_saved(kv_memory_saved));

    always #5 clk = ~clk;
    integer tp=0, tt=0, i;
    
    initial begin
        clk=0; rst=1; valid_in=0; q_heads=0; k_heads=0; v_heads=0;
        @(negedge clk); @(negedge clk); rst=0; @(negedge clk);

        $display("=================================================");
        $display("   Grouped Query Attention (GQA) Tests");
        $display("   Paper: Google 2023, ISOCC 2025 FPGA");
        $display("   Config: %0d Q heads, %0d KV heads (%0dx savings)",
                 NUM_Q_HEADS, NUM_KV_HEADS, NUM_Q_HEADS/NUM_KV_HEADS);
        $display("=================================================");

        // TEST 1: Basic GQA with identical Q/K → positive scores
        tt = tt + 1;
        for (i = 0; i < NUM_Q_HEADS*HEAD_DIM; i = i + 1)
            q_heads[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
        for (i = 0; i < NUM_KV_HEADS*HEAD_DIM; i = i + 1)
            k_heads[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
        valid_in = 1; @(negedge clk);  // Output registered here
        if (valid_out) begin
            $display("[PASS] Test 1: GQA scores computed — S[0]=%0d, S[1]=%0d, S[2]=%0d, S[3]=%0d",
                $signed(attention_scores[15:0]), $signed(attention_scores[31:16]),
                $signed(attention_scores[47:32]), $signed(attention_scores[63:48]));
            tp = tp + 1;
        end else $display("[FAIL] Test 1");
        valid_in = 0; @(negedge clk);

        // TEST 2: Q heads in same group get same KV → same scores
        tt = tt + 1;
        if ($signed(attention_scores[15:0]) == $signed(attention_scores[31:16])) begin
            $display("[PASS] Test 2: Q[0] and Q[1] share KV[0] → same scores (%0d == %0d)",
                $signed(attention_scores[15:0]), $signed(attention_scores[31:16]));
            tp = tp + 1;
        end else $display("[FAIL] Test 2: Scores differ within group");

        // TEST 3: KV memory savings calculation
        tt = tt + 1;
        if (kv_memory_saved == (NUM_Q_HEADS - NUM_KV_HEADS) * HEAD_DIM * MAX_SEQ_LEN) begin
            $display("[PASS] Test 3: KV memory saved = %0d entries (vs MHA)", kv_memory_saved);
            $display("         MHA would need %0d entries, GQA needs %0d (%0dx reduction)",
                NUM_Q_HEADS * HEAD_DIM * MAX_SEQ_LEN,
                NUM_KV_HEADS * HEAD_DIM * MAX_SEQ_LEN,
                NUM_Q_HEADS / NUM_KV_HEADS);
            tp = tp + 1;
        end else $display("[FAIL] Test 3: savings=%0d", kv_memory_saved);

        // TEST 4: Different KV groups → different scores
        tt = tt + 1;
        for (i = 0; i < HEAD_DIM; i = i + 1)
            k_heads[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd256;
        for (i = HEAD_DIM; i < NUM_KV_HEADS*HEAD_DIM; i = i + 1)
            k_heads[i*DATA_WIDTH +: DATA_WIDTH] = 16'sd128;
        valid_in = 1; @(negedge clk);
        if (valid_out && $signed(attention_scores[15:0]) != $signed(attention_scores[47:32])) begin
            $display("[PASS] Test 4: Different KV groups → different scores (group0=%0d, group1=%0d)",
                $signed(attention_scores[15:0]), $signed(attention_scores[47:32]));
            tp = tp + 1;
        end else $display("[FAIL] Test 4");
        valid_in = 0;

        $display("=================================================");
        $display("   GQA Tests: %0d / %0d PASSED", tp, tt);
        $display("=================================================");
        #10 $finish;
    end
    initial begin #10000; $display("TIMEOUT"); $finish; end
endmodule
