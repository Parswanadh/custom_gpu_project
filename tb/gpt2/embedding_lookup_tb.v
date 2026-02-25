// ============================================================================
// Testbench: embedding_lookup_tb
// Tests token + position embedding lookup
// ============================================================================
`timescale 1ns / 1ps

module embedding_lookup_tb;

    parameter VS = 16;
    parameter MS = 8;
    parameter ED = 4;
    parameter DW = 16;

    reg                     clk, rst;
    reg                     load_token_emb, load_pos_emb;
    reg  [3:0]              load_token_idx;
    reg  [1:0]              load_dim_idx;
    reg  signed [DW-1:0]    load_data;
    reg  [2:0]              load_pos_idx;
    reg                     valid_in;
    reg  [3:0]              token_id;
    reg  [2:0]              position;
    wire [ED*DW-1:0]        emb_out;
    wire                    valid_out;

    embedding_lookup #(.VOCAB_SIZE(VS), .MAX_SEQ_LEN(MS), .EMBED_DIM(ED), .DATA_WIDTH(DW)) uut (
        .clk(clk), .rst(rst),
        .load_token_emb(load_token_emb), .load_token_idx(load_token_idx),
        .load_dim_idx(load_dim_idx), .load_data(load_data),
        .load_pos_emb(load_pos_emb), .load_pos_idx(load_pos_idx),
        .valid_in(valid_in), .token_id(token_id), .position(position),
        .emb_out(emb_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;
    integer ii;
    real val_real;

    task load_tok;
        input [3:0] tidx;
        input [1:0] didx;
        input signed [DW-1:0] val;
        begin
            @(negedge clk);
            load_token_emb = 1'b1; load_token_idx = tidx; load_dim_idx = didx; load_data = val;
            @(negedge clk);
            load_token_emb = 1'b0;
        end
    endtask

    task load_pos;
        input [2:0] pidx;
        input [1:0] didx;
        input signed [DW-1:0] val;
        begin
            @(negedge clk);
            load_pos_emb = 1'b1; load_pos_idx = pidx; load_dim_idx = didx; load_data = val;
            @(negedge clk);
            load_pos_emb = 1'b0;
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/embedding_lookup.vcd");
        $dumpvars(0, embedding_lookup_tb);
    end

    initial begin
        $display("============================================");
        $display("  Embedding Lookup Testbench (Q8.8)");
        $display("============================================");

        rst = 1; valid_in = 0;
        load_token_emb = 0; load_pos_emb = 0;
        load_token_idx = 0; load_dim_idx = 0; load_data = 0; load_pos_idx = 0;
        token_id = 0; position = 0;
        #25; rst = 0; #15;

        // Load token embedding for token 5: [1.0, 2.0, 3.0, 4.0]
        load_tok(4'd5, 2'd0, 16'sd256);   // dim 0 = 1.0
        load_tok(4'd5, 2'd1, 16'sd512);   // dim 1 = 2.0
        load_tok(4'd5, 2'd2, 16'sd768);   // dim 2 = 3.0
        load_tok(4'd5, 2'd3, 16'sd1024);  // dim 3 = 4.0

        // Load position embedding for pos 2: [0.1, 0.2, 0.3, 0.4]
        load_pos(3'd2, 2'd0, 16'sd26);    // ~0.1 in Q8.8
        load_pos(3'd2, 2'd1, 16'sd51);    // ~0.2
        load_pos(3'd2, 2'd2, 16'sd77);    // ~0.3
        load_pos(3'd2, 2'd3, 16'sd102);   // ~0.4

        #10;

        // Lookup: token 5 at position 2
        @(negedge clk);
        token_id = 4'd5; position = 3'd2; valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;
        #1;

        if (valid_out) begin
            $display("[PASS] Embedding lookup outputs:");
            for (ii = 0; ii < ED; ii = ii + 1) begin
                val_real = $itor($signed(emb_out[ii*DW +: DW])) / 256.0;
                $display("  emb[%0d] = %.3f", ii, val_real);
            end
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Embedding lookup - no valid output");
            fail_count = fail_count + 1;
        end

        // Test 2: Unknown token (defaults to 0)
        @(negedge clk);
        token_id = 4'd0; position = 3'd0; valid_in = 1'b1;
        @(negedge clk);
        valid_in = 1'b0;
        #1;

        if (valid_out) begin
            $display("[PASS] Zero token/position outputs:");
            for (ii = 0; ii < ED; ii = ii + 1) begin
                val_real = $itor($signed(emb_out[ii*DW +: DW])) / 256.0;
                $display("  emb[%0d] = %.3f", ii, val_real);
            end
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Zero token - no valid output");
            fail_count = fail_count + 1;
        end

        #20;
        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
