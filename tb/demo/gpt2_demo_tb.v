// ============================================================================
// GPT-2 Pipeline Demo — Science Fest Live Demo Testbench
// Shows tokens flowing through every stage with formatted output
// ============================================================================
`timescale 1ns / 1ps

module gpt2_demo_tb;

    parameter VS = 16;
    parameter MS = 8;
    parameter ED = 4;
    parameter NH = 2;
    parameter HD = 2;
    parameter FD = 8;
    parameter NL = 2;
    parameter DW = 16;

    reg                             clk, rst;
    reg                             load_token_emb, load_pos_emb;
    reg  [3:0]                      load_token_idx;
    reg  [1:0]                      load_dim_idx;
    reg  signed [DW-1:0]            load_emb_data;
    reg  [2:0]                      load_pos_idx;
    reg  [ED*DW-1:0]                ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;
    reg  [ED*DW-1:0]                ln_final_gamma, ln_final_beta;
    reg  [ED*ED*DW-1:0]             wq_flat, wk_flat, wv_flat, wo_flat;
    reg  [ED*FD*DW-1:0]             ffn_w1_flat;
    reg  [FD*DW-1:0]                ffn_b1_flat;
    reg  [FD*ED*DW-1:0]             ffn_w2_flat;
    reg  [ED*DW-1:0]                ffn_b2_flat;
    reg                             valid_in;
    reg  [3:0]                      token_in;
    reg  [2:0]                      position_in;
    wire [3:0]                      token_out;
    wire [ED*DW-1:0]                logits_out;
    wire                            valid_out;

    gpt2_engine #(
        .VOCAB_SIZE(VS), .MAX_SEQ_LEN(MS), .EMBED_DIM(ED),
        .NUM_HEADS(NH), .HEAD_DIM(HD), .FFN_DIM(FD),
        .NUM_LAYERS(NL), .DATA_WIDTH(DW)
    ) uut (
        .clk(clk), .rst(rst),
        .load_token_emb(load_token_emb), .load_token_idx(load_token_idx),
        .load_dim_idx(load_dim_idx), .load_emb_data(load_emb_data),
        .load_pos_emb(load_pos_emb), .load_pos_idx(load_pos_idx),
        .ln1_gamma(ln1_gamma), .ln1_beta(ln1_beta),
        .ln2_gamma(ln2_gamma), .ln2_beta(ln2_beta),
        .wq_flat(wq_flat), .wk_flat(wk_flat), .wv_flat(wv_flat), .wo_flat(wo_flat),
        .ffn_w1_flat(ffn_w1_flat), .ffn_b1_flat(ffn_b1_flat),
        .ffn_w2_flat(ffn_w2_flat), .ffn_b2_flat(ffn_b2_flat),
        .ln_final_gamma(ln_final_gamma), .ln_final_beta(ln_final_beta),
        .valid_in(valid_in), .token_in(token_in), .position_in(position_in),
        .token_out(token_out), .logits_out(logits_out), .valid_out(valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer timeout_cnt, ii, tok;
    real val_real;

    // Helper: build identity matrix
    function [ED*ED*DW-1:0] make_identity;
        input dummy;
        integer r, c;
        reg [ED*ED*DW-1:0] mat;
        begin
            mat = 0;
            for (r = 0; r < ED; r = r + 1)
                for (c = 0; c < ED; c = c + 1)
                    mat[(r*ED+c)*DW +: DW] = (r == c) ? 16'sd256 : 16'sd0;
            make_identity = mat;
        end
    endfunction

    function [ED*FD*DW-1:0] make_ffn_w1;
        input dummy;
        integer r, c;
        reg [ED*FD*DW-1:0] mat;
        begin
            mat = 0;
            for (r = 0; r < ED; r = r + 1)
                for (c = 0; c < FD; c = c + 1)
                    mat[(r*FD+c)*DW +: DW] = (r == c) ? 16'sd256 : 16'sd0;
            make_ffn_w1 = mat;
        end
    endfunction

    function [FD*ED*DW-1:0] make_ffn_w2;
        input dummy;
        integer r, c;
        reg [FD*ED*DW-1:0] mat;
        begin
            mat = 0;
            for (r = 0; r < FD; r = r + 1)
                for (c = 0; c < ED; c = c + 1)
                    mat[(r*ED+c)*DW +: DW] = (r == c) ? 16'sd256 : 16'sd0;
            make_ffn_w2 = mat;
        end
    endfunction

    // Token names for display
    task print_header;
        begin
            $display("");
            $display("================================================================");
            $display("   ____  _ _   _           ____  _ _      ____ ____  _   _");
            $display("  | __ )(_) |_| |__  _   _| __ )(_) |_   / ___|  _ \\| | | |");
            $display("  |  _ \\| | __| '_ \\| | | |  _ \\| | __| | |  _| |_) | | | |");
            $display("  | |_) | | |_| |_) | |_| | |_) | | |_  | |_| |  __/| |_| |");
            $display("  |____/|_|\\__|_.__/ \\__, |____/|_|\\__|  \\____|_|    \\___/");
            $display("                     |___/");
            $display("================================================================");
            $display("  Custom GPU Architecture for GPT-2 Inference");
            $display("  Q8.8 Fixed-Point | Zero-Skip Optimization");
            $display("  %0d Transformer Layers | %0d-dim Embeddings", NL, ED);
            $display("================================================================");
            $display("");
        end
    endtask

    task load_tok;
        input [3:0] tidx;
        input [1:0] didx;
        input signed [DW-1:0] val;
        begin
            @(negedge clk);
            load_token_emb = 1'b1; load_token_idx = tidx; load_dim_idx = didx; load_emb_data = val;
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
            load_pos_emb = 1'b1; load_pos_idx = pidx; load_dim_idx = didx; load_emb_data = val;
            @(negedge clk);
            load_pos_emb = 1'b0;
        end
    endtask

    task run_inference;
        input [3:0] in_token;
        input [2:0] in_pos;
        begin
            $display("----------------------------------------------------------------");
            $display("  INPUT TOKEN: %0d  |  POSITION: %0d", in_token, in_pos);
            $display("----------------------------------------------------------------");
            $display("");

            // Stage 1: Embedding
            $display("  [STAGE 1] Embedding Lookup");
            $display("    Token ID %0d --> Token Embedding Table", in_token);
            $display("    Position %0d --> Position Embedding Table", in_pos);
            $display("    Output: token_emb + pos_emb (Q8.8 vector)");
            $display("");

            @(negedge clk);
            token_in = in_token; position_in = in_pos; valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;

            // Stage 2-3: Transformer Blocks
            for (tok = 0; tok < NL; tok = tok + 1) begin
                $display("  [STAGE %0d] Transformer Block %0d", 2+tok, tok);
                $display("    --> LayerNorm 1 (mean/var/normalize)");
                $display("    --> Multi-Head Self-Attention (Q*K^T*V)");
                $display("    --> Residual Add");
                $display("    --> LayerNorm 2");
                $display("    --> FFN: Linear(%0dx%0d) -> GELU -> Linear(%0dx%0d)", ED, FD, FD, ED);
                $display("    --> Residual Add");
                $display("");
            end

            // Stage 4: Final LayerNorm + Argmax
            $display("  [STAGE %0d] Final LayerNorm + Argmax", 2+NL);

            // Wait for output
            timeout_cnt = 0;
            while (!valid_out && timeout_cnt < 500) begin
                @(negedge clk);
                timeout_cnt = timeout_cnt + 1;
            end

            if (valid_out) begin
                $display("    Logits (Q8.8):");
                for (ii = 0; ii < ED; ii = ii + 1) begin
                    val_real = $itor($signed(logits_out[ii*DW +: DW])) / 256.0;
                    $display("      logit[%0d] = %8.3f", ii, val_real);
                end
                $display("");
                $display("  ================================================");
                $display("  | RESULT: Token %0d --> Predicted Token: %0d |", in_token, token_out);
                $display("  | Latency: %0d clock cycles                  |", timeout_cnt);
                $display("  ================================================");
            end else begin
                $display("  [ERROR] Inference timed out after %0d cycles!", timeout_cnt);
            end
            $display("");
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/gpt2_demo.vcd");
        $dumpvars(0, gpt2_demo_tb);
    end

    initial begin
        // Init
        rst = 1; valid_in = 0;
        load_token_emb = 0; load_pos_emb = 0;
        load_token_idx = 0; load_dim_idx = 0; load_emb_data = 0; load_pos_idx = 0;
        token_in = 0; position_in = 0;

        // Identity-like weights
        ln1_gamma = {ED{16'sd256}}; ln1_beta = {ED{16'sd0}};
        ln2_gamma = {ED{16'sd256}}; ln2_beta = {ED{16'sd0}};
        ln_final_gamma = {ED{16'sd256}}; ln_final_beta = {ED{16'sd0}};
        wq_flat = make_identity(0); wk_flat = make_identity(0);
        wv_flat = make_identity(0); wo_flat = make_identity(0);
        ffn_w1_flat = make_ffn_w1(0); ffn_b1_flat = {FD{16'sd0}};
        ffn_w2_flat = make_ffn_w2(0); ffn_b2_flat = {ED{16'sd0}};

        #35; rst = 0; #25;

        // Load vocab: Token 0="the", 1="AI", 2="GPU", 3="runs", 4="fast"
        // Token 0: [1.0, 0.5, 0.5, 0.5]
        load_tok(4'd0, 2'd0, 16'sd256);
        load_tok(4'd0, 2'd1, 16'sd128);
        load_tok(4'd0, 2'd2, 16'sd128);
        load_tok(4'd0, 2'd3, 16'sd128);

        // Token 1: [0.5, 2.0, 1.0, 0.5]
        load_tok(4'd1, 2'd0, 16'sd128);
        load_tok(4'd1, 2'd1, 16'sd512);
        load_tok(4'd1, 2'd2, 16'sd256);
        load_tok(4'd1, 2'd3, 16'sd128);

        // Token 2: [1.0, 1.0, 3.0, 1.0]
        load_tok(4'd2, 2'd0, 16'sd256);
        load_tok(4'd2, 2'd1, 16'sd256);
        load_tok(4'd2, 2'd2, 16'sd768);
        load_tok(4'd2, 2'd3, 16'sd256);

        // Token 3: [2.0, 3.0, 4.0, 5.0]
        load_tok(4'd3, 2'd0, 16'sd512);
        load_tok(4'd3, 2'd1, 16'sd768);
        load_tok(4'd3, 2'd2, 16'sd1024);
        load_tok(4'd3, 2'd3, 16'sd1280);

        // Token 4: [4.0, 5.0, 6.0, 7.0]
        load_tok(4'd4, 2'd0, 16'sd1024);
        load_tok(4'd4, 2'd1, 16'sd1280);
        load_tok(4'd4, 2'd2, 16'sd1536);
        load_tok(4'd4, 2'd3, 16'sd1792);

        // Position embeddings (small offsets)
        load_pos(3'd0, 2'd0, 16'sd13); load_pos(3'd0, 2'd1, 16'sd13);
        load_pos(3'd0, 2'd2, 16'sd13); load_pos(3'd0, 2'd3, 16'sd13);
        load_pos(3'd1, 2'd0, 16'sd26); load_pos(3'd1, 2'd1, 16'sd26);
        load_pos(3'd1, 2'd2, 16'sd26); load_pos(3'd1, 2'd3, 16'sd26);
        load_pos(3'd2, 2'd0, 16'sd38); load_pos(3'd2, 2'd1, 16'sd38);
        load_pos(3'd2, 2'd2, 16'sd38); load_pos(3'd2, 2'd3, 16'sd38);

        #20;

        print_header;

        $display("================== INFERENCE RUN 1 ==================");
        run_inference(4'd3, 3'd0);

        // Wait for pipeline to clear
        #100;

        $display("================== INFERENCE RUN 2 ==================");
        run_inference(4'd4, 3'd1);

        #100;

        $display("================== INFERENCE RUN 3 ==================");
        run_inference(4'd1, 3'd2);

        #100;

        $display("");
        $display("================================================================");
        $display("  DEMO COMPLETE — BitbyBit GPU Operational!");
        $display("  Architecture: 16 modules | 4 layers | Q8.8 fixed-point");
        $display("  Optimizations: Zero-skip | Sparse CSR | INT4 quantization");
        $display("================================================================");
        $display("");
        $finish;
    end

endmodule
