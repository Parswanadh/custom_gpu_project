`timescale 1ns / 1ps

module layer_pipeline_controller_tb;

    parameter NUM_STAGES = 5;
    parameter TOKEN_WIDTH = 8;

    reg clk, rst;
    always #5 clk = ~clk;

    reg token_valid;
    reg [TOKEN_WIDTH-1:0] token_in;
    wire token_ready;
    
    reg [NUM_STAGES*8-1:0] stage_cycles_packed;
    
    wire [NUM_STAGES-1:0] stage_active;
    wire [NUM_STAGES*TOKEN_WIDTH-1:0] stage_tokens;
    wire [NUM_STAGES*8-1:0] stage_progress_packed;
    wire token_out_valid;
    wire [TOKEN_WIDTH-1:0] token_out;
    wire [31:0] tokens_processed, total_cycles, pipeline_stalls;
    wire pipeline_full;

    layer_pipeline_controller #(.NUM_STAGES(NUM_STAGES)) uut (
        .clk(clk), .rst(rst),
        .token_valid(token_valid), .token_in(token_in),
        .token_ready(token_ready),
        .stage_cycles_packed(stage_cycles_packed),
        .stage_active(stage_active), .stage_tokens(stage_tokens),
        .stage_progress_packed(stage_progress_packed),
        .token_out_valid(token_out_valid), .token_out(token_out),
        .tokens_processed(tokens_processed), .total_cycles(total_cycles),
        .pipeline_stalls(pipeline_stalls), .pipeline_full(pipeline_full)
    );

    integer tests_passed, tests_total;
    integer i;
    integer send_idx, recv_idx;
    integer accepted_count;
    integer ready_low_cycles;
    integer mismatch_count;
    integer extra_output_count;
    integer drain_timeout;
    integer seed;
    reg [TOKEN_WIDTH-1:0] expected_token;
    localparam integer MAX_SCOREBOARD_TOKENS = 512;
    reg [TOKEN_WIDTH-1:0] expected_tokens [0:MAX_SCOREBOARD_TOKENS-1];
    
    integer outputs_seen;
    integer last_out_cycle;
    integer max_gap;

    task scoreboard_reset;
    begin
        send_idx = 0;
        recv_idx = 0;
        accepted_count = 0;
        ready_low_cycles = 0;
        mismatch_count = 0;
        extra_output_count = 0;
    end
    endtask

    task scoreboard_record_accept;
        input [TOKEN_WIDTH-1:0] accepted_token;
    begin
        if (accepted_count < MAX_SCOREBOARD_TOKENS) begin
            expected_tokens[accepted_count] = accepted_token;
            accepted_count = accepted_count + 1;
        end else begin
            mismatch_count = mismatch_count + 1;
            $display("[FAIL] Scoreboard overflow at accepted_count=%0d", accepted_count);
        end
    end
    endtask

    task scoreboard_check_output;
    begin
        if (token_out_valid) begin
            if (recv_idx >= accepted_count) begin
                extra_output_count = extra_output_count + 1;
                $display("[FAIL] Extra output token=0x%02h at out_idx=%0d (accepted=%0d)",
                         token_out, recv_idx, accepted_count);
            end else begin
                expected_token = expected_tokens[recv_idx];
                if (token_out !== expected_token) begin
                    mismatch_count = mismatch_count + 1;
                    $display("[FAIL] Output mismatch expected=0x%02h got=0x%02h at out_idx=%0d",
                             expected_token, token_out, recv_idx);
                end
            end
            recv_idx = recv_idx + 1;
        end
    end
    endtask

    initial begin
        clk = 0; rst = 1; token_valid = 0; token_in = 0;
        tests_passed = 0; tests_total = 0;
        stage_cycles_packed = 0;
        
        @(negedge clk); @(negedge clk); rst = 0;
        repeat(2) @(negedge clk);

        $display("=================================================");
        $display("   Layer Pipeline Controller Tests");
        $display("   Ready/Valid + Skid Backpressure Validation");
        $display("=================================================");

        // TEST 1: Backpressure must assert under sustained valid input.
        tests_total = tests_total + 1;
        rst = 1; @(negedge clk); rst = 0;
        repeat(2) @(negedge clk);
        stage_cycles_packed = {8'd1, 8'd4, 8'd5, 8'd2, 8'd6};
        token_valid = 0; token_in = 0;
        send_idx = 0; accepted_count = 0; ready_low_cycles = 0;
        
        for (i = 0; i < 30; i = i + 1) begin
            @(negedge clk);
            if (send_idx < 4) begin
                token_valid = 1'b1;
                token_in = 8'h10 + send_idx;
            end else begin
                token_valid = 1'b0;
            end

            if (token_valid && token_ready) begin
                send_idx = send_idx + 1;
                accepted_count = accepted_count + 1;
            end

            if (token_valid && !token_ready)
                ready_low_cycles = ready_low_cycles + 1;
        end
        token_valid = 0;
        @(negedge clk);

        if (accepted_count == 4 && ready_low_cycles > 0) begin
            $display("[PASS] Test 1: Backpressure asserted (accepted=%0d, ready_low=%0d)",
                     accepted_count, ready_low_cycles);
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 1: accepted=%0d, ready_low=%0d",
                     accepted_count, ready_low_cycles);
        end

        // TEST 2: Sustained valid stream with strict no-loss/no-reorder checks.
        tests_total = tests_total + 1;
        rst = 1; @(negedge clk); rst = 0;
        repeat(2) @(negedge clk);
        stage_cycles_packed = {8'd1, 8'd4, 8'd5, 8'd2, 8'd6};
        token_valid = 0; token_in = 0;
        scoreboard_reset();
        drain_timeout = 0;

        for (i = 0; i < 4000 && send_idx < 48; i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();

            token_valid = 1'b1;
            token_in = 8'h40 + send_idx;

            if (token_valid && token_ready) begin
                scoreboard_record_accept(token_in);
                send_idx = send_idx + 1;
            end

            if (token_valid && !token_ready)
                ready_low_cycles = ready_low_cycles + 1;
        end
        @(posedge clk);
        token_valid = 0;

        for (i = 0; i < 2000 && recv_idx < accepted_count; i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();
        end
        if (recv_idx < accepted_count)
            drain_timeout = 1;

        for (i = 0; i < 16; i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();
        end

        if (send_idx == 48 && accepted_count == 48 &&
            recv_idx == accepted_count && mismatch_count == 0 &&
            extra_output_count == 0 && !drain_timeout &&
            ready_low_cycles > 0) begin
            $display("[PASS] Test 2: No loss/reorder/duplication under sustained valid (accepted=%0d, output=%0d)",
                     accepted_count, recv_idx);
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 2: accepted=%0d, output=%0d, mismatches=%0d, extras=%0d, ready_low=%0d, timeout=%0d",
                     accepted_count, recv_idx, mismatch_count, extra_output_count, ready_low_cycles, drain_timeout);
        end

        // TEST 3: Bubble-free ordered flow when all stages are 1-cycle.
        tests_total = tests_total + 1;
        rst = 1; @(negedge clk); rst = 0;
        repeat(2) @(negedge clk);
        stage_cycles_packed = {8'd1, 8'd1, 8'd1, 8'd1, 8'd1};
        token_valid = 0; token_in = 0;
        scoreboard_reset();
        drain_timeout = 0;
        outputs_seen = 0; last_out_cycle = -1; max_gap = 0;

        for (i = 0; i < 500 && send_idx < 40; i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();

            if (token_out_valid) begin
                if (last_out_cycle >= 0 && (total_cycles - last_out_cycle) > max_gap)
                    max_gap = total_cycles - last_out_cycle;
                last_out_cycle = total_cycles;
                outputs_seen = outputs_seen + 1;
            end

            token_valid = 1'b1;
            token_in = 8'hA0 + send_idx;

            if (token_valid && token_ready) begin
                scoreboard_record_accept(token_in);
                send_idx = send_idx + 1;
            end
        end
        @(posedge clk);
        token_valid = 0;

        for (i = 0; i < 2000 && recv_idx < accepted_count; i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();
            if (token_out_valid) begin
                if (last_out_cycle >= 0 && (total_cycles - last_out_cycle) > max_gap)
                    max_gap = total_cycles - last_out_cycle;
                last_out_cycle = total_cycles;
                outputs_seen = outputs_seen + 1;
            end
        end
        if (recv_idx < accepted_count)
            drain_timeout = 1;

        for (i = 0; i < 16; i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();
            if (token_out_valid) begin
                if (last_out_cycle >= 0 && (total_cycles - last_out_cycle) > max_gap)
                    max_gap = total_cycles - last_out_cycle;
                last_out_cycle = total_cycles;
                outputs_seen = outputs_seen + 1;
            end
        end

        if (send_idx == 40 && accepted_count == 40 &&
            recv_idx == accepted_count &&
            mismatch_count == 0 && extra_output_count == 0 &&
            !drain_timeout && max_gap <= 1) begin
            $display("[PASS] Test 3: Bubble-free ordered stream (outputs=%0d, max_gap=%0d)",
                     outputs_seen, max_gap);
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 3: accepted=%0d, output=%0d, mismatches=%0d, extras=%0d, max_gap=%0d, timeout=%0d",
                     accepted_count, recv_idx, mismatch_count, extra_output_count, max_gap, drain_timeout);
        end

        // TEST 4: stage_cycles=0 is clamped to deterministic 1-cycle behavior.
        tests_total = tests_total + 1;
        rst = 1; @(negedge clk); rst = 0;
        repeat(2) @(negedge clk);
        stage_cycles_packed = {8'd0, 8'd0, 8'd0, 8'd0, 8'd0};
        token_valid = 0; token_in = 0;
        scoreboard_reset();
        drain_timeout = 0;
        outputs_seen = 0; last_out_cycle = -1; max_gap = 0;

        for (i = 0; i < 600 && send_idx < 32; i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();

            if (token_out_valid) begin
                if (last_out_cycle >= 0 && (total_cycles - last_out_cycle) > max_gap)
                    max_gap = total_cycles - last_out_cycle;
                last_out_cycle = total_cycles;
                outputs_seen = outputs_seen + 1;
            end

            token_valid = 1'b1;
            token_in = 8'h20 + send_idx;

            if (token_valid && token_ready) begin
                scoreboard_record_accept(token_in);
                send_idx = send_idx + 1;
            end
        end
        @(posedge clk);
        token_valid = 0;

        for (i = 0; i < 2000 && recv_idx < accepted_count; i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();
            if (token_out_valid) begin
                if (last_out_cycle >= 0 && (total_cycles - last_out_cycle) > max_gap)
                    max_gap = total_cycles - last_out_cycle;
                last_out_cycle = total_cycles;
                outputs_seen = outputs_seen + 1;
            end
        end
        if (recv_idx < accepted_count)
            drain_timeout = 1;

        for (i = 0; i < 16; i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();
            if (token_out_valid) begin
                if (last_out_cycle >= 0 && (total_cycles - last_out_cycle) > max_gap)
                    max_gap = total_cycles - last_out_cycle;
                last_out_cycle = total_cycles;
                outputs_seen = outputs_seen + 1;
            end
        end

        if (send_idx == 32 && accepted_count == 32 &&
            recv_idx == accepted_count && mismatch_count == 0 &&
            extra_output_count == 0 && !drain_timeout &&
            max_gap <= 1) begin
            $display("[PASS] Test 4: stage_cycles=0 clamp is deterministic (outputs=%0d, max_gap=%0d)",
                     outputs_seen, max_gap);
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 4: accepted=%0d, output=%0d, mismatches=%0d, extras=%0d, max_gap=%0d, timeout=%0d",
                     accepted_count, recv_idx, mismatch_count, extra_output_count, max_gap, drain_timeout);
        end

        // TEST 5: Varied-pattern stress with randomized valid and scoreboard checking.
        tests_total = tests_total + 1;
        rst = 1; @(negedge clk); rst = 0;
        repeat(2) @(negedge clk);
        stage_cycles_packed = {8'd1, 8'd4, 8'd5, 8'd2, 8'd6};
        token_valid = 0; token_in = 0;
        scoreboard_reset();
        drain_timeout = 0;
        seed = 32'h5EEDBEEF;

        for (i = 0; i < 2500 && (send_idx < 96 || recv_idx < accepted_count); i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();

            case ((i / 80) % 4)
                0: stage_cycles_packed = {8'd1, 8'd4, 8'd5, 8'd2, 8'd6};
                1: stage_cycles_packed = {8'd1, 8'd1, 8'd1, 8'd1, 8'd1};
                2: stage_cycles_packed = {8'd0, 8'd2, 8'd1, 8'd3, 8'd1};
                default: stage_cycles_packed = {8'd7, 8'd1, 8'd4, 8'd2, 8'd1};
            endcase

            if (send_idx < 96) begin
                token_valid = (($random(seed) & 32'h3) != 0);
                if ((i % 23) == 0)
                    token_valid = 1'b0;
                token_in = (8'hC0 + send_idx) ^ i[7:0];
            end else begin
                token_valid = 1'b0;
            end

            if (token_valid && token_ready) begin
                scoreboard_record_accept(token_in);
                send_idx = send_idx + 1;
            end
        end
        token_valid = 0;
        if (send_idx < 96)
            drain_timeout = 1;

        for (i = 0; i < 1200 && recv_idx < accepted_count; i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();
        end
        if (recv_idx < accepted_count)
            drain_timeout = 1;

        for (i = 0; i < 20; i = i + 1) begin
            @(negedge clk);
            scoreboard_check_output();
        end

        if (send_idx == 96 && accepted_count == 96 &&
            recv_idx == accepted_count && mismatch_count == 0 &&
            extra_output_count == 0 && !drain_timeout) begin
            $display("[PASS] Test 5: Varied-pattern randomized stress preserved exact IO count/order (accepted=%0d, output=%0d)",
                     accepted_count, recv_idx);
            tests_passed = tests_passed + 1;
        end else begin
            $display("[FAIL] Test 5: accepted=%0d, output=%0d, mismatches=%0d, extras=%0d, timeout=%0d",
                     accepted_count, recv_idx, mismatch_count, extra_output_count, drain_timeout);
        end

        $display("=================================================");
        $display("   Pipeline Controller Tests: %0d / %0d PASSED", tests_passed, tests_total);
        $display("=================================================");
        #20 $finish;
    end

endmodule
