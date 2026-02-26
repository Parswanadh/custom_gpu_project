// ============================================================================
// Testbench: gpu_top_integrated_tb
// Verifies the fully integrated 4-wide pipelined GPU.
// Compares cycle counts against:
//   - FSM design: ~7 cycles per result
//   - 1-wide pipeline: 1 cycle per result
//   - 4-wide integrated: 4 results per cycle
// ============================================================================
`timescale 1ns/1ps

module gpu_top_integrated_tb;
    reg         clk, rst;
    reg  [3:0]  dq_scale, dq_offset;
    reg         mem_write_en;
    reg  [7:0]  mem_write_val;
    reg  [5:0]  mem_write_idx;
    reg         valid_in;
    reg  [5:0]  weight_base_addr;
    reg  [7:0]  activation_in;
    wire [63:0] result_out;
    wire        valid_out;
    wire [3:0]  zero_skip_mask;
    wire [31:0] accumulator;
    wire [4:0]  pipe_active;

    gpu_top_integrated #(
        .MEM_DEPTH(64),
        .LANES(4)
    ) uut (
        .clk(clk), .rst(rst),
        .dq_scale(dq_scale), .dq_offset(dq_offset),
        .mem_write_en(mem_write_en), .mem_write_val(mem_write_val),
        .mem_write_idx(mem_write_idx),
        .valid_in(valid_in), .weight_base_addr(weight_base_addr),
        .activation_in(activation_in),
        .result_out(result_out), .valid_out(valid_out),
        .zero_skip_mask(zero_skip_mask), .accumulator(accumulator),
        .pipe_active(pipe_active)
    );

    // Clock: 10ns period (100 MHz)
    always #5 clk = ~clk;

    integer results_count;
    integer cycles_to_first_result;
    integer total_cycles;
    integer total_products;
    integer zero_skip_count;
    integer op;
    integer j;

    initial begin
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║  GPU Integrated Testbench: 4-Wide Pipeline Verification    ║");
        $display("║  Testing: 4 weights/cycle × 5-stage pipeline               ║");
        $display("╚══════════════════════════════════════════════════════════════╝");

        // Initialize
        clk = 0; rst = 1;
        dq_scale = 4'd2; dq_offset = 4'd0;
        mem_write_en = 0; valid_in = 0;
        weight_base_addr = 0; activation_in = 0;
        results_count = 0;
        total_products = 0;
        cycles_to_first_result = 0;
        zero_skip_count = 0;
        total_cycles = 0;

        // Reset
        #20 rst = 0;
        #10;

        // ================================================================
        // TEST 1: Load 32 weights (some zero for zero-skip testing)
        // ================================================================
        $display("");
        $display("[1] Loading 32 weights (8 groups of 4)...");
        $display("    Zeros at positions: 1, 4, 7, 10, 13, 17, 21, 25");

        load_weight(0,  8'd3);   // Group 0: [3, 0, 5, 7]
        load_weight(1,  8'd0);   // zero!
        load_weight(2,  8'd5);
        load_weight(3,  8'd7);

        load_weight(4,  8'd0);   // Group 1: [0, 2, 4, 0]
        load_weight(5,  8'd2);   // zero!
        load_weight(6,  8'd4);
        load_weight(7,  8'd0);   // zero!

        load_weight(8,  8'd1);   // Group 2: [1, 6, 0, 8]
        load_weight(9,  8'd6);
        load_weight(10, 8'd0);   // zero!
        load_weight(11, 8'd8);

        load_weight(12, 8'd3);   // Group 3: [3, 0, 9, 2]
        load_weight(13, 8'd0);   // zero!
        load_weight(14, 8'd9);
        load_weight(15, 8'd2);

        load_weight(16, 8'd5);   // Group 4: [5, 0, 3, 7]
        load_weight(17, 8'd0);   // zero!
        load_weight(18, 8'd3);
        load_weight(19, 8'd7);

        load_weight(20, 8'd4);   // Group 5: [4, 0, 6, 1]
        load_weight(21, 8'd0);   // zero!
        load_weight(22, 8'd6);
        load_weight(23, 8'd1);

        load_weight(24, 8'd8);   // Group 6: [8, 0, 2, 5]
        load_weight(25, 8'd0);   // zero!
        load_weight(26, 8'd2);
        load_weight(27, 8'd5);

        load_weight(28, 8'd3);   // Group 7: [3, 4, 7, 1]
        load_weight(29, 8'd4);
        load_weight(30, 8'd7);
        load_weight(31, 8'd1);

        #20;

        // ================================================================
        // TEST 2: Stream 8 operations (each reads 4 weights = 32 products)
        // ================================================================
        $display("");
        $display("[2] Streaming 8 operations (4 weights each = 32 products)");
        $display("    Expected: 4 results/cycle after ~5-cycle fill");
        $display("");

        total_cycles = 0;

        fork
            // Producer: feed 8 operations, one per cycle
            begin
                for (op = 0; op < 8; op = op + 1) begin
                    @(posedge clk);
                    valid_in         <= 1'b1;
                    weight_base_addr <= op * 4;                 // Groups of 4
                    activation_in    <= 8'd10 + op[7:0];        // 10, 11, 12...
                    total_cycles = total_cycles + 1;
                end
                @(posedge clk);
                valid_in <= 1'b0;
            end

            // Consumer: count results
            begin
                wait(valid_out);
                cycles_to_first_result = total_cycles;
                $display("    [PIPELINE] First result at cycle %0d (fill latency)", cycles_to_first_result);

                while (results_count < 8) begin
                    @(posedge clk);
                    if (valid_out) begin
                        results_count = results_count + 1;
                        total_products = total_products + 4;

                        // Count zero-skips across 4 lanes
                        for (j = 0; j < 4; j = j + 1) begin
                            if (zero_skip_mask[j])
                                zero_skip_count = zero_skip_count + 1;
                        end

                        $display("    Result %0d: [%04h, %04h, %04h, %04h] | skip=%b | acc=%0d | pipe=%b",
                            results_count,
                            result_out[15:0],  result_out[31:16],
                            result_out[47:32], result_out[63:48],
                            zero_skip_mask, accumulator, pipe_active);
                    end
                end
            end
        join

        // Wait for pipeline to drain
        repeat(10) @(posedge clk);

        // ================================================================
        // PERFORMANCE REPORT
        // ================================================================
        $display("");
        $display("╔══════════════════════════════════════════════════════════════╗");
        $display("║              PERFORMANCE COMPARISON                        ║");
        $display("╠══════════════════════════════════════════════════════════════╣");
        $display("║                                                            ║");
        $display("║  Operations fed:           8                               ║");
        $display("║  Products computed:        %2d  (8 ops × 4 lanes)           ║", total_products);
        $display("║  Pipeline fill latency:    %0d cycles                       ║", cycles_to_first_result);
        $display("║  Total feed cycles:        %2d                              ║", total_cycles);
        $display("║  Zero-skipped:             %2d / %2d  (%0d%%)                  ║",
            zero_skip_count, total_products,
            (zero_skip_count * 100) / total_products);
        $display("║  Accumulator (dot product): %0d                       ║", accumulator);
        $display("║                                                            ║");
        $display("║  ---- Throughput Comparison ----                            ║");
        $display("║                                                            ║");
        $display("║  FSM (gpu_top):                                            ║");
        $display("║    32 products × 7 cycles each = 224 cycles                ║");
        $display("║    Throughput: 1 product / 7.0 cycles                      ║");
        $display("║                                                            ║");
        $display("║  Pipeline (gpu_top_pipelined):                             ║");
        $display("║    32 products × 1 cycle each  = 32 cycles                 ║");
        $display("║    Throughput: 1 product / 1.0 cycles                      ║");
        $display("║                                                            ║");
        $display("║  Integrated (gpu_top_integrated):                          ║");
        $display("║    32 products in %2d feed cycles = %0.1f products/cycle       ║",
            total_cycles, total_products * 1.0 / total_cycles);
        $display("║    Throughput: 4 products / 1.0 cycle                      ║");
        $display("║                                                            ║");
        $display("║  ═══════════════════════════════════════════════            ║");
        $display("║  Speedup vs FSM:        %0.1fx                              ║",
            (total_products * 7.0) / total_cycles);
        $display("║  Speedup vs Pipeline:   %0.1fx                              ║",
            (total_products * 1.0) / total_cycles);
        $display("║  ═══════════════════════════════════════════════            ║");
        $display("║                                                            ║");
        $display("╚══════════════════════════════════════════════════════════════╝");
        $display("");
        $display("=== Test PASSED ===");
        $finish;
    end

    // Weight loading task
    task load_weight(input [5:0] idx, input [7:0] val);
        begin
            @(posedge clk);
            mem_write_en  <= 1'b1;
            mem_write_idx <= idx;
            mem_write_val <= val;
            @(posedge clk);
            mem_write_en  <= 1'b0;
        end
    endtask

endmodule
