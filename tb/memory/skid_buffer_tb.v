`timescale 1ns/1ps

module skid_buffer_tb;
    parameter DATA_WIDTH = 128;
    parameter CLK_PERIOD = 10;

    reg clk;
    reg rst;
    reg valid_in;
    reg [DATA_WIDTH-1:0] data_in;
    reg ready_from_downstream;

    wire ready_for_upstream;
    wire valid_out;
    wire [DATA_WIDTH-1:0] data_out;

    reg [DATA_WIDTH-1:0] data_check_val;

    // Instantiate DUT
    skid_buffer #(DATA_WIDTH) dut (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .data_in(data_in),
        .ready_from_downstream(ready_from_downstream),
        .ready_for_upstream(ready_for_upstream),
        .valid_out(valid_out),
        .data_out(data_out)
    );

    // Clock generation
    always #(CLK_PERIOD/2) clk = ~clk;

    // Test sequence
    integer i;
    integer success_count = 0;
    integer total_tests = 100;
    reg [DATA_WIDTH-1:0] expected_data[$];

    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        valid_in = 0;
        data_in = 0;
        ready_from_downstream = 0;

        #(CLK_PERIOD * 5);
        rst = 0;
        #(CLK_PERIOD * 2);

        $display("Starting Skid Buffer Randomized Testbench (SystemVerilog Mode)...");

        fork
            // Driver: Robust Handshake
            begin
                for (i = 0; i < total_tests; i = i + 1) begin
                    // Random delay before putting data
                    if ($urandom % 5 == 0) begin
                        valid_in <= 0;
                        repeat ($urandom % 3) @(posedge clk);
                    end
                    
                    data_in <= i + 1000;
                    valid_in <= 1;
                    expected_data.push_back(i + 1000);
                    
                    // Wait for it to be accepted
                    @(posedge clk);
                    while (!ready_for_upstream) @(posedge clk);
                    valid_in <= 0;
                end
            end

            // Monitor/Backpressure: Randomly pull data
            begin
                while (success_count < total_tests) begin
                    ready_from_downstream <= ($urandom % 2); // Randomly assert/deassert ready
                    @(posedge clk);
                    if (valid_out && ready_from_downstream) begin
                        data_check_val = expected_data.pop_front();
                        if (data_out === data_check_val) begin
                            success_count = success_count + 1;
                            if (success_count % 10 == 0) $display("Progress: %d/%d", success_count, total_tests);
                        end else begin
                            $display("ERROR: Data mismatch! Expected %h, Got %h", data_check_val, data_out);
                            $finish;
                        end
                    end
                end
            end
        join

        $display("PASS: Verified %d transfers with randomized backpressure.", success_count);
        $finish;
    end

    // Timeout
    initial begin
        #(CLK_PERIOD * total_tests * 20);
        $display("FAIL: Testbench timed out at success_count = %d", success_count);
        $finish;
    end

endmodule
