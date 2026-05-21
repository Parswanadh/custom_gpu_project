`timescale 1ns/1ps

module rmsnorm_vp_tb;

    reg clk;
    reg rst_n;
    reg start;
    reg [1:0] precision_ctrl;
    reg [15:0] data_in;
    reg data_in_valid;
    reg end_vector;
    
    wire [15:0] data_out;
    wire data_out_valid;
    wire ready;
    
    // Instantiate DUT
    rmsnorm_vp #(
        .MAX_VEC_LEN(256)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .precision_ctrl(precision_ctrl),
        .data_in(data_in),
        .data_in_valid(data_in_valid),
        .end_vector(end_vector),
        .data_out(data_out),
        .data_out_valid(data_out_valid),
        .ready(ready)
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    reg [15:0] in_mem [0:255];
    reg [15:0] out_mem [0:255];
    
    integer i, errors;
    
    initial begin
        // Initialize memory to prevent X propagation
        for (i = 0; i < 256; i = i + 1) begin
            in_mem[i] = 16'd0;
            out_mem[i] = 16'd0;
        end
        
        // Use $readmemh to load input and output vector data
        $readmemh("scripts/rmsnorm_input.hex", in_mem);
        $readmemh("scripts/rmsnorm_output.hex", out_mem);
        
        rst_n = 0;
        start = 0;
        precision_ctrl = 2'b10; // Use 24-bit precision for this test
        data_in = 0;
        data_in_valid = 0;
        end_vector = 0;
        errors = 0;
        
        #20;
        rst_n = 1;
        #20;
        
        // Wait for ready signal
        wait(ready);
        @(posedge clk);
        
        // Feed vector to DUT (Pass 1)
        for (i = 0; i < 16; i = i + 1) begin
            if (i == 0) start = 1; else start = 0;
            if (i == 15) end_vector = 1; else end_vector = 0;
            data_in = in_mem[i];
            data_in_valid = 1;
            @(posedge clk);
        end
        data_in_valid = 0;
        end_vector = 0;
        
        // Wait for output valid and verify (Pass 2)
        for (i = 0; i < 16; i = i + 1) begin
            wait(data_out_valid);
            // Verify output matches the expected hex output
            if (data_out !== out_mem[i] && out_mem[i] !== 16'hx) begin
                $display("Mismatch at index %0d: Expected %x, Got %x", i, out_mem[i], data_out);
                errors = errors + 1;
            end else begin
                $display("Match at index %0d: Expected %x, Got %x", i, out_mem[i], data_out);
            end
            @(posedge clk);
        end
        
        if (errors == 0) begin
            $display("TEST PASSED!");
        end else begin
            $display("TEST FAILED with %0d errors.", errors);
        end
        
        $finish;
    end

endmodule