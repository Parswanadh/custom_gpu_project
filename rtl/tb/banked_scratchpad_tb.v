`timescale 1ns/1ps

// ============================================================================
// Testbench: banked_scratchpad_tb
// Description: Testbench for banked_scratchpad module
// ============================================================================

module banked_scratchpad_tb;

    // Parameters
    localparam TOTAL_DEPTH = 4096;
    localparam NUM_BANKS = 8;
    localparam DATA_W = 16;
    localparam BANK_DEPTH = TOTAL_DEPTH / NUM_BANKS;
    localparam ADDR_W = $clog2(BANK_DEPTH);

    // Inputs
    reg clk;
    reg rst;
    
    // Port A: Compute pipeline
    reg a_read_en;
    reg [ADDR_W+$clog2(NUM_BANKS)-1:0] a_read_addr;
    wire [DATA_W-1:0] a_read_data;
    wire a_read_valid;
    reg a_write_en;
    reg [ADDR_W+$clog2(NUM_BANKS)-1:0] a_write_addr;
    reg [DATA_W-1:0] a_write_data;
    
    // Port B: DMA / Command processor
    reg b_read_en;
    reg [ADDR_W+$clog2(NUM_BANKS)-1:0] b_read_addr;
    wire [DATA_W-1:0] b_read_data;
    wire b_read_valid;
    reg b_write_en;
    reg [ADDR_W+$clog2(NUM_BANKS)-1:0] b_write_addr;
    reg [DATA_W-1:0] b_write_data;
    
    // Outputs
    wire [NUM_BANKS-1:0] bank_conflicts;
    wire [ADDR_W:0] usage_count;

    // Instantiate the Unit Under Test (UUT)
    banked_scratchpad #(
        .TOTAL_DEPTH(TOTAL_DEPTH),
        .NUM_BANKS(NUM_BANKS),
        .DATA_W(DATA_W),
        .BANK_DEPTH(BANK_DEPTH),
        .ADDR_W(ADDR_W)
    ) uut (
        .clk(clk),
        .rst(rst),
        // Port A: Compute pipeline
        .a_read_en(a_read_en),
        .a_read_addr(a_read_addr),
        .a_read_data(a_read_data),
        .a_read_valid(a_read_valid),
        .a_write_en(a_write_en),
        .a_write_addr(a_write_addr),
        .a_write_data(a_write_data),
        // Port B: DMA / Command processor
        .b_read_en(b_read_en),
        .b_read_addr(b_read_addr),
        .b_read_data(b_read_data),
        .b_read_valid(b_read_valid),
        .b_write_en(b_write_en),
        .b_write_addr(b_write_addr),
        .b_write_data(b_write_data),
        // Status and conflict detection
        .bank_conflicts(bank_conflicts),
        .usage_count(usage_count)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end

    // Test sequence
    initial begin
        // Initialize inputs
        rst = 1;
        a_read_en = 0;
        a_read_addr = 0;
        a_write_en = 0;
        a_write_addr = 0;
        a_write_data = 0;
        b_read_en = 0;
        b_read_addr = 0;
        b_write_en = 0;
        b_write_addr = 0;
        b_write_data = 0;
        
        // Apply reset
        #20 rst = 0;
        
        // Test 1: Basic write and read from same port
        #10;
        a_write_en = 1;
        a_write_addr = 0; // Bank 0, offset 0
        a_write_data = 16'h1234;
        #10;
        a_write_en = 0;
        
        #10;
        a_read_en = 1;
        a_read_addr = 0; // Bank 0, offset 0
        #10;
        a_read_en = 0;
        
        // Test 2: Basic write and read from different ports (no conflict)
        #10;
        b_write_en = 1;
        b_write_addr = 16; // Bank 0, offset 16 (different address)
        b_write_data = 16'hABCD;
        #10;
        b_write_en = 0;
        
        #10;
        b_read_en = 1;
        b_read_addr = 16; // Bank 0, offset 16
        #10;
        b_read_en = 0;
        
        // Test 3: Write conflict test (same address, both ports)
        #10;
        a_write_en = 1;
        a_write_addr = 32; // Bank 0, offset 32
        a_write_data = 16'h1111;
        b_write_en = 1;
        b_write_addr = 32; // Bank 0, offset 32 (same address!)
        b_write_data = 16'h2222;
        #10;
        a_write_en = 0;
        b_write_en = 0;
        
        // Test 4: Read after write conflict test
        #10;
        a_read_en = 1;
        a_read_addr = 32; // Bank 0, offset 32
        b_read_en = 1;
        b_read_addr = 32; // Bank 0, offset 32
        #10;
        a_read_en = 0;
        b_read_en = 0;
        
        // Test 5: Cross-bank access (no conflict)
        #10;
        a_write_en = 1;
        a_write_addr = 0; // Bank 0, offset 0
        a_write_data = 16'h3333;
        b_write_en = 1;
        b_write_addr = BANK_DEPTH; // Bank 1, offset 0
        b_write_data = 16'h4444;
        #10;
        a_write_en = 0;
        b_write_en = 0;
        
        #10;
        a_read_en = 1;
        a_read_addr = 0; // Bank 0, offset 0
        b_read_en = 1;
        b_read_addr = BANK_DEPTH; // Bank 1, offset 0
        #10;
        a_read_en = 0;
        b_read_en = 0;
        
        // Finish simulation
        #20 $finish;
    end
    
    // Monitor signals
    initial begin
        $display("Time\tClk\tRst\tA_Write\tA_Addr\tA_Data\tB_Write\tB_Addr\tB_Data\tA_Valid\tB_Valid\tA_Data\tB_Data\tConflicts");
        $monitor("%0t\t%b\t%b\t%b\t%h\t%h\t%b\t%h\t%h\t%b\t%b\t%h\t%h\t%b", 
                 $time, clk, rst, 
                 a_write_en, a_write_addr, a_write_data,
                 b_write_en, b_write_addr, b_write_data,
                 a_read_valid, b_read_valid, a_read_data, b_read_data,
                 bank_conflicts);
    end

endmodule