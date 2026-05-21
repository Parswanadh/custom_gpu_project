`timescale 1ns/1ps

module rope_unit_v2_tb;

    // Parameters
    localparam DATA_WIDTH = 16;
    localparam LUT_ADDR_WIDTH = 10;
    localparam LUT_DATA_WIDTH = 32;
    localparam LUT_DEPTH = 1024;
    localparam FRAC_BITS = 14;

    // Clock and Reset
    reg clk;
    reg rst_n;
    
    // Inputs
    reg enable;
    reg signed [DATA_WIDTH-1:0] q_in_0;
    reg signed [DATA_WIDTH-1:0] q_in_1;
    reg signed [DATA_WIDTH-1:0] k_in_0;
    reg signed [DATA_WIDTH-1:0] k_in_1;
    reg [LUT_ADDR_WIDTH-1:0]    sin_addr;
    reg [LUT_ADDR_WIDTH-1:0]    cos_addr;
    
    reg lut_we;
    reg [LUT_ADDR_WIDTH-1:0] lut_waddr;
    reg [LUT_DATA_WIDTH-1:0] lut_din;
    
    // Outputs
    wire signed [DATA_WIDTH-1:0] q_out_0;
    wire signed [DATA_WIDTH-1:0] q_out_1;
    wire signed [DATA_WIDTH-1:0] k_out_0;
    wire signed [DATA_WIDTH-1:0] k_out_1;
    wire valid_out;

    // Instantiate the Unit Under Test (UUT)
    rope_unit_v2 #(
        .DATA_WIDTH(DATA_WIDTH),
        .LUT_ADDR_WIDTH(LUT_ADDR_WIDTH),
        .LUT_DATA_WIDTH(LUT_DATA_WIDTH),
        .LUT_DEPTH(LUT_DEPTH),
        .FRAC_BITS(FRAC_BITS)
    ) uut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .q_in_0(q_in_0),
        .q_in_1(q_in_1),
        .k_in_0(k_in_0),
        .k_in_1(k_in_1),
        .sin_addr(sin_addr),
        .cos_addr(cos_addr),
        .lut_we(lut_we),
        .lut_waddr(lut_waddr),
        .lut_din(lut_din),
        .q_out_0(q_out_0),
        .q_out_1(q_out_1),
        .k_out_0(k_out_0),
        .k_out_1(k_out_1),
        .valid_out(valid_out)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 100MHz clock
    end

    // Memory arrays for hex files
    reg [31:0] rope_input_qk [0:31];
    reg [15:0] rope_output_q [0:31];
    reg [15:0] rope_output_k [0:31];
    
    integer i, pos, dim_idx;
    integer errors = 0;
    
    // Test Queues
    reg signed [DATA_WIDTH-1:0] q0_queue [0:31];
    reg signed [DATA_WIDTH-1:0] q1_queue [0:31];
    reg signed [DATA_WIDTH-1:0] k0_queue [0:31];
    reg signed [DATA_WIDTH-1:0] k1_queue [0:31];
    integer queue_head = 0;
    integer queue_tail = 0;
    
    real theta_base = 10000.0;
    real power, theta_i, cos_th, sin_th;
    integer sin_val_int, cos_val_int;

    integer tolerance = 2;
    
    function integer abs_diff;
        input integer a;
        input integer b;
        begin
            if (a > b) abs_diff = a - b;
            else abs_diff = b - a;
        end
    endfunction

    initial begin
        // Initialize Inputs
        rst_n = 0;
        enable = 0;
        q_in_0 = 0;
        q_in_1 = 0;
        k_in_0 = 0;
        k_in_1 = 0;
        sin_addr = 0;
        cos_addr = 0;
        lut_we = 0;
        lut_waddr = 0;
        lut_din = 0;

        // Load Files (using relative path that typically matches project root or sim dir)
        // If run from custom_gpu_project/tb/transformer, it should be ../../../scripts/
        // If run from workspace root, it should be custom_gpu_project/scripts/
        // We'll try to rely on typical relative directory
        $readmemh("scripts/rope_input_qk.hex", rope_input_qk);
        $readmemh("scripts/rope_output_q.hex", rope_output_q);
        $readmemh("scripts/rope_output_k.hex", rope_output_k);
        
        #100;
        rst_n = 1;
        #10;
        
        // Populate LUT
        for (pos = 0; pos < 4; pos = pos + 1) begin
            for (dim_idx = 0; dim_idx < 8; dim_idx = dim_idx + 2) begin
                power = dim_idx / 8.0;
                theta_i = pos / (theta_base ** power);
                cos_th = $cos(theta_i);
                sin_th = $sin(theta_i);
                
                sin_val_int = $rtoi(sin_th * 16384.0);
                cos_val_int = $rtoi(cos_th * 16384.0);
                
                // Write SIN
                @(posedge clk);
                lut_we <= 1;
                lut_waddr <= pos * 4 + (dim_idx/2);
                lut_din <= sin_val_int;
                
                // Write COS
                @(posedge clk);
                lut_we <= 1;
                lut_waddr <= pos * 4 + (dim_idx/2) + 64;
                lut_din <= cos_val_int;
            end
        end
        @(posedge clk);
        lut_we <= 0;
        #20;
        
        // Run test vectors
        for (i = 0; i < 32; i = i + 2) begin
            @(posedge clk);
            enable <= 1;
            q_in_0 <= rope_input_qk[i][31:16];
            k_in_0 <= rope_input_qk[i][15:0];
            q_in_1 <= rope_input_qk[i+1][31:16];
            k_in_1 <= rope_input_qk[i+1][15:0];
            
            pos = i / 8;
            dim_idx = (i % 8);
            
            sin_addr <= pos * 4 + (dim_idx/2);
            cos_addr <= pos * 4 + (dim_idx/2) + 64;
            
            q0_queue[queue_tail] <= rope_output_q[i];
            q1_queue[queue_tail] <= rope_output_q[i+1];
            k0_queue[queue_tail] <= rope_output_k[i];
            k1_queue[queue_tail] <= rope_output_k[i+1];
            queue_tail = queue_tail + 1;
        end
        
        @(posedge clk);
        enable <= 0;
        
        #100;
        if (errors == 0)
            $display("SUCCESS: All tests passed!");
        else
            $display("FAILED: %0d errors found.", errors);
            
        $finish;
    end

    // Monitor output
    always @(posedge clk) begin
        if (valid_out) begin
            if (abs_diff(q_out_0, q0_queue[queue_head]) > tolerance || 
                abs_diff(q_out_1, q1_queue[queue_head]) > tolerance ||
                abs_diff(k_out_0, k0_queue[queue_head]) > tolerance || 
                abs_diff(k_out_1, k1_queue[queue_head]) > tolerance) begin
                $display("ERROR at index %0d: Expected Q={%h, %h}, K={%h, %h}, Got Q={%h, %h}, K={%h, %h}",
                         queue_head, q0_queue[queue_head], q1_queue[queue_head], 
                         k0_queue[queue_head], k1_queue[queue_head],
                         q_out_0, q_out_1, k_out_0, k_out_1);
                errors = errors + 1;
            end else begin
                $display("PASS at index %0d: Q={%h, %h}, K={%h, %h}", 
                         queue_head, q_out_0, q_out_1, k_out_0, k_out_1);
            end
            queue_head = queue_head + 1;
        end
    end

endmodule
