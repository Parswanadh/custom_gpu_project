`timescale 1ns/1ps

module rmsnorm_vp #(
    parameter MAX_VEC_LEN = 256
)(
    input  wire        clk,
    input  wire        rst_n,
    
    input  wire        start,
    input  wire [1:0]  precision_ctrl, // 00: 12-bit, 01: 16-bit, 10: 24-bit
    input  wire [15:0] data_in,
    input  wire        data_in_valid,
    input  wire        end_vector,
    
    output reg  [15:0] data_out,
    output reg         data_out_valid,
    output reg         ready
);

    localparam STATE_IDLE  = 2'd0;
    localparam STATE_ACCUM = 2'd1;
    localparam STATE_CALC  = 2'd2;
    localparam STATE_SCALE = 2'd3;

    reg [1:0] state;
    
    // Internal Buffer to store vector for Pass 2
    reg [15:0] buffer [0:MAX_VEC_LEN-1];
    reg [7:0]  count;
    reg [7:0]  vec_len;
    
    // Accumulator for Sum of Squares
    reg [23:0] accum;
    
    // LUT Connections
    wire [7:0]  lut_addr;
    wire [15:0] lut_data_out;
    
    inv_sqrt_lut_256 u_lut (
        .clk (clk),
        .addr(lut_addr),
        .dout(lut_data_out)
    );
    
    // Square calculation (assume signed inputs)
    wire signed [15:0] signed_data_in = data_in;
    wire signed [31:0] square = signed_data_in * signed_data_in;
    
    // Dynamically scale precision of the squared value before accumulation
    reg [23:0] masked_square;
    always @(*) begin
        case (precision_ctrl)
            2'b00: masked_square = {12'b0, square[23:12]}; // 12-bit accumulation
            2'b01: masked_square = {8'b0,  square[23:8]};  // 16-bit accumulation
            2'b10: masked_square = square[23:0];           // 24-bit accumulation
            default: masked_square = square[23:0];
        endcase
    end
    
    // Derive address for LUT based on accumulated value
    // In a real implementation, this would involve dividing by vec_len first
    // For this prototype, we use the upper 8 bits of the accumulator
    assign lut_addr = accum[23:16];
    
    reg [15:0] inv_sqrt_reg;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= STATE_IDLE;
            accum <= 24'd0;
            count <= 8'd0;
            vec_len <= 8'd0;
            data_out_valid <= 1'b0;
            data_out <= 16'd0;
            ready <= 1'b1;
            inv_sqrt_reg <= 16'd0;
        end else begin
            data_out_valid <= 1'b0;
            
            case (state)
                STATE_IDLE: begin
                    ready <= 1'b1;
                    if (start && data_in_valid) begin
                        ready <= 1'b0;
                        accum <= masked_square;
                        buffer[0] <= data_in;
                        count <= 8'd1;
                        state <= STATE_ACCUM;
                    end
                end
                
                STATE_ACCUM: begin
                    if (data_in_valid) begin
                        accum <= accum + masked_square;
                        buffer[count] <= data_in;
                        count <= count + 8'd1;
                        if (end_vector) begin
                            vec_len <= count + 8'd1;
                            state <= STATE_CALC;
                        end
                    end
                end
                
                STATE_CALC: begin
                    // One cycle delay for synchronous LUT read
                    inv_sqrt_reg <= lut_data_out;
                    count <= 8'd0;
                    state <= STATE_SCALE;
                end
                
                STATE_SCALE: begin
                    if (count < vec_len) begin
                        // Pass 2: Multiply by inverse square root and shift
                        data_out <= ($signed(buffer[count]) * $signed({1'b0, inv_sqrt_reg[14:0]})) >>> 8;
                        data_out_valid <= 1'b1;
                        count <= count + 8'd1;
                    end else begin
                        state <= STATE_IDLE;
                        ready <= 1'b1;
                    end
                end
            endcase
        end
    end

endmodule