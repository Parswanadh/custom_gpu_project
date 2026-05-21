`timescale 1ns / 1ps

// ============================================================================
// Module: mem_controller_mixed
// Description: Memory Controller with Mixed-Precision Fetch Logic.
//
// Supports fetching data at different precisions:
// - 4-bit: 8 elements per 32-bit word
// - 8-bit: 4 elements per 32-bit word
// - 16-bit: 2 elements per 32-bit word
//
// The precision signal defines how the 32-bit word is sliced:
// 00: 4-bit precision
// 01: 8-bit precision
// 10: 16-bit precision
//
// req_addr is the element index. The controller calculates the word address
// and the bit offset within that word.
// ============================================================================

module mem_controller_mixed #(
    parameter ADDR_WIDTH = 12,
    parameter DATA_WIDTH = 32,  // Internal word width
    parameter DEPTH      = 4096
)(
    input  wire                   clk,
    input  wire                   rst,
    
    // Request interface
    input  wire                   req_valid,
    input  wire [ADDR_WIDTH-1:0]  req_addr,      // Element index
    input  wire [1:0]             precision,     // 00=4b, 01=8b, 10=16b
    output reg                    req_ready,
    
    // Response interface
    output reg  [15:0]            resp_data,     // Max precision is 16-bit
    output reg                    resp_valid
);

    // Internal Memory
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    
    // Address and offset calculation
    reg [ADDR_WIDTH-1:0] word_addr;
    reg [4:0]            bit_offset;
    
    always @(*) begin
        case (precision)
            2'b00: begin // 4-bit
                word_addr  = req_addr >> 3; // 8 elements per word
                bit_offset = (req_addr[2:0]) << 2;
            end
            2'b01: begin // 8-bit
                word_addr  = req_addr >> 2; // 4 elements per word
                bit_offset = (req_addr[1:0]) << 3;
            end
            2'b10: begin // 16-bit
                word_addr  = req_addr >> 1; // 2 elements per word
                bit_offset = (req_addr[0]) << 4;
            end
            default: begin
                word_addr  = 0;
                bit_offset = 0;
            end
        endcase
    end

    // FSM
    localparam IDLE  = 1'b0;
    localparam FETCH = 1'b1;
    reg state;

    always @(posedge clk) begin
        if (rst) begin
            state      <= IDLE;
            req_ready  <= 1'b1;
            resp_valid <= 1'b0;
            resp_data  <= 0;
        end else begin
            case (state)
                IDLE: begin
                    resp_valid <= 1'b0;
                    if (req_valid && req_ready) begin
                        req_ready <= 1'b0;
                        state     <= FETCH;
                        // $display("FSM: IDLE -> FETCH (addr=%d)", req_addr);
                    end
                end
                
                FETCH: begin
                    case (precision)
                        2'b00: resp_data <= {12'b0, mem[word_addr][bit_offset +: 4]};
                        2'b01: resp_data <= {8'b0,  mem[word_addr][bit_offset +: 8]};
                        2'b10: resp_data <= mem[word_addr][bit_offset +: 16];
                        default: resp_data <= 0;
                    endcase
                    resp_valid <= 1'b1;
                    req_ready  <= 1'b1;
                    state      <= IDLE;
                    // $display("FSM: FETCH -> IDLE (data=%x)", resp_data);
                end
            endcase
        end
    end

    // Task for preloading memory (simulation only)
    task preload_mem(input [ADDR_WIDTH-1:0] addr, input [DATA_WIDTH-1:0] data);
        begin
            mem[addr] = data;
        end
    endtask

endmodule
