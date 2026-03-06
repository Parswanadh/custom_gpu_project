// ============================================================================
// Module: weight_double_buffer
// Description: Double-buffered weight loader for zero-stall layer transitions.
//   Maintains two weight banks that ping-pong: while one bank feeds computation,
//   the other is loaded with the next layer's weights.
//
//   This hides memory latency between layers entirely.
//
//   Architecture:
//     Bank A ←→ Compute path (active bank feeds multipliers)
//     Bank B ←→ Load path   (next layer's weights being loaded)
//     After compute completes, banks swap roles.
//
// Parameters: NUM_WEIGHTS, DATA_WIDTH
// ============================================================================
module weight_double_buffer #(
    parameter NUM_WEIGHTS = 64,    // Max weights per layer
    parameter DATA_WIDTH  = 16,
    parameter ADDR_WIDTH  = $clog2(NUM_WEIGHTS)
)(
    input  wire                        clk,
    input  wire                        rst,
    
    // Load interface (writes to inactive bank)
    input  wire                        load_en,
    input  wire [ADDR_WIDTH-1:0]       load_addr,
    input  wire signed [DATA_WIDTH-1:0] load_data,
    
    // Read interface (reads from active bank)
    input  wire                        read_en,
    input  wire [ADDR_WIDTH-1:0]       read_addr,
    output reg  signed [DATA_WIDTH-1:0] read_data,
    output reg                         read_valid,
    
    // Bank swap control
    input  wire                        swap_banks,    // Pulse to swap active/inactive
    output reg                         active_bank,   // 0 = Bank A active, 1 = Bank B active
    
    // Status
    output reg  [31:0]                 loads_completed,
    output reg  [31:0]                 swaps_completed
);

    // Two weight banks
    reg signed [DATA_WIDTH-1:0] bank_a [0:NUM_WEIGHTS-1];
    reg signed [DATA_WIDTH-1:0] bank_b [0:NUM_WEIGHTS-1];

    integer i;

    always @(posedge clk) begin
        if (rst) begin
            active_bank      <= 1'b0;
            read_data        <= 0;
            read_valid       <= 1'b0;
            loads_completed  <= 32'd0;
            swaps_completed  <= 32'd0;
            for (i = 0; i < NUM_WEIGHTS; i = i + 1) begin
                bank_a[i] <= 0;
                bank_b[i] <= 0;
            end
        end else begin
            read_valid <= 1'b0;
            
            // Bank swap on command
            if (swap_banks) begin
                active_bank     <= ~active_bank;
                swaps_completed <= swaps_completed + 1;
            end
            
            // Write to INACTIVE bank
            if (load_en) begin
                if (active_bank == 1'b0)
                    bank_b[load_addr] <= load_data;  // Bank A active → write Bank B
                else
                    bank_a[load_addr] <= load_data;  // Bank B active → write Bank A
                loads_completed <= loads_completed + 1;
            end
            
            // Read from ACTIVE bank
            if (read_en) begin
                if (active_bank == 1'b0)
                    read_data <= bank_a[read_addr];  // Bank A active → read Bank A
                else
                    read_data <= bank_b[read_addr];  // Bank B active → read Bank B
                read_valid <= 1'b1;
            end
        end
    end

endmodule
