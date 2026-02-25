// ============================================================================
// Module: sparse_memory_ctrl
// Description: Compressed Sparse Row (CSR) format memory controller.
//   Stores only non-zero values and their indices. On read, searches for
//   the requested index; returns the value if found, or 0 if not found.
// ============================================================================
module sparse_memory_ctrl #(
    parameter MAX_VALUES  = 16,   // Maximum non-zero values stored
    parameter DATA_WIDTH  = 8,    // Bits per value
    parameter INDEX_WIDTH = 4     // Bits per index (supports 2^4 = 16 positions)
)(
    input  wire                    clk,
    input  wire                    rst,
    // Write interface
    input  wire                    write_en,
    input  wire [DATA_WIDTH-1:0]   write_val,
    input  wire [INDEX_WIDTH-1:0]  write_idx,
    // Read interface
    input  wire                    read_en,
    input  wire [INDEX_WIDTH-1:0]  read_idx,
    output reg  [DATA_WIDTH-1:0]   read_data,
    output reg                     valid_out,
    // Status
    output reg  [$clog2(MAX_VALUES):0] num_stored  // Number of entries stored
);

    // Storage arrays
    reg [DATA_WIDTH-1:0]   values  [0:MAX_VALUES-1];
    reg [INDEX_WIDTH-1:0]  indices [0:MAX_VALUES-1];

    // Write pointer
    reg [$clog2(MAX_VALUES):0] write_ptr;

    // Search variables
    integer i;
    reg found;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            write_ptr  <= 0;
            num_stored <= 0;
            read_data  <= {DATA_WIDTH{1'b0}};
            valid_out  <= 1'b0;
            // Clear storage
            for (i = 0; i < MAX_VALUES; i = i + 1) begin
                values[i]  <= {DATA_WIDTH{1'b0}};
                indices[i] <= {INDEX_WIDTH{1'b0}};
            end
        end else begin
            valid_out <= 1'b0;  // Default

            // Write operation: store value + index pair
            if (write_en && write_ptr < MAX_VALUES) begin
                values[write_ptr]  <= write_val;
                indices[write_ptr] <= write_idx;
                write_ptr  <= write_ptr + 1;
                num_stored <= write_ptr + 1;
            end

            // Read operation: linear search through stored indices
            if (read_en) begin
                found = 1'b0;
                read_data <= {DATA_WIDTH{1'b0}};  // Default: zero
                for (i = 0; i < MAX_VALUES; i = i + 1) begin
                    if (i < write_ptr && indices[i] == read_idx && !found) begin
                        read_data <= values[i];
                        found = 1'b1;
                    end
                end
                valid_out <= 1'b1;
            end
        end
    end

endmodule
