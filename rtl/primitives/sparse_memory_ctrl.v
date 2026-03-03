// ============================================================================
// Module: sparse_memory_ctrl
// Description: Direct-mapped sparse memory controller with CSR-like interface.
//   FIXES: Synchronous reset (#6), direct-mapped storage instead of linear
//   search (#4 - sparse_memory_ctrl O(n) issue). Now uses direct-mapped
//   array for O(1) read access. Still tracks num_stored for compatibility.
// ============================================================================
module sparse_memory_ctrl #(
    parameter MAX_VALUES  = 16,
    parameter DATA_WIDTH  = 8,
    parameter INDEX_WIDTH = 4
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
    output reg  [$clog2(MAX_VALUES):0] num_stored
);

    // Direct-mapped storage (O(1) read instead of O(n) search)
    reg [DATA_WIDTH-1:0] values [0:MAX_VALUES-1];
    reg                  occupied [0:MAX_VALUES-1];

    integer i;

    always @(posedge clk) begin  // Synchronous reset
        if (rst) begin
            num_stored <= 0;
            read_data  <= {DATA_WIDTH{1'b0}};
            valid_out  <= 1'b0;
            for (i = 0; i < MAX_VALUES; i = i + 1) begin
                values[i]   <= {DATA_WIDTH{1'b0}};
                occupied[i] <= 1'b0;
            end
        end else begin
            valid_out <= 1'b0;

            // Write: direct-mapped by index
            if (write_en) begin
                values[write_idx] <= write_val;
                if (!occupied[write_idx]) begin
                    occupied[write_idx] <= 1'b1;
                    num_stored <= num_stored + 1;
                end
            end

            // Read: O(1) direct access
            if (read_en) begin
                if (occupied[read_idx])
                    read_data <= values[read_idx];
                else
                    read_data <= {DATA_WIDTH{1'b0}};
                valid_out <= 1'b1;
            end
        end
    end

endmodule
