// ============================================================================
// Module: sparse_memory_ctrl_wide
// Description: Wide-port sparse memory controller with prefetch buffer.
//   Reads 4 values per cycle (vs 1 in original sparse_memory_ctrl).
//
//   FIXES APPLIED:
//     - Issue #16: Parity bits on all stored data with error detection
//     - Issue #6:  Synchronous reset
// ============================================================================
module sparse_memory_ctrl_wide #(
    parameter MAX_VALUES  = 64,         // Total storage capacity
    parameter DATA_WIDTH  = 8,          // Width of each value
    parameter INDEX_WIDTH = 6,          // Address width (log2 of MAX_VALUES)
    parameter READ_WIDTH  = 4           // Number of values read per cycle
)(
    input  wire                              clk,
    input  wire                              rst,
    // Write interface
    input  wire                              write_en,
    input  wire [DATA_WIDTH-1:0]             write_val,
    input  wire [INDEX_WIDTH-1:0]            write_idx,
    // Wide read interface
    input  wire                              read_en,
    input  wire [INDEX_WIDTH-1:0]            read_base_idx,
    output reg  [READ_WIDTH*DATA_WIDTH-1:0]  read_data,
    output reg                               valid_out,
    // Prefetch interface
    input  wire                              prefetch_en,
    input  wire [INDEX_WIDTH-1:0]            prefetch_base_idx,
    output reg                               prefetch_ready,
    // Status
    output wire [INDEX_WIDTH:0]              num_stored,
    // Error flag (Issue #16)
    output reg                               parity_error
);

    // Storage arrays with parity
    reg [DATA_WIDTH-1:0] values [0:MAX_VALUES-1];
    reg                  par    [0:MAX_VALUES-1];  // Parity bits
    reg [INDEX_WIDTH:0]  count;

    // Prefetch buffer
    reg [READ_WIDTH*DATA_WIDTH-1:0] prefetch_buf;
    reg                              prefetch_valid;

    integer i;

    assign num_stored = count;

    // Write logic with parity generation
    always @(posedge clk) begin
        if (rst) begin
            count <= 0;
            prefetch_valid <= 1'b0;
            prefetch_ready <= 1'b0;
            parity_error   <= 1'b0;
            for (i = 0; i < MAX_VALUES; i = i + 1) begin
                values[i] <= {DATA_WIDTH{1'b0}};
                par[i]    <= 1'b0;
            end
        end else begin
            if (write_en) begin
                values[write_idx] <= write_val;
                par[write_idx]    <= ^write_val;  // Parity generation
                if (write_idx >= count)
                    count <= write_idx + 1;
            end

            // Prefetch
            if (prefetch_en) begin
                for (i = 0; i < READ_WIDTH; i = i + 1) begin
                    if (prefetch_base_idx + i < MAX_VALUES)
                        prefetch_buf[i*DATA_WIDTH +: DATA_WIDTH] <= values[prefetch_base_idx + i];
                    else
                        prefetch_buf[i*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                end
                prefetch_valid <= 1'b1;
                prefetch_ready <= 1'b1;
            end
        end
    end

    // Wide read logic with parity check
    always @(posedge clk) begin
        if (rst) begin
            read_data    <= {(READ_WIDTH*DATA_WIDTH){1'b0}};
            valid_out    <= 1'b0;
        end else if (read_en) begin
            parity_error <= 1'b0;
            for (i = 0; i < READ_WIDTH; i = i + 1) begin
                if (read_base_idx + i < MAX_VALUES) begin
                    read_data[i*DATA_WIDTH +: DATA_WIDTH] <= values[read_base_idx + i];
                    // Parity check on read (Issue #16)
                    if (^values[read_base_idx + i] != par[read_base_idx + i])
                        parity_error <= 1'b1;
                end else begin
                    read_data[i*DATA_WIDTH +: DATA_WIDTH] <= {DATA_WIDTH{1'b0}};
                end
            end
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
