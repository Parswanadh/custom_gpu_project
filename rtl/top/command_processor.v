// ============================================================================
// Module: command_processor
// Description: Command queue processor for standalone GPU operation (Issue #18).
//   Reads command descriptors from a FIFO and dispatches operations:
//     - CMD_NOP          (0x00): No operation
//     - CMD_LOAD_WEIGHTS (0x01): Load weights from external memory to SRAM
//     - CMD_MATMUL       (0x02): Matrix multiply using systolic array / gpu_core
//     - CMD_ACTIVATION   (0x03): Apply activation function (GELU/ReLU)
//     - CMD_LAYERNORM    (0x04): Layer normalization
//     - CMD_SOFTMAX      (0x05): Softmax normalization
//     - CMD_RESIDUAL_ADD (0x06): Element-wise add (residual connection)
//     - CMD_EMBEDDING    (0x07): Embedding lookup
//     - CMD_FENCE        (0x0F): Wait for all pending operations to complete
//
//   Command format (64-bit descriptor):
//     [63:56] = opcode
//     [55:48] = flags (activation type, precision mode, etc.)
//     [47:32] = src_addr (scratchpad address)
//     [31:16] = dst_addr (scratchpad address)
//     [15:0]  = size/count
//
//   Connects to: scratchpad, gpu_core, DMA engine, activation units
// ============================================================================
module command_processor #(
    parameter CMD_WIDTH   = 64,
    parameter FIFO_DEPTH  = 16,
    parameter ADDR_WIDTH  = 16
)(
    input  wire                    clk,
    input  wire                    rst,

    // Command input (from host via AXI)
    input  wire                    cmd_valid,
    input  wire [CMD_WIDTH-1:0]    cmd_data,
    output wire                    cmd_ready,

    // Scratchpad interface
    output reg                     sp_read_en,
    output reg  [ADDR_WIDTH-1:0]   sp_read_addr,
    input  wire [15:0]             sp_read_data,
    output reg                     sp_write_en,
    output reg  [ADDR_WIDTH-1:0]   sp_write_addr,
    output reg  [15:0]             sp_write_data,

    // DMA interface
    output reg                     dma_start,
    output reg  [31:0]             dma_ext_addr,
    output reg  [ADDR_WIDTH-1:0]   dma_local_addr,
    output reg  [15:0]             dma_length,
    input  wire                    dma_done,

    // Compute dispatch
    output reg                     compute_start,
    output reg  [7:0]              compute_opcode,
    output reg  [ADDR_WIDTH-1:0]   compute_src_addr,
    output reg  [ADDR_WIDTH-1:0]   compute_dst_addr,
    output reg  [15:0]             compute_size,
    output reg  [7:0]              compute_flags,
    input  wire                    compute_done,

    // Status
    output reg                     busy,
    output reg                     idle,
    output reg  [31:0]             cmds_executed,

    // Interrupt
    output reg                     interrupt_out       // Pulse when FENCE completes
);

    // Opcodes
    localparam CMD_NOP          = 8'h00;
    localparam CMD_LOAD_WEIGHTS = 8'h01;
    localparam CMD_MATMUL       = 8'h02;
    localparam CMD_ACTIVATION   = 8'h03;
    localparam CMD_LAYERNORM    = 8'h04;
    localparam CMD_SOFTMAX      = 8'h05;
    localparam CMD_RESIDUAL_ADD = 8'h06;
    localparam CMD_EMBEDDING    = 8'h07;
    localparam CMD_FENCE        = 8'h0F;

    // Command FIFO
    reg [CMD_WIDTH-1:0] fifo_mem [0:FIFO_DEPTH-1];
    reg [$clog2(FIFO_DEPTH):0] fifo_wr_ptr, fifo_rd_ptr, fifo_count;

    wire fifo_full  = (fifo_count == FIFO_DEPTH);
    wire fifo_empty = (fifo_count == 0);
    assign cmd_ready = ~fifo_full;

    // FIFO write
    always @(posedge clk) begin
        if (rst) begin
            fifo_wr_ptr <= 0;
            fifo_count  <= 0;
        end else begin
            if (cmd_valid && !fifo_full) begin
                fifo_mem[fifo_wr_ptr[$clog2(FIFO_DEPTH)-1:0]] <= cmd_data;
                fifo_wr_ptr <= fifo_wr_ptr + 1;
            end
            // Update count
            if (cmd_valid && !fifo_full && !(state == DISPATCH && !fifo_empty))
                fifo_count <= fifo_count + 1;
            else if (!(cmd_valid && !fifo_full) && (state == DISPATCH && !fifo_empty))
                fifo_count <= fifo_count - 1;
        end
    end

    // State machine
    reg [3:0] state;
    localparam IDLE_ST    = 4'd0;
    localparam DISPATCH   = 4'd1;
    localparam WAIT_DMA   = 4'd2;
    localparam WAIT_COMP  = 4'd3;
    localparam FENCE_WAIT = 4'd4;

    // Current command decode
    reg [CMD_WIDTH-1:0] cur_cmd;
    wire [7:0]  cur_opcode = cur_cmd[63:56];
    wire [7:0]  cur_flags  = cur_cmd[55:48];
    wire [15:0] cur_src    = cur_cmd[47:32];
    wire [15:0] cur_dst    = cur_cmd[31:16];
    wire [15:0] cur_size   = cur_cmd[15:0];

    always @(posedge clk) begin
        if (rst) begin
            state          <= IDLE_ST;
            busy           <= 1'b0;
            idle           <= 1'b1;
            cmds_executed  <= 32'd0;
            interrupt_out  <= 1'b0;
            dma_start      <= 1'b0;
            compute_start  <= 1'b0;
            sp_read_en     <= 1'b0;
            sp_write_en    <= 1'b0;
            fifo_rd_ptr    <= 0;
        end else begin
            interrupt_out  <= 1'b0;
            dma_start      <= 1'b0;
            compute_start  <= 1'b0;
            sp_read_en     <= 1'b0;
            sp_write_en    <= 1'b0;

            case (state)
                IDLE_ST: begin
                    idle <= 1'b1;
                    busy <= 1'b0;
                    if (!fifo_empty) begin
                        cur_cmd <= fifo_mem[fifo_rd_ptr[$clog2(FIFO_DEPTH)-1:0]];
                        fifo_rd_ptr <= fifo_rd_ptr + 1;
                        state <= DISPATCH;
                        idle <= 1'b0;
                        busy <= 1'b1;
                    end
                end

                DISPATCH: begin
                    case (cur_opcode)
                        CMD_NOP: begin
                            cmds_executed <= cmds_executed + 1;
                            state <= IDLE_ST;
                        end

                        CMD_LOAD_WEIGHTS: begin
                            dma_start      <= 1'b1;
                            dma_ext_addr   <= {16'd0, cur_src};
                            dma_local_addr <= cur_dst;
                            dma_length     <= cur_size;
                            state <= WAIT_DMA;
                        end

                        CMD_MATMUL, CMD_ACTIVATION, CMD_LAYERNORM,
                        CMD_SOFTMAX, CMD_RESIDUAL_ADD, CMD_EMBEDDING: begin
                            compute_start    <= 1'b1;
                            compute_opcode   <= cur_opcode;
                            compute_src_addr <= cur_src;
                            compute_dst_addr <= cur_dst;
                            compute_size     <= cur_size;
                            compute_flags    <= cur_flags;
                            state <= WAIT_COMP;
                        end

                        CMD_FENCE: begin
                            state <= FENCE_WAIT;
                        end

                        default: begin
                            cmds_executed <= cmds_executed + 1;
                            state <= IDLE_ST;
                        end
                    endcase
                end

                WAIT_DMA: begin
                    if (dma_done) begin
                        cmds_executed <= cmds_executed + 1;
                        state <= IDLE_ST;
                    end
                end

                WAIT_COMP: begin
                    if (compute_done) begin
                        cmds_executed <= cmds_executed + 1;
                        state <= IDLE_ST;
                    end
                end

                FENCE_WAIT: begin
                    // FENCE: just ensure all prior commands finished (they're sequential)
                    interrupt_out <= 1'b1;
                    cmds_executed <= cmds_executed + 1;
                    state <= IDLE_ST;
                end
            endcase
        end
    end

endmodule

