`timescale 1ns / 1ps

// ============================================================================
// Module: simd_ternary_engine
// Description: SIMD Ternary MAC — processes MULTIPLE weights per cycle.
//
//   Our original ternary_mac_engine processes 1 weight/cycle = 19 cycles
//   for 16 weights. This SIMD version processes LANES weights per cycle
//   using a parallel adder tree.
//
//   With LANES=4: 16 weights / 4 = 4 compute cycles + overhead ≈ 7 cycles
//   With LANES=8: 16 weights / 8 = 2 compute cycles + overhead ≈ 5 cycles
//
//   INSIGHT: Each ternary weight is only 2 bits. A 32-bit SRAM word holds
//   16 weights. We can decode ALL of them simultaneously, then reduce
//   with a tree of adders. Add/sub gates are tiny (~50 gates each).
//
//   REFERENCE: BitNet 1.58 (Microsoft, 2024) + SIMD vectorization
// ============================================================================
module simd_ternary_engine #(
    parameter NUM_WEIGHTS = 16,       // Total weights per dot product
    parameter LANES       = 4,        // Parallel processing lanes
    parameter DATA_WIDTH  = 8,
    parameter ACC_WIDTH   = 24
)(
    input  wire                        clk,
    input  wire                        rst,
    input  wire                        start,
    
    // Weight memory: 2 bits per weight, packed
    input  wire [2*LANES-1:0]          weight_chunk,     // LANES weights per cycle
    output reg  [$clog2(NUM_WEIGHTS/LANES):0] weight_addr,
    
    // Activation: all lanes use same activation (broadcast)
    input  wire [DATA_WIDTH-1:0]       activation_in,
    output reg  [$clog2(NUM_WEIGHTS)-1:0] activation_addr,
    
    output reg  signed [ACC_WIDTH-1:0] result,
    output reg                         done,
    
    // Statistics
    output reg  [15:0]                 total_adds,
    output reg  [15:0]                 total_subs,
    output reg  [15:0]                 total_skips,
    output reg  [15:0]                 cycles_used
);

    localparam NUM_CHUNKS = NUM_WEIGHTS / LANES;
    
    reg [2:0] state;
    localparam IDLE    = 3'd0;
    localparam LOAD    = 3'd1;
    localparam COMPUTE = 3'd2;
    localparam DONE_ST = 3'd3;
    
    reg [$clog2(NUM_CHUNKS):0] chunk_idx;
    reg signed [ACC_WIDTH-1:0] accumulator;
    reg [15:0] cycle_counter;
    
    // SIMD lane results — computed combinationally
    wire signed [DATA_WIDTH:0] lane_result [0:LANES-1];
    wire [1:0] lane_weight [0:LANES-1];
    
    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : lanes
            assign lane_weight[g] = weight_chunk[g*2 +: 2];
            assign lane_result[g] = (lane_weight[g] == 2'b01) ?  {1'b0, activation_in} :  // +1 → add
                                    (lane_weight[g] == 2'b10) ? -{1'b0, activation_in} :  // -1 → sub
                                                                  {(DATA_WIDTH+1){1'b0}};  //  0 → skip
        end
    endgenerate
    
    // Parallel reduction tree (all lanes summed in 1 cycle)
    wire signed [ACC_WIDTH-1:0] lane_sum;
    generate
        if (LANES == 4) begin : reduce4
            wire signed [DATA_WIDTH+1:0] pair0 = lane_result[0] + lane_result[1];
            wire signed [DATA_WIDTH+1:0] pair1 = lane_result[2] + lane_result[3];
            assign lane_sum = {{(ACC_WIDTH-DATA_WIDTH-2){pair0[DATA_WIDTH+1]}}, pair0} + 
                              {{(ACC_WIDTH-DATA_WIDTH-2){pair1[DATA_WIDTH+1]}}, pair1};
        end else if (LANES == 8) begin : reduce8
            wire signed [DATA_WIDTH+1:0] p0 = lane_result[0] + lane_result[1];
            wire signed [DATA_WIDTH+1:0] p1 = lane_result[2] + lane_result[3];
            wire signed [DATA_WIDTH+1:0] p2 = lane_result[4] + lane_result[5];
            wire signed [DATA_WIDTH+1:0] p3 = lane_result[6] + lane_result[7];
            wire signed [DATA_WIDTH+2:0] q0 = p0 + p1;
            wire signed [DATA_WIDTH+2:0] q1 = p2 + p3;
            assign lane_sum = {{(ACC_WIDTH-DATA_WIDTH-3){q0[DATA_WIDTH+2]}}, q0} + 
                              {{(ACC_WIDTH-DATA_WIDTH-3){q1[DATA_WIDTH+2]}}, q1};
        end else begin : reduce2
            assign lane_sum = {{(ACC_WIDTH-DATA_WIDTH-1){lane_result[0][DATA_WIDTH]}}, lane_result[0]} + 
                              {{(ACC_WIDTH-DATA_WIDTH-1){lane_result[1][DATA_WIDTH]}}, lane_result[1]};
        end
    endgenerate

    integer l;
    reg [7:0] adds_this_cycle, subs_this_cycle, skips_this_cycle;

    always @(*) begin
        adds_this_cycle = 8'd0;
        subs_this_cycle = 8'd0;
        skips_this_cycle = 8'd0;
        for (l = 0; l < LANES; l = l + 1) begin
            if (lane_weight[l] == 2'b01)
                adds_this_cycle = adds_this_cycle + 1'b1;
            else if (lane_weight[l] == 2'b10)
                subs_this_cycle = subs_this_cycle + 1'b1;
            else
                skips_this_cycle = skips_this_cycle + 1'b1;
        end
    end
    
    always @(posedge clk) begin
        if (rst) begin
            state        <= IDLE;
            done         <= 1'b0;
            result       <= 0;
            accumulator  <= 0;
            chunk_idx    <= 0;
            weight_addr  <= 0;
            activation_addr <= 0;
            cycle_counter <= 0;
            total_adds   <= 0;
            total_subs   <= 0;
            total_skips  <= 0;
            cycles_used  <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        accumulator   <= 0;
                        chunk_idx     <= 0;
                        weight_addr   <= 0;
                        activation_addr <= 0;
                        cycle_counter <= 0;
                        state <= COMPUTE;
                    end
                end
                
                COMPUTE: begin
                    // ALL LANES compute simultaneously — this is the SIMD advantage
                    accumulator <= accumulator + lane_sum;
                    cycle_counter <= cycle_counter + 1;
                    
                    // Track stats
                    total_adds <= total_adds + adds_this_cycle;
                    total_subs <= total_subs + subs_this_cycle;
                    total_skips <= total_skips + skips_this_cycle;
                    
                    chunk_idx <= chunk_idx + 1;
                    weight_addr <= weight_addr + 1;
                    activation_addr <= activation_addr + LANES;
                    
                    if (chunk_idx + 1 >= NUM_CHUNKS)
                        state <= DONE_ST;
                end
                
                DONE_ST: begin
                    result      <= accumulator;
                    done        <= 1'b1;
                    cycles_used <= cycle_counter;
                    state       <= IDLE;
                end
            endcase
        end
    end

endmodule
