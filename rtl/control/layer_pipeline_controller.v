`timescale 1ns / 1ps

// ============================================================================
// Module: layer_pipeline_controller
// Description: Layer-Level Pipeline Controller for transformer inference.
//   Overlaps stages so multiple tokens process simultaneously.
//   After pipeline fills, throughput = 1 token per longest-stage cycles.
//   REFERENCE: Hennessy & Patterson pipeline architecture.
// ============================================================================
module layer_pipeline_controller #(
    parameter NUM_STAGES   = 5,
    parameter TOKEN_WIDTH  = 8
)(
    input  wire                      clk,
    input  wire                      rst,
    
    input  wire                      token_valid,
    input  wire [TOKEN_WIDTH-1:0]    token_in,
    output wire                      token_ready,
    
    // Packed stage cycles: 8 bits per stage
    input  wire [NUM_STAGES*8-1:0]   stage_cycles_packed,
    
    output reg  [NUM_STAGES-1:0]     stage_active,
    output reg  [NUM_STAGES*TOKEN_WIDTH-1:0] stage_tokens,
    output reg  [NUM_STAGES*8-1:0]   stage_progress_packed,
    
    output reg                       token_out_valid,
    output reg  [TOKEN_WIDTH-1:0]    token_out,
    
    output reg  [31:0]               tokens_processed,
    output reg  [31:0]               total_cycles,
    output reg  [31:0]               pipeline_stalls,
    output reg                       pipeline_full
);

    integer s;
    integer p;
    wire [NUM_STAGES*8-1:0] sc_eff_packed;
    reg [NUM_STAGES-1:0] stage_active_after_advance;
    reg [NUM_STAGES-1:0] next_stage_active;
    reg [NUM_STAGES*TOKEN_WIDTH-1:0] next_stage_tokens;
    reg [NUM_STAGES*8-1:0] next_stage_progress;
    reg                    skid_valid;
    reg [TOKEN_WIDTH-1:0]  skid_token;
    reg                    next_skid_valid;
    reg [TOKEN_WIDTH-1:0]  next_skid_token;
    integer stall_incr;
    wire skid_will_dequeue;
    wire token_fire;
    
    genvar gi;
    generate
        for (gi = 0; gi < NUM_STAGES; gi = gi + 1) begin : unpack
            assign sc_eff_packed[gi*8 +: 8] =
                (stage_cycles_packed[gi*8 +: 8] == 8'd0) ? 8'd1 : stage_cycles_packed[gi*8 +: 8];
        end
    endgenerate

    // 1-entry skid semantics:
    // - stage_cycles=0 is clamped to 1 cycle (via sc_eff) for deterministic behavior.
    // - When skid dequeues into stage0 this cycle, keep token_ready high so the
    //   upstream can enqueue one new token into skid in the same cycle.
    always @* begin
        stage_active_after_advance = stage_active;
        for (p = NUM_STAGES-1; p >= 0; p = p - 1) begin
            if (stage_active_after_advance[p]) begin
                if (stage_progress_packed[p*8 +: 8] + 1 >= sc_eff_packed[p*8 +: 8]) begin
                    if (p == NUM_STAGES-1) begin
                        stage_active_after_advance[p] = 1'b0;
                    end else if (!stage_active_after_advance[p+1]) begin
                        stage_active_after_advance[p+1] = 1'b1;
                        stage_active_after_advance[p]   = 1'b0;
                    end
                end
            end
        end
    end

    assign skid_will_dequeue = skid_valid && !stage_active_after_advance[0];
    assign token_ready = !skid_valid || skid_will_dequeue;
    assign token_fire  = token_valid && token_ready;

    always @(posedge clk) begin
        if (rst) begin
            stage_active         <= 0;
            stage_tokens         <= 0;
            stage_progress_packed <= 0;
            skid_valid          <= 1'b0;
            skid_token          <= 0;
            token_out_valid      <= 1'b0;
            token_out            <= 0;
            tokens_processed     <= 0;
            total_cycles         <= 0;
            pipeline_stalls      <= 0;
            pipeline_full        <= 1'b0;
        end else begin
            token_out_valid <= 1'b0;
            total_cycles    <= total_cycles + 1;

            next_stage_active   = stage_active;
            next_stage_tokens   = stage_tokens;
            next_stage_progress = stage_progress_packed;
            next_skid_valid     = skid_valid;
            next_skid_token     = skid_token;
            stall_incr          = 0;

            // Advance pipeline back-to-front so downstream frees are visible
            // immediately, avoiding ordering bubbles.
            for (s = NUM_STAGES-1; s >= 0; s = s - 1) begin
                if (next_stage_active[s]) begin
                    if (next_stage_progress[s*8 +: 8] + 1 >= sc_eff_packed[s*8 +: 8]) begin
                        if (s == NUM_STAGES-1) begin
                            token_out_valid  <= 1'b1;
                            token_out        <= next_stage_tokens[s*TOKEN_WIDTH +: TOKEN_WIDTH];
                            tokens_processed <= tokens_processed + 1;
                            next_stage_active[s]      = 1'b0;
                            next_stage_progress[s*8 +: 8] = 8'd0;
                        end else if (!next_stage_active[s+1]) begin
                            next_stage_tokens[(s+1)*TOKEN_WIDTH +: TOKEN_WIDTH] =
                                next_stage_tokens[s*TOKEN_WIDTH +: TOKEN_WIDTH];
                            next_stage_progress[(s+1)*8 +: 8] = 8'd0;
                            next_stage_active[s+1] = 1'b1;
                            next_stage_active[s]   = 1'b0;
                            next_stage_progress[s*8 +: 8] = 8'd0;
                        end else begin
                            stall_incr = stall_incr + 1;
                        end
                    end else begin
                        next_stage_progress[s*8 +: 8] = next_stage_progress[s*8 +: 8] + 1'b1;
                    end
                end
            end

            // Ingress valid/ready + one-token skid:
            //   1) If stage0 is free, consume oldest pending token first.
            //   2) Else consume live input when valid/ready fires.
            //   3) If stage0 is busy, accepted live input parks in skid.
            if (!next_stage_active[0]) begin
                if (next_skid_valid) begin
                    next_stage_tokens[0 +: TOKEN_WIDTH] = next_skid_token;
                    next_stage_progress[0 +: 8]  = 8'd0;
                    next_stage_active[0] = 1'b1;
                    next_skid_valid = 1'b0;

                    if (token_fire) begin
                        next_skid_token = token_in;
                        next_skid_valid = 1'b1;
                    end
                end else if (token_fire) begin
                    next_stage_tokens[0 +: TOKEN_WIDTH] = token_in;
                    next_stage_progress[0 +: 8]  = 8'd0;
                    next_stage_active[0] = 1'b1;
                end
            end else if (token_fire) begin
                next_skid_token = token_in;
                next_skid_valid = 1'b1;
            end

            stage_active         <= next_stage_active;
            stage_tokens         <= next_stage_tokens;
            stage_progress_packed <= next_stage_progress;
            skid_valid           <= next_skid_valid;
            skid_token           <= next_skid_token;
            pipeline_full        <= (next_stage_active == {NUM_STAGES{1'b1}});
            pipeline_stalls      <= pipeline_stalls + stall_incr;
        end
    end

endmodule
