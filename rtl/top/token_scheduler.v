// ============================================================================
// Module: token_scheduler
// Description: Hardware token scheduler for autonomous autoregressive generation.
//   Drives the full token generation loop in hardware without CPU intervention:
//     CPU: "Generate N tokens starting from seed_token"
//     Scheduler: Runs the engine N times, feeds each output back as input
//     CPU: Gets final sequence via interrupt
//
//   This eliminates the CPU overhead of the autoregressive loop — the GPU
//   generates full sequences autonomously.
//
// Parameters: VOCAB_SIZE, MAX_GEN_LEN
// ============================================================================
module token_scheduler #(
    parameter VOCAB_BITS  = 4,    // $clog2(VOCAB_SIZE)
    parameter SEQ_BITS    = 4,    // $clog2(MAX_SEQ_LEN)
    parameter MAX_GEN_LEN = 16    // Maximum tokens to generate
)(
    input  wire                     clk,
    input  wire                     rst,
    
    // Host control
    input  wire                     start,            // Begin generation
    input  wire [VOCAB_BITS-1:0]    seed_token,       // Starting token
    input  wire [7:0]               num_tokens,       // How many to generate
    
    // Engine interface (drives the GPT-2 engine)
    output reg                      engine_valid_in,
    output reg  [VOCAB_BITS-1:0]    engine_token_in,
    output reg  [SEQ_BITS-1:0]      engine_position,
    input  wire                     engine_valid_out,
    input  wire [VOCAB_BITS-1:0]    engine_token_out,
    
    // Output sequence buffer
    output reg  [MAX_GEN_LEN*VOCAB_BITS-1:0] generated_sequence,
    output reg  [7:0]               tokens_generated,
    output reg                      generation_done,
    output reg                      busy
);

    reg [2:0] state;
    localparam S_IDLE        = 3'd0;
    localparam S_FEED_TOKEN  = 3'd1;
    localparam S_WAIT_OUTPUT = 3'd2;
    localparam S_STORE       = 3'd3;
    localparam S_DONE        = 3'd4;
    
    reg [7:0] target_count;
    reg [7:0] current_idx;

    always @(posedge clk) begin
        if (rst) begin
            state              <= S_IDLE;
            engine_valid_in    <= 1'b0;
            engine_token_in    <= 0;
            engine_position    <= 0;
            generated_sequence <= 0;
            tokens_generated   <= 8'd0;
            generation_done    <= 1'b0;
            busy               <= 1'b0;
            target_count       <= 8'd0;
            current_idx        <= 8'd0;
        end else begin
            engine_valid_in <= 1'b0;
            generation_done <= 1'b0;
            
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy             <= 1'b1;
                        target_count     <= num_tokens;
                        current_idx      <= 8'd0;
                        tokens_generated <= 8'd0;
                        engine_token_in  <= seed_token;
                        engine_position  <= 0;
                        // Store seed token
                        generated_sequence[0 +: VOCAB_BITS] <= seed_token;
                        state <= S_FEED_TOKEN;
                    end
                end
                
                S_FEED_TOKEN: begin
                    // Send current token to engine
                    engine_valid_in <= 1'b1;
                    engine_position <= current_idx[SEQ_BITS-1:0];
                    state <= S_WAIT_OUTPUT;
                end
                
                S_WAIT_OUTPUT: begin
                    // Wait for engine to produce next token
                    if (engine_valid_out) begin
                        state <= S_STORE;
                    end
                end
                
                S_STORE: begin
                    // Store generated token and feed it back
                    current_idx <= current_idx + 1;
                    tokens_generated <= tokens_generated + 1;
                    
                    // Store in output buffer
                    generated_sequence[(current_idx + 1) * VOCAB_BITS +: VOCAB_BITS] <= engine_token_out;
                    
                    // Feed back for next iteration
                    engine_token_in <= engine_token_out;
                    
                    if (current_idx + 1 >= target_count) begin
                        state <= S_DONE;
                    end else begin
                        state <= S_FEED_TOKEN;
                    end
                end
                
                S_DONE: begin
                    generation_done <= 1'b1;
                    busy <= 1'b0;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
