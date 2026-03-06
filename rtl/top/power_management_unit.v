// ============================================================================
// Module: power_management_unit
// Description: Hardware power management for edge AI deployment.
//   Monitors compute activity and controls power states:
//     - FULL:  All cores enabled, maximum performance
//     - ECO:   Idle cores clock-gated, reduced power
//     - SLEEP: All cores disabled, minimal power (wake on interrupt)
//
//   Features:
//     - Per-core clock enable signals
//     - Auto-sleep after configurable idle timeout
//     - Activity monitoring with hysteresis
//     - Power mode register (software-controlled override)
//     - Estimated power savings counter
//
// Parameters: NUM_CORES, IDLE_TIMEOUT
// ============================================================================
module power_management_unit #(
    parameter NUM_CORES    = 4,
    parameter IDLE_TIMEOUT = 256   // Clock cycles before auto-sleep
)(
    input  wire                     clk,
    input  wire                     rst,
    
    // Activity inputs (one per core)
    input  wire [NUM_CORES-1:0]     core_active,     // 1 = core is computing
    
    // Power mode control
    input  wire [1:0]               power_mode_req,  // 00=AUTO, 01=FULL, 10=ECO, 11=SLEEP
    input  wire                     wake_interrupt,   // Wake from SLEEP
    
    // Power control outputs
    output reg  [NUM_CORES-1:0]     core_clk_en,     // Clock enable per core
    output reg  [1:0]               current_mode,    // Current power mode
    
    // Monitoring outputs
    output reg  [31:0]              idle_cycles,      // Total idle cycles
    output reg  [31:0]              active_cycles,    // Total active cycles
    output reg  [31:0]              sleep_cycles,     // Total sleep cycles
    output reg  [31:0]              gated_core_cycles // Cycles saved by gating
);

    localparam MODE_FULL  = 2'd0;
    localparam MODE_ECO   = 2'd1;
    localparam MODE_SLEEP = 2'd2;
    
    reg [15:0] idle_counter;  // Counts consecutive idle cycles
    wire any_active = |core_active;
    
    integer i;

    always @(posedge clk) begin
        if (rst) begin
            core_clk_en       <= {NUM_CORES{1'b1}};  // All enabled on reset
            current_mode      <= MODE_FULL;
            idle_cycles       <= 32'd0;
            active_cycles     <= 32'd0;
            sleep_cycles      <= 32'd0;
            gated_core_cycles <= 32'd0;
            idle_counter      <= 16'd0;
        end else begin
            // ---- Mode selection logic ----
            case (power_mode_req)
                2'b01: current_mode <= MODE_FULL;   // Forced FULL
                2'b10: current_mode <= MODE_ECO;    // Forced ECO
                2'b11: current_mode <= MODE_SLEEP;  // Forced SLEEP
                default: begin  // AUTO mode
                    if (any_active) begin
                        current_mode <= MODE_FULL;
                        idle_counter <= 16'd0;
                    end else begin
                        idle_counter <= idle_counter + 1;
                        if (idle_counter >= IDLE_TIMEOUT)
                            current_mode <= MODE_SLEEP;
                        else if (idle_counter >= (IDLE_TIMEOUT >> 2))
                            current_mode <= MODE_ECO;
                    end
                    
                    // Wake from sleep on interrupt
                    if (current_mode == MODE_SLEEP && wake_interrupt) begin
                        current_mode <= MODE_FULL;
                        idle_counter <= 16'd0;  // Reset so we don't auto-sleep again
                    end
                end
            endcase
            
            // ---- Clock enable logic based on mode ----
            case (current_mode)
                MODE_FULL: begin
                    core_clk_en <= {NUM_CORES{1'b1}};  // All cores on
                    if (any_active)
                        active_cycles <= active_cycles + 1;
                    else
                        idle_cycles <= idle_cycles + 1;
                end
                
                MODE_ECO: begin
                    // Only enable cores that are active
                    core_clk_en <= core_active;
                    idle_cycles <= idle_cycles + 1;
                    // Count gated cycles
                    for (i = 0; i < NUM_CORES; i = i + 1)
                        if (!core_active[i])
                            gated_core_cycles <= gated_core_cycles + 1;
                end
                
                MODE_SLEEP: begin
                    core_clk_en <= {NUM_CORES{1'b0}};  // All cores off
                    sleep_cycles <= sleep_cycles + 1;
                    gated_core_cycles <= gated_core_cycles + NUM_CORES;
                end
                
                default: begin
                    core_clk_en <= {NUM_CORES{1'b1}};
                end
            endcase
        end
    end

endmodule
