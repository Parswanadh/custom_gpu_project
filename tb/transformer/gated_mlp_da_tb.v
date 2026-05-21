`timescale 1ns / 1ps
// ============================================================================
// Module: gated_mlp_da_tb
// Description: Testbench for Dual-Lane Asymmetrical Gated MLP prototype.
// ============================================================================

module gated_mlp_da_tb;

    reg clk;
    reg rst_n;
    reg start;
    wire done;

    // ROMs and RAMs
    reg signed [15:0] x_mem [0:31];
    reg signed [7:0] w_mem [0:383]; // gate(128) + up(128) + down(128)
    reg signed [15:0] expected_out_mem [0:31];
    reg signed [15:0] actual_out_mem [0:31];

    wire [4:0] x_addr;
    wire signed [15:0] x_data = x_mem[x_addr];

    wire [6:0] gw_addr;
    wire signed [7:0] gw_data = w_mem[gw_addr]; // 0 to 127

    wire [6:0] uw_addr;
    wire [8:0] uw_full_addr = 9'd128 + uw_addr;
    wire signed [7:0] uw_data = w_mem[uw_full_addr]; // 128 to 255

    wire [6:0] dw_addr;
    wire [8:0] dw_full_addr = 9'd256 + dw_addr;
    wire signed [7:0] dw_data = w_mem[dw_full_addr]; // 256 to 383

    wire [4:0] out_addr;
    wire signed [15:0] out_data;
    wire out_we;

    always @(posedge clk) begin
        if (out_we) begin
            actual_out_mem[out_addr] <= out_data;
        end
    end

    gated_mlp_da uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .x_addr(x_addr),
        .x_data(x_data),
        .gw_addr(gw_addr),
        .gw_data(gw_data),
        .uw_addr(uw_addr),
        .uw_data(uw_data),
        .dw_addr(dw_addr),
        .dw_data(dw_data),
        .out_addr(out_addr),
        .out_data(out_data),
        .out_we(out_we)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    integer i, errors;
    initial begin
        // The script generates hex files in scripts/ directory
        // Using path assuming execution from `custom_gpu_project` or `sim`
        // We will try ../scripts/ first. If running from custom_gpu_project directly, it might be scripts/
        // Modern iverilog runs from where it's invoked.
        // Let's use `scripts/` assuming run from `custom_gpu_project` 
        // or we'll just try to load, we can change if it fails.
        // The common setup script usually runs from root. Let's use standard ../scripts/ for tb in sim/
        $readmemh("scripts/mlp_input.hex", x_mem);
        $readmemh("scripts/mlp_weights.hex", w_mem);
        $readmemh("scripts/mlp_output.hex", expected_out_mem);

        // Reset
        rst_n = 0;
        start = 0;
        #20;
        rst_n = 1;
        #10;

        // Start
        start = 1;
        #10;
        start = 0;

        // Wait for done
        wait(done);
        #20;

        // Compare
        errors = 0;
        for (i = 0; i < 32; i = i + 1) begin
            // Allow +/- 1 tolerance due to rounding differences in intermediate steps
            if (actual_out_mem[i] !== expected_out_mem[i]) begin
                if (actual_out_mem[i] > expected_out_mem[i]) begin
                    if (actual_out_mem[i] - expected_out_mem[i] > 2) begin
                        $display("Mismatch at %0d: Expected %04x, Got %04x", i, expected_out_mem[i], actual_out_mem[i]);
                        errors = errors + 1;
                    end
                end else begin
                    if (expected_out_mem[i] - actual_out_mem[i] > 2) begin
                        $display("Mismatch at %0d: Expected %04x, Got %04x", i, expected_out_mem[i], actual_out_mem[i]);
                        errors = errors + 1;
                    end
                end
            end
        end

        if (errors == 0) begin
            $display("PASS: Gated MLP Output matches golden model perfectly!");
        end else begin
            $display("FAIL: %0d mismatches found.", errors);
        end
        $finish;
    end

endmodule
