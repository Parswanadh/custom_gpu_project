/*
 * Skid Buffer Formal Verification Specification
 * Proving deadlock-free handshakes using SVA.
 */

module skid_buffer_formal (
    input logic clk,
    input logic rst_n,
    input logic valid_in,
    input logic ready_out,
    input logic ready_for_upstream,
    input logic full // Representing internal buffer state
);

    // Default clocking for formal verification
    default clocking @(posedge clk);
    default disable iff (!rst_n);

    // Safety Property:
    // If buffer is full, we must NOT assert ready_for_upstream to prevent data loss.
    property p_safety_full_no_ready;
        full |-> !ready_for_upstream;
    endproperty

    assert_safety_full_no_ready: assert property (p_safety_full_no_ready)
        else $error("Safety Violation: ready_for_upstream high while buffer full!");

    // Liveness Property:
    // If valid_in is high, valid_out must eventually be high.
    // Assuming downstream eventually accepts via ready_out.
    property p_liveness_valid_in_to_valid_out;
        valid_in |-> s_eventually (ready_out);
    endproperty

    assert_liveness_valid_in_to_valid_out: assert property (p_liveness_valid_in_to_valid_out)
        else $error("Liveness Violation: valid_in asserted but ready_out never occurred!");

endmodule
