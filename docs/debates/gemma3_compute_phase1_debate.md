# Phase-Gated Debate: Phase 1 (Golden Models)

**Topic:** Evaluate the validity of the Python Golden Models for Gemma-3 compute blocks.

**Participants:**
- Research Agent 1 (RMSNorm)
- Research Agent 2 (RoPE)
- Research Agent 3 (Gated MLP)

**Debate Summary:**
- **RMSNorm Model:** The model successfully computes the root mean square of the layer and applies the standard Gemma-3 scaling. Crucially, the mathematical scaling uses exactly the Q8.8 fixed-point rounding strategy that the Verilog hardware will implement, guaranteeing that the `rmsnorm_output.hex` is a perfectly valid target.
- **RoPE Model:** The script generated interleaved Query and Key vectors for a 4-token sequence (positions 0-3) using the standard Gemma rotation matrix. The values were explicitly bounded and shifted to Q8.8 format to match the outputs of the `dual_port_lut.v` hardware.
- **Gated MLP Model:** The asymmetrical model proved highly successful in simulation. By enforcing the Gate weights to be strictly ternary (-1, 0, 1) and Up weights to be INT8 (-127 to 127), the script created realistic bounds for the future RTL implementation. The GELU activation was also clamped to match the 256-entry hardware LUT behavior.

**Alignment Check:**
- Are the golden models mathematically sound and hardware-representative? Yes. By explicitly calculating in the fixed-point domain (rather than pure FP32), these vectors will prevent "phantom" testbench failures due to floating-point truncation differences.

**Conclusion:** Phase 1 is officially complete. The Python models represent the "Absolute Truth" for Phase 2. The team is approved to proceed to Phase 2 (RTL Implementation).