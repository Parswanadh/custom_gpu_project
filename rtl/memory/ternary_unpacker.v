// ternary_unpacker.v
// Unpacks 8 ternary weights (2-bit each) from a single 16-bit word.
// Input: [15:0] packed_word (8 * 2 bits)
// Output: [15:0] unpacked_weights (8 x 2-bit values, zero-extended or just mapped)
// Convention: weight[0] = packed_word[1:0], weight[1] = packed_word[3:2], ...

module ternary_unpacker (
    input  wire [15:0] packed_word,
    output wire [15:0] unpacked_weights
);

    // Explicit mapping for clarity
    assign unpacked_weights[1:0]   = packed_word[1:0];
    assign unpacked_weights[3:2]   = packed_word[3:2];
    assign unpacked_weights[5:4]   = packed_word[5:4];
    assign unpacked_weights[7:6]   = packed_word[7:6];
    assign unpacked_weights[9:8]   = packed_word[9:8];
    assign unpacked_weights[11:10] = packed_word[11:10];
    assign unpacked_weights[13:12] = packed_word[13:12];
    assign unpacked_weights[15:14] = packed_word[15:14];

endmodule
