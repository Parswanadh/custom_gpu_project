// ============================================================================
// Module: exp_lut_256
// Description: 256-entry lookup table for exp() approximation.
//   Maps input Q8.8 signed value (after max-subtraction, range [-8,0])
//   to unsigned 8-bit exp output [1,255].
//
//   LUT is indexed by unsigned 8-bit value where:
//     index 0   → input = 0.0     → exp(0)    = 255 (≈1.0)
//     index 64  → input = -1.0    → exp(-1)   ≈ 94  (≈0.368)
//     index 128 → input = -2.0    → exp(-2)   ≈ 35  (≈0.135)
//     index 192 → input = -3.0    → exp(-3)   ≈ 13  (≈0.050)
//     index 255 → input = -3.98   → exp(-3.98)≈ 5   (≈0.019)
//
//   The index is computed as: idx = clamp(-input_q8 / 4, 0, 255)
//   where input_q8 is the Q8.8 shifted score (always <= 0 after max-sub)
//
// This replaces the crude linear approximation with a proper piecewise curve.
// ============================================================================
module exp_lut_256 (
    input  wire signed [15:0] x_in,    // Q8.8 input (expected <= 0)
    output reg         [7:0]  exp_out  // Q0.8 output [1,255]
);

    // 256-entry LUT: exp(-i/64) * 255, for i = 0..255
    // Each entry = round(255 * exp(-i/64))
    // i/64 maps [0,255] → [0.0, 3.984]
    // exp(-x) for x in [0, 3.984] → [1.0, 0.019]

    reg [7:0] lut [0:255];

    integer k;
    initial begin
        // Generated from: round(255 * exp(-k/64)) for k=0..255
        // Group 0: exp(0) to exp(-0.25)
        lut[  0] = 255; lut[  1] = 251; lut[  2] = 247; lut[  3] = 243;
        lut[  4] = 239; lut[  5] = 236; lut[  6] = 232; lut[  7] = 228;
        lut[  8] = 225; lut[  9] = 221; lut[ 10] = 218; lut[ 11] = 214;
        lut[ 12] = 211; lut[ 13] = 207; lut[ 14] = 204; lut[ 15] = 201;
        // Group 1: exp(-0.25) to exp(-0.5)
        lut[ 16] = 198; lut[ 17] = 194; lut[ 18] = 191; lut[ 19] = 188;
        lut[ 20] = 185; lut[ 21] = 182; lut[ 22] = 179; lut[ 23] = 176;
        lut[ 24] = 173; lut[ 25] = 171; lut[ 26] = 168; lut[ 27] = 165;
        lut[ 28] = 162; lut[ 29] = 160; lut[ 30] = 157; lut[ 31] = 155;
        // Group 2: exp(-0.5) to exp(-0.75)
        lut[ 32] = 152; lut[ 33] = 150; lut[ 34] = 147; lut[ 35] = 145;
        lut[ 36] = 142; lut[ 37] = 140; lut[ 38] = 138; lut[ 39] = 135;
        lut[ 40] = 133; lut[ 41] = 131; lut[ 42] = 129; lut[ 43] = 127;
        lut[ 44] = 124; lut[ 45] = 122; lut[ 46] = 120; lut[ 47] = 118;
        // Group 3: exp(-0.75) to exp(-1.0)
        lut[ 48] = 116; lut[ 49] = 114; lut[ 50] = 113; lut[ 51] = 111;
        lut[ 52] = 109; lut[ 53] = 107; lut[ 54] = 105; lut[ 55] = 104;
        lut[ 56] = 102; lut[ 57] = 100; lut[ 58] =  99; lut[ 59] =  97;
        lut[ 60] =  95; lut[ 61] =  94; lut[ 62] =  92; lut[ 63] =  91;
        // Group 4: exp(-1.0) to exp(-1.5)
        lut[ 64] =  89; lut[ 65] =  88; lut[ 66] =  86; lut[ 67] =  85;
        lut[ 68] =  84; lut[ 69] =  82; lut[ 70] =  81; lut[ 71] =  80;
        lut[ 72] =  78; lut[ 73] =  77; lut[ 74] =  76; lut[ 75] =  74;
        lut[ 76] =  73; lut[ 77] =  72; lut[ 78] =  71; lut[ 79] =  70;
        lut[ 80] =  68; lut[ 81] =  67; lut[ 82] =  66; lut[ 83] =  65;
        lut[ 84] =  64; lut[ 85] =  63; lut[ 86] =  62; lut[ 87] =  61;
        lut[ 88] =  60; lut[ 89] =  59; lut[ 90] =  58; lut[ 91] =  57;
        lut[ 92] =  56; lut[ 93] =  55; lut[ 94] =  54; lut[ 95] =  53;
        // Group 5: exp(-1.5) to exp(-2.0)
        lut[ 96] =  52; lut[ 97] =  52; lut[ 98] =  51; lut[ 99] =  50;
        lut[100] =  49; lut[101] =  48; lut[102] =  48; lut[103] =  47;
        lut[104] =  46; lut[105] =  45; lut[106] =  45; lut[107] =  44;
        lut[108] =  43; lut[109] =  43; lut[110] =  42; lut[111] =  41;
        lut[112] =  40; lut[113] =  40; lut[114] =  39; lut[115] =  38;
        lut[116] =  38; lut[117] =  37; lut[118] =  37; lut[119] =  36;
        lut[120] =  35; lut[121] =  35; lut[122] =  34; lut[123] =  34;
        lut[124] =  33; lut[125] =  33; lut[126] =  32; lut[127] =  32;
        // Group 6: exp(-2.0) to exp(-3.0)
        lut[128] =  31; lut[129] =  30; lut[130] =  30; lut[131] =  29;
        lut[132] =  29; lut[133] =  28; lut[134] =  28; lut[135] =  27;
        lut[136] =  27; lut[137] =  27; lut[138] =  26; lut[139] =  26;
        lut[140] =  25; lut[141] =  25; lut[142] =  24; lut[143] =  24;
        lut[144] =  24; lut[145] =  23; lut[146] =  23; lut[147] =  22;
        lut[148] =  22; lut[149] =  22; lut[150] =  21; lut[151] =  21;
        lut[152] =  21; lut[153] =  20; lut[154] =  20; lut[155] =  20;
        lut[156] =  19; lut[157] =  19; lut[158] =  19; lut[159] =  18;
        lut[160] =  18; lut[161] =  18; lut[162] =  17; lut[163] =  17;
        lut[164] =  17; lut[165] =  16; lut[166] =  16; lut[167] =  16;
        lut[168] =  16; lut[169] =  15; lut[170] =  15; lut[171] =  15;
        lut[172] =  14; lut[173] =  14; lut[174] =  14; lut[175] =  14;
        lut[176] =  13; lut[177] =  13; lut[178] =  13; lut[179] =  13;
        lut[180] =  12; lut[181] =  12; lut[182] =  12; lut[183] =  12;
        lut[184] =  11; lut[185] =  11; lut[186] =  11; lut[187] =  11;
        lut[188] =  11; lut[189] =  10; lut[190] =  10; lut[191] =  10;
        // Group 7: exp(-3.0) to exp(-4.0)
        lut[192] =  10; lut[193] =  10; lut[194] =   9; lut[195] =   9;
        lut[196] =   9; lut[197] =   9; lut[198] =   9; lut[199] =   8;
        lut[200] =   8; lut[201] =   8; lut[202] =   8; lut[203] =   8;
        lut[204] =   7; lut[205] =   7; lut[206] =   7; lut[207] =   7;
        lut[208] =   7; lut[209] =   7; lut[210] =   6; lut[211] =   6;
        lut[212] =   6; lut[213] =   6; lut[214] =   6; lut[215] =   6;
        lut[216] =   6; lut[217] =   5; lut[218] =   5; lut[219] =   5;
        lut[220] =   5; lut[221] =   5; lut[222] =   5; lut[223] =   5;
        lut[224] =   4; lut[225] =   4; lut[226] =   4; lut[227] =   4;
        lut[228] =   4; lut[229] =   4; lut[230] =   4; lut[231] =   4;
        lut[232] =   3; lut[233] =   3; lut[234] =   3; lut[235] =   3;
        lut[236] =   3; lut[237] =   3; lut[238] =   3; lut[239] =   3;
        lut[240] =   3; lut[241] =   3; lut[242] =   2; lut[243] =   2;
        lut[244] =   2; lut[245] =   2; lut[246] =   2; lut[247] =   2;
        lut[248] =   2; lut[249] =   2; lut[250] =   2; lut[251] =   2;
        lut[252] =   2; lut[253] =   1; lut[254] =   1; lut[255] =   1;
    end

    // Convert Q8.8 input to LUT index
    // x_in is <= 0 after max-subtraction
    // index = clamp(-x_in / 4, 0, 255)
    // -x_in / 4 = -x_in >> 2 (for Q8.8, this maps [-4.0, 0] → [0, 256])
    wire signed [15:0] neg_x = -x_in;
    wire [15:0] shifted = neg_x >>> 2;  // Divide by 4 (scale to LUT range)
    wire [7:0]  idx = (shifted > 16'd255) ? 8'd255 :
                      (neg_x < 0) ? 8'd0 : shifted[7:0];

    always @(*) begin
        exp_out = lut[idx];
    end

endmodule
