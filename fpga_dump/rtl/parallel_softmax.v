`timescale 1ns / 1ps

module parallel_softmax #(
    parameter VECTOR_LEN = 4,
    parameter PARALLEL_UNITS = 4,
    parameter DATA_WIDTH = 16
) (
    input wire clk,
    input wire rst,
    input wire valid_in,
    input wire [VECTOR_LEN*DATA_WIDTH-1:0] x_in,
    output reg valid_out,
    output reg [VECTOR_LEN*8-1:0] prob_out,
    output reg [15:0] cycles_used
);

    localparam IDLE      = 3'd0;
    localparam FIND_MAX  = 3'd1;
    localparam CALC_EXP  = 3'd2;
    localparam SUM_EXP   = 3'd3;
    localparam RECIP     = 3'd4;
    localparam NORMALIZE = 3'd5;
    localparam DONE_ST   = 3'd6;

    reg [2:0] state;
    reg [15:0] cycle_cnt;

    reg signed [DATA_WIDTH-1:0] x_buf [0:VECTOR_LEN-1];
    reg signed [DATA_WIDTH-1:0] max_val;
    reg signed [DATA_WIDTH-1:0] exp_lut_in [0:VECTOR_LEN-1];
    reg [7:0] exp_val [0:VECTOR_LEN-1];
    reg [15:0] exp_sum;
    reg [7:0] recip_idx;
    reg [15:0] recip_scale;

    integer i;
    reg signed [DATA_WIDTH:0] diff_calc;
    reg signed [DATA_WIDTH:0] diff_scaled;
    reg signed [DATA_WIDTH-1:0] exp_input_calc;
    reg [7:0] exp_scaled;
    reg [15:0] exp_sum_reduce;
    reg [31:0] norm_mul1;
    reg [47:0] norm_mul2;
    reg [15:0] prob_tmp;
    reg [7:0] prob_calc [0:VECTOR_LEN-1];
    reg [15:0] prob_sum_calc;
    reg [3:0] prob_max_idx;
    reg [7:0] prob_max_val;
    reg [15:0] sum_delta;

    wire [7:0] exp_lut_out [0:VECTOR_LEN-1];
    wire [15:0] recip_lut_out;

    genvar gi;
    generate
        for (gi = 0; gi < VECTOR_LEN; gi = gi + 1) begin : gen_exp_luts
            exp_lut_256 u_exp_lut (
                .x_in(exp_lut_in[gi]),
                .exp_out(exp_lut_out[gi])
            );
        end
    endgenerate

    recip_lut_256 u_recip_lut (
        .x_in(recip_idx),
        .recip_out(recip_lut_out)
    );

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            valid_out <= 1'b0;
            prob_out <= 0;
            cycles_used <= 0;
            cycle_cnt <= 0;
            max_val <= 0;
            exp_sum <= 16'd0;
            recip_idx <= 8'd1;
            recip_scale <= 16'd0;
            for (i = 0; i < VECTOR_LEN; i = i + 1) begin
                x_buf[i] <= 0;
                exp_lut_in[i] <= 0;
                exp_val[i] <= 0;
                prob_calc[i] <= 0;
            end
        end else begin
            valid_out <= 1'b0;

            case (state)
                IDLE: begin
                    if (valid_in) begin
                        for (i = 0; i < VECTOR_LEN; i = i + 1)
                            x_buf[i] <= x_in[i*DATA_WIDTH +: DATA_WIDTH];
                        cycle_cnt <= 16'd0;
                        state <= FIND_MAX;
                    end
                end

                FIND_MAX: begin
                    max_val = x_buf[0];
                    for (i = 1; i < VECTOR_LEN; i = i + 1)
                        if (x_buf[i] > max_val)
                            max_val = x_buf[i];
                    cycle_cnt <= cycle_cnt + 1'b1;
                    state <= CALC_EXP;
                end

                CALC_EXP: begin
                    for (i = 0; i < VECTOR_LEN; i = i + 1) begin
                        diff_calc = x_buf[i] - max_val;
                        diff_scaled = diff_calc >>> 3;
                        if (diff_scaled < -127)
                            exp_input_calc = -127;
                        else if (diff_scaled > 0)
                            exp_input_calc = 0;
                        else
                            exp_input_calc = diff_scaled[DATA_WIDTH-1:0];
                        exp_lut_in[i] <= exp_input_calc;
                    end
                    cycle_cnt <= cycle_cnt + 1'b1;
                    state <= SUM_EXP;
                end

                SUM_EXP: begin
                    exp_sum_reduce = 16'd0;
                    for (i = 0; i < VECTOR_LEN; i = i + 1) begin
                        exp_scaled = exp_lut_out[i] >> 2;
                        if (exp_scaled == 8'd0)
                            exp_scaled = 8'd1;
                        exp_val[i] <= exp_scaled;
                        exp_sum_reduce = exp_sum_reduce + {8'd0, exp_scaled};
                    end
                    if (exp_sum_reduce == 16'd0)
                        exp_sum_reduce = 16'd1;
                    exp_sum <= exp_sum_reduce;
                    recip_idx <= exp_sum_reduce[7:0];
                    cycle_cnt <= cycle_cnt + 1'b1;
                    state <= RECIP;
                end

                RECIP: begin
                    recip_scale <= recip_lut_out;
                    cycle_cnt <= cycle_cnt + 1'b1;
                    state <= NORMALIZE;
                end

                NORMALIZE: begin
                    prob_sum_calc = 16'd0;
                    prob_max_idx = 4'd0;
                    prob_max_val = 8'd0;
                    for (i = 0; i < VECTOR_LEN; i = i + 1) begin
                        norm_mul1 = {8'd0, exp_val[i]} * 16'd255;
                        norm_mul2 = norm_mul1 * recip_scale;
                        prob_tmp = norm_mul2 >> 16;
                        if (prob_tmp > 16'd255)
                            prob_calc[i] = 8'hFF;
                        else
                            prob_calc[i] = prob_tmp[7:0];
                        prob_sum_calc = prob_sum_calc + {8'd0, prob_calc[i]};
                        if (prob_calc[i] >= prob_max_val) begin
                            prob_max_val = prob_calc[i];
                            prob_max_idx = i[3:0];
                        end
                    end

                    if (prob_sum_calc > 16'd255) begin
                        sum_delta = prob_sum_calc - 16'd255;
                        if (prob_calc[prob_max_idx] > sum_delta[7:0])
                            prob_calc[prob_max_idx] = prob_calc[prob_max_idx] - sum_delta[7:0];
                        else
                            prob_calc[prob_max_idx] = 8'd0;
                    end else if (prob_sum_calc < 16'd252) begin
                        sum_delta = 16'd252 - prob_sum_calc;
                        if ({8'd0, prob_calc[prob_max_idx]} + sum_delta > 16'd255)
                            prob_calc[prob_max_idx] = 8'hFF;
                        else
                            prob_calc[prob_max_idx] = prob_calc[prob_max_idx] + sum_delta[7:0];
                    end

                    for (i = 0; i < VECTOR_LEN; i = i + 1)
                        prob_out[i*8 +: 8] <= prob_calc[i];

                    cycle_cnt <= cycle_cnt + 1'b1;
                    state <= DONE_ST;
                end

                DONE_ST: begin
                    valid_out <= 1'b1;
                    cycles_used <= cycle_cnt;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
