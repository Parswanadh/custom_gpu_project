// Minimal debug testbench for accelerated_attention
`timescale 1ns/1ps
module accelerated_attention_tb;
    parameter ED = 4;
    parameter DW = 16;
    parameter MSL = 8;

    reg clk, rst, valid_in;
    reg [ED*DW-1:0] x_in;
    reg [2:0] seq_pos;
    reg [ED*ED*DW-1:0] wq, wk, wv, wo;
    wire [ED*DW-1:0] y_out;
    wire valid_out;
    wire [31:0] zsc;

    accelerated_attention #(.EMBED_DIM(ED),.NUM_HEADS(1),.HEAD_DIM(ED),
        .MAX_SEQ_LEN(MSL),.DATA_WIDTH(DW)) uut (
        .clk(clk),.rst(rst),.valid_in(valid_in),.x_in(x_in),.seq_pos(seq_pos),
        .wq_flat(wq),.wk_flat(wk),.wv_flat(wv),.wo_flat(wo),
        .y_out(y_out),.valid_out(valid_out),.zero_skip_count(zsc));

    always #5 clk = ~clk;
    integer i, cycle;

    initial begin
        clk=0; rst=1; valid_in=0; x_in=0; seq_pos=0;
        // Simple diagonal weights = 1.0 in Q8.8 (256)
        wq=0; wk=0; wv=0; wo=0;
        for (i=0; i<ED; i=i+1) begin
            wq[(i*ED+i)*DW +: DW] = 16'sd256;
            wk[(i*ED+i)*DW +: DW] = 16'sd256;
            wv[(i*ED+i)*DW +: DW] = 16'sd256;
            wo[(i*ED+i)*DW +: DW] = 16'sd256;
        end

        #30; rst=0; #10;

        // Feed token 0: x = [256, 256, 256, 256] (all 1.0)
        $display("Feeding token 0: x=[256,256,256,256]");
        x_in = 0;
        x_in[0*DW +: DW] = 16'sd256;
        x_in[1*DW +: DW] = 16'sd256;
        x_in[2*DW +: DW] = 16'sd256;
        x_in[3*DW +: DW] = 16'sd256;
        seq_pos = 0;

        @(posedge clk); valid_in <= 1;
        @(posedge clk); valid_in <= 0;

        // Wait and monitor
        cycle = 0;
        while (!valid_out && cycle < 50) begin
            @(posedge clk);
            cycle = cycle + 1;
        end

        $display("Output after %0d cycles: y=[%0d,%0d,%0d,%0d] valid=%b",
            cycle,
            $signed(y_out[0*DW +: DW]), $signed(y_out[1*DW +: DW]),
            $signed(y_out[2*DW +: DW]), $signed(y_out[3*DW +: DW]),
            valid_out);

        // Wait one more cycle to see if output persists
        @(posedge clk);
        $display("Next cycle:               y=[%0d,%0d,%0d,%0d] valid=%b",
            $signed(y_out[0*DW +: DW]), $signed(y_out[1*DW +: DW]),
            $signed(y_out[2*DW +: DW]), $signed(y_out[3*DW +: DW]),
            valid_out);

        #50;

        // Feed token 1
        $display("\nFeeding token 1: x=[512,128,384,64]");
        x_in = 0;
        x_in[0*DW +: DW] = 16'sd512;
        x_in[1*DW +: DW] = 16'sd128;
        x_in[2*DW +: DW] = 16'sd384;
        x_in[3*DW +: DW] = 16'sd64;
        seq_pos = 1;

        @(posedge clk); valid_in <= 1;
        @(posedge clk); valid_in <= 0;

        cycle = 0;
        while (!valid_out && cycle < 50) begin
            @(posedge clk);
            cycle = cycle + 1;
        end

        $display("Output after %0d cycles: y=[%0d,%0d,%0d,%0d] valid=%b zsc=%0d",
            cycle,
            $signed(y_out[0*DW +: DW]), $signed(y_out[1*DW +: DW]),
            $signed(y_out[2*DW +: DW]), $signed(y_out[3*DW +: DW]),
            valid_out, zsc);

        $display("\n=== Test complete ===");
        $finish;
    end
endmodule
