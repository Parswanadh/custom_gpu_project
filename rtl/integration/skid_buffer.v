module skid_buffer #(parameter DATA_WIDTH = 128) (
    input  clk, rst,
    input  valid_in,
    input  [DATA_WIDTH-1:0] data_in,
    input  ready_from_downstream,
    
    output ready_for_upstream,
    output valid_out,
    output [DATA_WIDTH-1:0] data_out
);

    reg [DATA_WIDTH-1:0] mem [0:1];
    reg [1:0] count;
    reg head;
    
    wire tail = head + count[0];

    assign ready_for_upstream = (count < 2);
    assign valid_out = (count > 0);
    assign data_out = mem[head];

    always @(posedge clk) begin
        if (rst) begin
            count <= 2'b0;
            head  <= 1'b0;
        end else begin
            case ({valid_in && ready_for_upstream, ready_from_downstream && valid_out})
                2'b10: begin // Push only
                    mem[tail] <= data_in;
                    count <= count + 1;
                end
                2'b01: begin // Pop only
                    head <= head + 1;
                    count <= count - 1;
                end
                2'b11: begin // Push and Pop
                    mem[tail] <= data_in;
                    head <= head + 1;
                    // count remains same
                end
                default: ; // Do nothing
            endcase
        end
    end

endmodule
