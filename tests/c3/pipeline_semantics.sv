module Pipe(
  input  logic       clock,
  input  logic       reset,
  input  logic       io_in_valid,
  output logic       io_in_ready,
  input  logic [3:0] io_in_bits_set,
  input  logic [7:0] io_in_bits_tag,
  input  logic [3:0] req_set,
  output logic       blockB_s1
);
  logic       task_s1_valid;
  logic [3:0] task_s1_bits_set;
  logic [7:0] task_s1_bits_tag;
  logic       task_s2_valid;
  logic [3:0] task_s2_bits_set;
  logic [7:0] task_s2_bits_tag;
  logic       _blockB_T_1;
  logic       _blockB_T_2;

  assign io_in_ready = !task_s1_valid;
  assign _blockB_T_1 =
    task_s1_valid && (task_s1_bits_set == req_set);
  assign _blockB_T_2 =
    task_s2_valid && (task_s2_bits_set == req_set);
  assign blockB_s1 = _blockB_T_1 || _blockB_T_2;

  always_ff @(posedge clock) begin
    if (reset) begin
      task_s1_valid <= 1'b0;
      task_s2_valid <= 1'b0;
    end else begin
      task_s1_valid <= io_in_valid && io_in_ready;
      task_s1_bits_set <= io_in_bits_set;
      task_s1_bits_tag <= io_in_bits_tag;
      task_s2_valid <= task_s1_valid;
      task_s2_bits_set <= task_s1_bits_set;
      task_s2_bits_tag <= task_s1_bits_tag;
    end
  end
endmodule

module C3Top(
  input  logic       clock,
  input  logic       reset,
  input  logic       left_valid,
  input  logic       right_valid,
  input  logic [3:0] set,
  input  logic [7:0] tag,
  output logic       left_ready,
  output logic       right_ready,
  output logic       left_block,
  output logic       right_block
);
  logic [3:0] vec_0_tag;
  logic [3:0] vec_1_tag;

  Pipe left (
    .clock(clock),
    .reset(reset),
    .io_in_valid(left_valid),
    .io_in_ready(left_ready),
    .io_in_bits_set(set),
    .io_in_bits_tag(tag),
    .req_set(set),
    .blockB_s1(left_block)
  );
  Pipe right (
    .clock(clock),
    .reset(reset),
    .io_in_valid(right_valid),
    .io_in_ready(right_ready),
    .io_in_bits_set(set),
    .io_in_bits_tag(tag),
    .req_set(set),
    .blockB_s1(right_block)
  );
endmodule
