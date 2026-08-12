module C2Top(
  input  logic       clock,
  input  logic       reset,
  input  logic       active,
  input  logic       clear,
  input  logic [7:0] data,
  output logic [7:0] observed
);
  logic [7:0] timer;
  logic [7:0] _T_1;
  logic [7:0] _GEN_0;

  assign _T_1 = data;
  assign _GEN_0 = active ? _T_1 : 8'h00;
  assign observed = _GEN_0;

  always_ff @(posedge clock) begin
    if (reset) begin
      timer <= 8'h00;
    end else if (clear) begin
      timer <= 8'h00;
    end else if (active) begin
      timer <= timer + 8'h01;
    end
  end
endmodule
