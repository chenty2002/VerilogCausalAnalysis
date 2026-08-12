module Leaf(
  input  logic in,
  output logic out
);
  logic _T_1;
  assign _T_1 = in;
  assign out = _T_1;
endmodule

module Top(
  input  logic clock,
  input  logic a,
  input  logic b,
  output logic y0,
  output logic y1
);
  Leaf left (
    .in  (a),
    .out (y0)
  );
  Leaf right (
    .in  (b),
    .out (y1)
  );
endmodule
