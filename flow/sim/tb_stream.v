// Testbench for a single-input, single-output XLS streaming proc
// (ports: clk, rst, _in_r, _in_r_vld, _in_r_rdy, _out_s, _out_s_vld, _out_s_rdy).
//
// Defines: DUT (module name), IN_W, OUT_W, HALF_PS (half clock period; must match the SDC).
// Plusargs:
//   +stim=<file>     one hex word per cycle: {rst, vld, data[IN_W-1:0]}
//   +cycles=<n>      number of stimulus cycles
//   +win_start=<c>   first cycle dumped to VCD
//   +win_end=<c>     simulation ends at this cycle (exclusive)
//   +vcd=<file>      VCD of the DUT's top-level nets, window only
//   +log=<file>      per-cycle log: cycle in_rdy out_vld out_data
//   +clk_stop        hold the clock low inside the window (leakage run)
// Inputs change on the falling edge, so zero-delay sim has no races.
`timescale 1ps/1ps

module tb;
  localparam integer HALF = `HALF_PS;
  localparam integer W = `IN_W + 2;

  reg clk = 0, rst = 1, vld = 0;
  reg [`IN_W-1:0] data = 0;
  wire in_rdy, out_vld;
  wire [`OUT_W-1:0] out_data;

  `DUT dut (
    .clk(clk), .rst(rst),
    ._in_r(data), ._in_r_vld(vld), ._in_r_rdy(in_rdy),
    ._out_s(out_data), ._out_s_vld(out_vld), ._out_s_rdy(1'b1)
  );

  reg [W-1:0] stim [0:1048575];
  reg [8*256-1:0] stim_f, vcd_f, log_f;
  integer cycles, win_start, win_end, clk_stop, fd, c;
  reg clk_en = 1;

  always #HALF if (clk_en) clk = ~clk;

  initial begin
    if (!$value$plusargs("stim=%s", stim_f)) $fatal(1, "missing +stim");
    if (!$value$plusargs("cycles=%d", cycles)) $fatal(1, "missing +cycles");
    if (!$value$plusargs("win_start=%d", win_start)) win_start = 0;
    if (!$value$plusargs("win_end=%d", win_end)) win_end = cycles;
    if (!$value$plusargs("vcd=%s", vcd_f)) $fatal(1, "missing +vcd");
    if (!$value$plusargs("log=%s", log_f)) $fatal(1, "missing +log");
    clk_stop = $test$plusargs("clk_stop");
    $readmemh(stim_f, stim, 0, cycles - 1);
    fd = $fopen(log_f, "w");

    // With period T: clock rises at Tc+T/2 and falls at Tc. Cycle c: drive at Tc,
    // sample 1 ps before the rising edge, then let the edge happen.
    for (c = 0; c < win_end; c = c + 1) begin
      {rst, vld, data} = stim[c];
      if (c == win_start) begin
        if (clk_stop) begin clk_en = 0; clk = 0; end
        $dumpfile(vcd_f);
        $dumpvars(1, dut);
      end
      #(HALF - 1);
      $fwrite(fd, "%0d %b %b %h\n", c, in_rdy, out_vld, out_data);
      #(HALF + 1);
    end
    $fclose(fd);
    $finish;
  end
endmodule
