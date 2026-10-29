"""Per-block definitions: word layout, stimulus suite, reference model.

Each block module exposes:
  NAME, DSLX, TOP     source and XLS top proc
  IN_W, OUT_W         input/output word widths of the streaming interface
  suite(n)            list[Stimulus] covering leak / idle / average / worst
  reference(stim)     expected output words, in order
  same(got, want)     output equality (e.g. any NaN matches any NaN)
"""

from . import bf16_mul, mac, matmul4

BLOCKS = {m.NAME: m for m in (mac, bf16_mul, matmul4)}
