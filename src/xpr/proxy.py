"""Fast toggle proxy: a Verilator model of the DUT in a long-lived process.

  p = Proxy.build(block, run_dir, top, level="rtl")   # or level="synth"
  p.evaluate([words, ...], win_start)                # -> list[ProxyResult]

level="rtl" compiles XLS's Verilog; level="synth" compiles the synthesized
netlist (the one xpr.gatesim simulates at synth) with the ASAP7 cell models.
`words` are full per-cycle {rst, vld, data} words including the reset prefix.
alpha = toggles per non-clock bit of the DUT's top-level nets per window cycle,
matching the Icarus VCD measurement in xpr.gatesim at the same level
(checked by scripts/proxy_calibrate.py).
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from . import gatesim

REPO = Path(__file__).resolve().parents[2]
LEVELS = ("rtl", "synth")
_UDP = re.compile(r"^\s*primitive\b.*?^\s*endprimitive\b", re.S | re.M)


@dataclass(frozen=True)
class ProxyResult:
    alpha: float
    toggles: int
    bits: int
    window_cycles: int
    vld_not_rdy: int  # cycles in the window with valid high and ready low
    out_vld: int  # output-valid cycles in the window
    per_cycle: tuple[int, ...] | None = None  # toggles per window cycle (per_cycle=True)


def verilator_cell_models(out: Path) -> list[Path]:
    """Copies of the ASAP7 models without the sequential UDPs appended to each file.

    The netlists only instantiate DFFHQNx1, which dff.v models behaviorally; the
    UDPs are unused and Verilator does not accept them.
    """
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in gatesim.CELL_MODELS:
        dst = out / name
        dst.write_text(_UDP.sub("", (gatesim.MODELS / name).read_text()))
        paths.append(dst)
    return paths


class Proxy:
    def __init__(self, binary: Path, level: str, per_cycle: bool = False):
        self.binary = binary
        self.level = level
        env = dict(os.environ)
        if per_cycle:
            env["PROXY_PERCYCLE"] = "1"
        self.proc = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=env,
        )
        self.evals = 0

    @classmethod
    def build(cls, block: ModuleType, run_dir: Path, top: str, level: str = "rtl",
              per_cycle: bool = False) -> "Proxy":
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}")
        out = run_dir / f"proxy_build_{level}"
        binary = out / "Vdut"
        src = REPO / "flow/verilator/proxy.cpp"
        if level == "rtl":
            sources = [gatesim.rtl_source(run_dir, top)]
        else:
            gatesim.prepare(run_dir, top)
            sources = [run_dir / "sim" / "1_synth_sim.v"]
        stale = not binary.exists() or binary.stat().st_mtime < max(p.stat().st_mtime for p in [*sources, src])
        if stale:
            if level == "synth":
                sources += verilator_cell_models(out / "cells")
            subprocess.run(
                ["verilator", "--cc", "--exe", "--build", "-O3", "--public-flat-rw", "-Wno-fatal",
                 "-Wno-lint", "-Wno-style", "--prefix", "Vdut", "--top-module", top, "-Mdir", str(out),
                 "-CFLAGS", f"-O2 -DIN_W={block.IN_W}", *map(str, sources), str(src)],
                check=True, capture_output=True, text=True,
            )
        return cls(binary, level, per_cycle)

    def evaluate(self, sequences, win_start: int) -> list[ProxyResult]:
        """One request line per sequence; responses come back in order."""
        lines = [f"{len(s)} {win_start} " + " ".join(f"{int(w):x}" for w in s) for s in sequences]
        self.proc.stdin.write("\n".join(lines) + "\n")
        self.proc.stdin.flush()
        results = []
        for _ in sequences:
            head, _, tail = self.proc.stdout.readline().partition("|")
            t, b, c, vnr, ov = map(int, head.split())
            per_cycle = tuple(map(int, tail.split())) if tail else None
            results.append(ProxyResult(t / (b * c) if b and c else 0.0, t, b, c, vnr, ov, per_cycle))
        self.evals += len(sequences)
        return results

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
