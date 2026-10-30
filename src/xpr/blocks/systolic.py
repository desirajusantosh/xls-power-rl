"""INT8 4x4 weight-stationary systolic array (designs/systolic, a generated XLS proc network).

Same interface and semantics as xpr.blocks.matmul4 (input {load, row, x: s8[4]},
output s32[4]; weights before the word), so it reuses matmul4's packing, reference
model and stimulus suite. Two variants: without and with operand isolation.
"""

from __future__ import annotations

from types import SimpleNamespace

from . import matmul4 as _mm


def _variant(name: str, dslx: str, top: str) -> SimpleNamespace:
    return SimpleNamespace(
        NAME=name, DSLX=dslx, TOP=top, IN_W=_mm.IN_W, OUT_W=_mm.OUT_W, PROC_NETWORK=True,
        suite=_mm.suite, reference=_mm.reference, same=_mm.same,
    )


SYSTOLIC4 = _variant("systolic4", "designs/systolic/systolic4.x", "Systolic4")
SYSTOLIC4_ISO = _variant("systolic4_iso", "designs/systolic/systolic4_iso.x", "Systolic4Iso")
