"""Read results from ORFS runs (scripts/orfs_run.sh, scripts/orfs_power.sh)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_LINE = re.compile(r"^XPR_POWER\s+(.*)$", re.M)


@dataclass(frozen=True)
class PowerPoint:
    stage: str
    label: str  # vectorless input activity ("0.1") or a VCD/stimulus name
    internal_w: float
    switching_w: float
    leakage_w: float
    total_w: float

    @property
    def activity(self) -> float:
        """Vectorless input activity; only valid for vectorless points."""
        return float(self.label)


def parse_power(text: str) -> list[PowerPoint]:
    """Parse the XPR_POWER lines that flow/power.tcl prints."""
    points = []
    for body in _LINE.findall(text):
        kv = dict(item.split("=", 1) for item in body.split())
        points.append(
            PowerPoint(
                stage=kv["stage"],
                label=kv["activity"],
                internal_w=float(kv["internal"]),
                switching_w=float(kv["switching"]),
                leakage_w=float(kv["leakage"]),
                total_w=float(kv["total"]),
            )
        )
    return points


def read_power(run_dir: Path, stage: str) -> list[PowerPoint]:
    return parse_power((run_dir / f"power_{stage}.log").read_text())
