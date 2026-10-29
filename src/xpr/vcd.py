"""Minimal VCD reader: per-bit toggle counts for the signals of one scope."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Toggles:
    counts: dict[str, int]  # one entry per bit: "net" or "net[i]"

    def alpha(self, cycles: int, exclude: Callable[[str], bool] = lambda name: False) -> float:
        """Mean toggles per bit per clock cycle, over bits not excluded."""
        bits = [c for n, c in self.counts.items() if not exclude(n)]
        if not bits or cycles <= 0:
            return 0.0
        return sum(bits) / (len(bits) * cycles)


def _bit_names(ref: str, width: int, rng: str | None) -> list[str]:
    # 1-bit nets keep their reference verbatim: gate-level escaped names such as
    # "\state[0]$_SDFFE_" contain brackets that are not a range.
    if width == 1:
        return [ref + (rng or "")]
    if rng is None:
        raise ValueError(f"{width}-bit var {ref!r} has no range")
    hi, lo = (int(x) for x in rng.strip("[]").split(":"))
    step = -1 if hi >= lo else 1
    return [f"{ref}[{i}]" for i in range(hi, lo + step, step)]


def read_toggles(path: Path) -> Toggles:
    """Count 0<->1 transitions per bit. Changes to or from X/Z are not toggles."""
    ids: dict[str, list[str]] = {}  # id code -> per-bit names, MSB first
    last: dict[str, str] = {}
    counts: dict[str, int] = {}
    tokens = iter(Path(path).read_text().split())

    def until_end() -> list[str]:
        return list(iter(lambda: next(tokens), "$end"))

    for tok in tokens:
        if tok == "$var":
            _kind, width, code, ref, *rest = until_end()
            names = _bit_names(ref, int(width), rest[0] if rest else None)
            ids.setdefault(code, names)
            for n in names:
                counts.setdefault(n, 0)
        elif tok == "$enddefinitions":
            until_end()
            break
        elif tok.startswith("$") and tok != "$end":
            until_end()

    def change(code: str, value: str) -> None:
        names = ids.get(code)
        if names is None:
            return
        pad = "0" if value[0] in "01" else value[0]
        value = value.rjust(len(names), pad)[-len(names):]
        for name, v in zip(names, value):
            prev = last.get(name)
            if prev is not None and prev in "01" and v in "01" and prev != v:
                counts[name] += 1
            last[name] = v

    for tok in tokens:
        c = tok[0]
        if c == "#" or c == "$":
            continue
        if c in "bB":
            change(next(tokens), tok[1:].lower())
        elif c in "rR":
            next(tokens)
        else:
            change(tok[1:], c.lower())
    return Toggles(counts)
