"""Run XLS DSLX -> IR -> opt -> pipelined Verilog, with a provenance manifest.

Every build writes `manifest.json` next to its outputs: XLS release, exact
command lines, and sha256 of every input and output. Results downstream cite
the manifest, so any number can be traced back to the RTL it came from.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

XLS_ROOT = Path(os.environ.get("XLS_ROOT", Path.home() / "opt" / "xls"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xls_release(root: Path = XLS_ROOT) -> str:
    """Release tag, taken from the versioned directory the `xls` symlink points to."""
    name = root.resolve().name
    return name.removeprefix("xls-")


@dataclass(frozen=True)
class BuildSpec:
    dslx: Path
    top: str
    stages: int
    delay_model: str = "asap7"
    reset: str = "rst"

    @property
    def module(self) -> str:
        return f"{self.top.lower()}_p{self.stages}"


def _run(cmd: list[str], stdout: Path | None = None) -> None:
    with open(stdout, "w") if stdout else open(os.devnull, "w") as out:
        subprocess.run(cmd, check=True, stdout=out, stderr=subprocess.PIPE, text=True)


def build(spec: BuildSpec, out_dir: Path, root: Path = XLS_ROOT) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stdlib = root / "xls" / "dslx" / "stdlib"
    ir, opt, v = out_dir / f"{spec.top}.ir", out_dir / f"{spec.top}.opt.ir", out_dir / f"{spec.module}.v"
    sig = out_dir / f"{spec.module}.sig.textproto"
    cmds = [
        ([str(root / "ir_converter_main"), f"--dslx_stdlib_path={stdlib}", f"--top={spec.top}", str(spec.dslx)], ir),
        ([str(root / "opt_main"), str(ir)], opt),
        (
            [
                str(root / "codegen_main"), str(opt),
                "--generator=pipeline",
                f"--delay_model={spec.delay_model}",
                f"--pipeline_stages={spec.stages}",
                f"--module_name={spec.module}",
                f"--reset={spec.reset}",
                "--use_system_verilog=false",
                "--flop_inputs", "--flop_outputs",
                f"--output_signature_path={sig}",
            ],
            v,
        ),
    ]
    for cmd, dst in cmds:
        _run(cmd, dst)

    manifest = {
        "xls_release": xls_release(root),
        "spec": {k: str(val) for k, val in vars(spec).items()},
        "commands": [cmd for cmd, _ in cmds],
        "inputs": {str(spec.dslx): sha256(spec.dslx)},
        "outputs": {p.name: sha256(p) for p in (ir, opt, v, sig)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
