import json
import shutil
import subprocess
from pathlib import Path

import pytest

from xpr.xlsflow import XLS_ROOT, BuildSpec, build

REPO = Path(__file__).resolve().parents[1]
needs_xls = pytest.mark.skipif(not (XLS_ROOT / "codegen_main").exists(), reason="XLS not installed")


@needs_xls
@pytest.mark.parametrize("stages", [1, 3])
def test_mac_builds_with_manifest(tmp_path, stages):
    spec = BuildSpec(REPO / "designs" / "mac" / "mac.x", top="Mac", stages=stages)
    m = build(spec, tmp_path)
    v = tmp_path / f"mac_p{stages}.v"
    assert v.exists() and f"module mac_p{stages}" in v.read_text()
    assert json.loads((tmp_path / "manifest.json").read_text()) == m
    assert m["xls_release"].startswith("v0.0.0-")
    assert set(m["outputs"]) == {"Mac.ir", "Mac.opt.ir", v.name, f"mac_p{stages}.sig.textproto"}


@needs_xls
@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_mac_verilog_lints(tmp_path):
    build(BuildSpec(REPO / "designs" / "mac" / "mac.x", top="Mac", stages=2), tmp_path)
    subprocess.run(["verilator", "--lint-only", "-Wall", "-Wno-fatal", str(tmp_path / "mac_p2.v")], check=True)
