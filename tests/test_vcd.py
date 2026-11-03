import pytest

from xpr.vcd import read_toggles

VCD = """\
$timescale 1ps $end
$scope module dut $end
$var wire 1 ! clk $end
$var wire 4 " data [3:0] $end
$var wire 1 # \\state[0]$_SDFFE_ $end
$upscope $end
$enddefinitions $end
#0
$dumpvars
0!
b0000 "
x#
$end
#250
1!
b101 "
0#
#500
0!
b0101 "
1#
#750
1!
bx "
#1000
0!
b0 "
"""


@pytest.fixture
def toggles(tmp_path):
    p = tmp_path / "t.vcd"
    p.write_text(VCD)
    return read_toggles(p)


def test_counts_per_bit(toggles):
    c = toggles.counts
    assert c["clk"] == 4
    # 0000 -> 0101 (bits 2 and 0 rise) -> 0101 -> xxxx -> 0000 (X is not a toggle)
    assert [c[f"data[{i}]"] for i in (3, 2, 1, 0)] == [0, 1, 0, 1]


def test_escaped_name_with_brackets_is_one_bit(toggles):
    # x -> 0 is not a toggle; 0 -> 1 is.
    assert toggles.counts["\\state[0]$_SDFFE_"] == 1


def test_alpha_excludes_clock(toggles):
    # 3 toggles over 5 non-clock bits and 2 cycles.
    assert toggles.alpha(cycles=2, exclude=lambda n: n == "clk") == pytest.approx(3 / 10)
