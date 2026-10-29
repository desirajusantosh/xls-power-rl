// bf16 multiplier. Datapath is stock XLS (`bfloat16::mul` -> `apfloat::mul`);
// only the streaming wrapper is original.
//
// apfloat::mul semantics (from its doc comment): subnormal inputs and outputs
// are flushed to zero, round-to-nearest-even only, NaN fraction not specified.
//
// Input word: {a[15:0], b[15:0]} (struct fields flatten MSB first).

import bfloat16;

type BF16 = bfloat16::BF16;

pub struct MulIn { a: BF16, b: BF16 }

pub proc Bf16Mul {
    in_r: chan<MulIn> in;
    out_s: chan<BF16> out;

    config(in_r: chan<MulIn> in, out_s: chan<BF16> out) { (in_r, out_s) }

    init { () }

    next(state: ()) {
        let (tok, x) = recv(join(), in_r);
        send(tok, out_s, bfloat16::mul(x.a, x.b));
    }
}

fn bf(raw: u16) -> BF16 { bfloat16::unflatten(raw) }

#[test_proc]
proc Bf16MulTest {
    terminator: chan<bool> out;
    in_s: chan<MulIn> out;
    out_r: chan<BF16> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<MulIn>("in");
        let (out_s, out_r) = chan<BF16>("out");
        spawn Bf16Mul(in_r, out_s);
        (terminator, in_s, out_r)
    }

    init { () }

    next(state: ()) {
        // 1.5 * -2.0 = -3.0
        let tok = send(join(), in_s, MulIn { a: bf(u16:0x3fc0), b: bf(u16:0xc000) });
        let (tok, y) = recv(tok, out_r);
        assert_eq(bfloat16::flatten(y), u16:0xc040);

        // Smallest normal * 0.5 underflows to a subnormal -> flushed to +0.
        let tok = send(tok, in_s, MulIn { a: bf(u16:0x0080), b: bf(u16:0x3f00) });
        let (tok, y) = recv(tok, out_r);
        assert_eq(bfloat16::flatten(y), u16:0x0000);

        // Subnormal input is treated as zero: subnormal * 2^100 -> 0.
        let tok = send(tok, in_s, MulIn { a: bf(u16:0x0001), b: bf(u16:0x7180) });
        let (tok, y) = recv(tok, out_r);
        assert_eq(bfloat16::flatten(y), u16:0x0000);

        // inf * 0 -> NaN.
        let tok = send(tok, in_s, MulIn { a: bf(u16:0x7f80), b: bf(u16:0x0000) });
        let (tok, y) = recv(tok, out_r);
        assert_eq(bfloat16::is_nan(y), true);

        send(tok, terminator, true);
    }
}
