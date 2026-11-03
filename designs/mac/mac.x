// INT8 multiply-accumulate. Original design (ref: xls/examples/dot_product.x).
//
// `mac` is the combinational datapath. `Mac` is a streaming proc: each accepted
// input adds a*b to a 32-bit accumulator; an input with `last` set emits the
// final sum and clears the accumulator. 2^16 worst-case products fit in s32.

pub struct MacIn { a: s8, b: s8, last: bool }

pub fn mac(a: s8, b: s8, acc: s32) -> s32 {
    acc + (a as s32) * (b as s32)
}

#[test]
fn mac_extremes() {
    assert_eq(mac(s8:-128, s8:-128, s32:0), s32:16384);
    assert_eq(mac(s8:-128, s8:127, s32:0), s32:-16256);
    assert_eq(mac(s8:3, s8:-4, s32:10), s32:-2);
}

pub proc Mac {
    in_r: chan<MacIn> in;
    out_s: chan<s32> out;

    config(in_r: chan<MacIn> in, out_s: chan<s32> out) { (in_r, out_s) }

    init { s32:0 }

    next(acc: s32) {
        let (tok, x) = recv(join(), in_r);
        let sum = mac(x.a, x.b, acc);
        send_if(tok, out_s, x.last, sum);
        if x.last { s32:0 } else { sum }
    }
}

#[test_proc]
proc MacTest {
    terminator: chan<bool> out;
    in_s: chan<MacIn> out;
    out_r: chan<s32> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<MacIn>("in");
        let (out_s, out_r) = chan<s32>("out");
        spawn Mac(in_r, out_s);
        (terminator, in_s, out_r)
    }

    init { () }

    next(state: ()) {
        // [1, 2, 3] . [4, -5, 6] = 12
        let tok = send(join(), in_s, MacIn { a: s8:1, b: s8:4, last: false });
        let tok = send(tok, in_s, MacIn { a: s8:2, b: s8:-5, last: false });
        let tok = send(tok, in_s, MacIn { a: s8:3, b: s8:6, last: true });
        let (tok, v) = recv(tok, out_r);
        assert_eq(v, s32:12);

        // Accumulator clears after `last`: a fresh 2-element dot product.
        let tok = send(tok, in_s, MacIn { a: s8:-128, b: s8:-128, last: false });
        let tok = send(tok, in_s, MacIn { a: s8:-128, b: s8:-128, last: true });
        let (tok, v) = recv(tok, out_r);
        assert_eq(v, s32:32768);

        send(tok, terminator, true);
    }
}
