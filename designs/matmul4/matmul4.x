// INT8 4x4 weight-stationary matrix-vector engine.
//
// Adapted from XLS's xls/examples/matmul_4x4/matmul_4x4.x (Apache-2.0), a float32
// systolic array of 16 FMA nodes with weights loaded once. Changes (original work):
// INT8 operands with s32 accumulation; a single proc holding the weights as state
// instead of a 16-node proc network; weights reloadable at any time through the
// same input stream, so a stimulus controls both weights and activations.
//
// Same dataflow as the example: activations enter per row, partial sums flow down
// columns, so y[c] = sum_r x[r] * W[r][c].
//
// Input word: { load: bool, row: u2, x: s8[4] }
//   load = 1: W[row] <- x (one weight row per word); no output.
//   load = 0: x is an activation vector; emits y = x . W as s32[4].

pub struct MmIn { load: bool, row: u2, x: s8[4] }

type Weights = s8[4][4];

pub fn matvec(x: s8[4], w: Weights) -> s32[4] {
    for (c, y): (u32, s32[4]) in u32:0..u32:4 {
        let col = for (r, acc): (u32, s32) in u32:0..u32:4 {
            acc + (x[r] as s32) * (w[r][c] as s32)
        }(s32:0);
        update(y, c, col)
    }(s32[4]:[0, ...])
}

#[test]
fn matvec_identity_and_extremes() {
    let eye = Weights:[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    assert_eq(matvec(s8[4]:[1, -2, 3, -4], eye), s32[4]:[1, -2, 3, -4]);
    let min = Weights:[[-128, ...], [-128, ...], [-128, ...], [-128, ...]];
    assert_eq(matvec(s8[4]:[-128, ...], min), s32[4]:[65536, ...]);
    let w = Weights:[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]];
    // y[c] = sum_r x[r] * w[r][c] with x = [1, 1, 1, 1]: column sums.
    assert_eq(matvec(s8[4]:[1, ...], w), s32[4]:[28, 32, 36, 40]);
}

pub proc Matmul4 {
    in_r: chan<MmIn> in;
    out_s: chan<s32[4]> out;

    config(in_r: chan<MmIn> in, out_s: chan<s32[4]> out) { (in_r, out_s) }

    init { Weights:[[0, ...], [0, ...], [0, ...], [0, ...]] }

    next(w: Weights) {
        let (tok, m) = recv(join(), in_r);
        send_if(tok, out_s, !m.load, matvec(m.x, w));
        if m.load { update(w, m.row, m.x) } else { w }
    }
}

#[test_proc]
proc Matmul4Test {
    terminator: chan<bool> out;
    in_s: chan<MmIn> out;
    out_r: chan<s32[4]> in;

    config(terminator: chan<bool> out) {
        let (in_s, in_r) = chan<MmIn>("in");
        let (out_s, out_r) = chan<s32[4]>("out");
        spawn Matmul4(in_r, out_s);
        (terminator, in_s, out_r)
    }

    init { () }

    next(state: ()) {
        // Load W = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]] row by row.
        let tok = send(join(), in_s, MmIn { load: true, row: u2:0, x: s8[4]:[1, 2, 3, 4] });
        let tok = send(tok, in_s, MmIn { load: true, row: u2:1, x: s8[4]:[5, 6, 7, 8] });
        let tok = send(tok, in_s, MmIn { load: true, row: u2:2, x: s8[4]:[9, 10, 11, 12] });
        let tok = send(tok, in_s, MmIn { load: true, row: u2:3, x: s8[4]:[13, 14, 15, 16] });
        // x = [1, 0, 0, -1]: y = W[0] - W[3] = [-12, -12, -12, -12].
        let tok = send(tok, in_s, MmIn { load: false, row: u2:0, x: s8[4]:[1, 0, 0, -1] });
        let (tok, y) = recv(tok, out_r);
        assert_eq(y, s32[4]:[-12, ...]);
        // Reload row 0 only, then x = [1, 0, 0, 0] reads it back.
        let tok = send(tok, in_s, MmIn { load: true, row: u2:0, x: s8[4]:[-128, 127, 0, -1] });
        let tok = send(tok, in_s, MmIn { load: false, row: u2:0, x: s8[4]:[1, 0, 0, 0] });
        let (tok, y) = recv(tok, out_r);
        assert_eq(y, s32[4]:[-128, 127, 0, -1]);
        send(tok, terminator, true);
    }
}
