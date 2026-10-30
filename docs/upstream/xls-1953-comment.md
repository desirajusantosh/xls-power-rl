<!-- Posted 2026-10-30: https://github.com/google/xls/issues/1953#issuecomment-6512141385 -->


Still reproduces on release `v0.0.0-10898-g8ddae8bf4` (2026-10-24). Some additional data that may help narrow it down:

**It isn't specific to `unroll_for!`.** In the DSLX interpreter, a `send(...)` that appears directly as an argument to `join(...)` is never delivered, including in straight-line code and when the joined token is used afterwards. Minimal repro, without `unroll_for!`:

```
proc Src {
    a: chan<u8> out;
    b: chan<u8> out;
    config(a: chan<u8> out, b: chan<u8> out) { (a, b) }
    init { u8:1 }
    next(n: u8) {
        join(send(join(), a, n), send(join(), b, n));   // neither send is delivered
        n + u8:1
    }
}

#[test_proc]
proc T {
    terminator: chan<bool> out;
    ra: chan<u8> in;
    rb: chan<u8> in;
    config(terminator: chan<bool> out) {
        let (sa, ra) = chan<u8>("a");
        let (sb, rb) = chan<u8>("b");
        spawn Src(sa, sb);
        (terminator, ra, rb)
    }
    init { () }
    next(state: ()) {
        let (t, x) = recv(join(), ra);
        let (t, y) = recv(t, rb);
        assert_eq(x, u8:1);
        assert_eq(y, u8:1);
        send(t, terminator, true);
    }
}
```

`interpreter_main` hits `DEADLINE_EXCEEDED`. Variants of `Src.next` (same test harness):

| Body of `Src.next` | Result |
|---|---|
| `let t = send(tok0, a, n); send(t, b, n);` | OK |
| `join(send(tok0, a, n), send(tok0, b, n));` | deadlock |
| `let t = join(send(tok0, a, n), send(tok0, b, n));` | deadlock |
| same, then `let _t2 = join(t);` (joined token used) | deadlock |
| `let t = join(tok0, send(tok0, a, n)); send(t, b, n);` | deadlock |
| `unroll_for!` body `join(t, send(t, a, n))`, then `send(t, b, n)` | deadlock |
| `unroll_for!` body `send(t, a, n)` (no join), then `send(t, b, n)` | OK |

**The IR is correct, so this is interpreter-only.** `ir_converter_main` keeps both sends for the join-wrapped case, before and after `opt_main`:

```
send.6: token = send(tok0, __state, predicate=literal.3, channel=_a, id=6)
send.7: token = send(tok0, __state, predicate=literal.3, channel=_b, id=7)
after_all.8: token = after_all(send.6, send.7, id=8)
```

So DSLX tests and the generated hardware disagree: a correct design can deadlock under `interpreter_main` (or, if the test doesn't depend on the dropped data, pass without checking it). We hit this while writing a systolic-array proc network in the style of `xls/examples/matmul_4x4`. The workaround is to bind each `send` to a `let` before joining.

A title like "DSLX interpreter drops `send()` used directly as an argument to `join()`" might help others find it.
